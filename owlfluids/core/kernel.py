"""The compiled time-stepping kernel.

State vector ``x``: for every non-boundary node its mass ``m`` and total internal
energy ``U`` (plus wall temperature when the node has a wall), and for every
pipe link its mass flow. Orifices, valves and fittings are quasi-steady: their
flow is an algebraic function of the node states on either side.

Each step is linearly implicit (backward Euler with one Newton iteration):

    (I - dt J) dx = dt f(x)

``J`` is built by finite differences, perturbing groups of structurally
independent states together (graph colouring), and the system is solved with a
banded LU after the unknowns were reordered with reverse Cuthill-McKee.
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

from ..fluids.props import saturation, state
from ..fluids.registry import L_PSAT, MODEL_IDEAL, MODEL_LIQUID
from .layout import (
    CTRL_CHECK, CTRL_REGULATOR, CTRL_RELIEF, CTRL_VALVE, DRAW_LIQUID, DRAW_MIX, DRAW_VAPOR,
    L_A, L_B, L_CTRL, L_DRAWA, L_DRAWB, L_KIND, L_SX,
    LF_AREA, LF_CDA, LF_DIAM, LF_K, LF_LEN, LF_P1, LF_P2, LF_ROUGH,
    LINK_PIPE, LO_MDOT, LO_OPEN,
    N_FLUID, N_KIND, N_PARTNER, N_SX, N_WALL,
    NF_AREA, NF_BP, NF_BRHO, NF_BT, NF_BU, NF_HA, NF_HAAMB, NF_MREF, NF_RHOREF, NF_TAMB, NF_VOL, NF_WALLMC,
    NODE_BOUNDARY, NODE_TANK_LIQUID, NODE_VOLUME,
    O_H, O_M, O_P, O_RHO, O_T, O_TW, O_X,
    V_CMD, V_POS, V_RATE_CLOSE, V_RATE_OPEN,
)

jit = nb.njit(cache=True, nogil=True)

G0 = 9.80665
DP_LIN = 100.0  # Pa; below this pressure difference flow is linear in dP (keeps the Jacobian finite)
R_LIN = 1e-3  # gas pressure ratio band near 1 treated the same way
CHOKE_TAU = 1e-3  # s; relaxation time that holds pipe gas flow at the choked limit

# scratch columns
W_P, W_T, W_X, W_RHO, W_U, W_H, W_G, W_MU, W_V, W_TW, W_S = range(11)
N_W = 11


@jit
def _orifice_flow(cda, p_up, p_dn, rho, gam, liquid, psat):
    """Mass flow (>= 0) through an effective area ``cda`` from p_up to p_dn."""
    if cda <= 0.0 or p_up <= p_dn:
        return 0.0
    dp = p_up - p_dn
    if liquid:
        spi = cda * math.sqrt(2.0 * rho * max(dp, DP_LIN))
        if dp < DP_LIN:
            spi *= dp / DP_LIN
        if psat > 0.0 and p_dn < psat:
            # flashing flow: Dyer-style blend of single-phase incompressible flow and a
            # cavitation-limited estimate of the equilibrium (HEM) flow
            k = math.sqrt(dp / max(psat - p_dn, 1.0))
            hem = cda * math.sqrt(2.0 * rho * max(p_up - psat, 0.0))
            if hem > spi:
                hem = spi
            return (k / (1.0 + k)) * spi + (1.0 / (1.0 + k)) * hem
        return spi
    g = gam
    if g < 1.01:
        g = 1.01
    r = p_dn / p_up
    rc = (2.0 / (g + 1.0)) ** (g / (g - 1.0))
    if r <= rc:
        return cda * math.sqrt(g * rho * p_up * (2.0 / (g + 1.0)) ** ((g + 1.0) / (g - 1.0)))
    rr = r
    if 1.0 - r < R_LIN:
        rr = 1.0 - R_LIN
    val = cda * math.sqrt(max(2.0 * rho * p_up * g / (g - 1.0) * (rr ** (2.0 / g) - rr ** ((g + 1.0) / g)), 0.0))
    if rr != r:
        val *= (1.0 - r) / R_LIN
    return val


@jit
def _choked_flux(p_up, rho, gam):
    g = max(gam, 1.01)
    return math.sqrt(g * rho * p_up * (2.0 / (g + 1.0)) ** ((g + 1.0) / (g - 1.0)))


@jit
def _friction_r(mdot, rho, mu, area, diam, length, rough, k_minor):
    """Pressure drop (signed with mdot) along a pipe of the given size."""
    am = abs(mdot)
    re = am * diam / (area * mu)
    if re < 2000.0:
        # laminar: dP = 32 mu L v / D^2, written to stay finite at zero flow
        lam = 32.0 * mu * length * mdot / (rho * area * diam * diam)
        return lam + k_minor * mdot * am / (2.0 * rho * area * area)
    e = rough / diam
    a = (2.457 * math.log(1.0 / ((7.0 / re) ** 0.9 + 0.27 * e))) ** 16
    b = (37530.0 / re) ** 16
    f = 8.0 * ((8.0 / re) ** 12 + 1.0 / (a + b) ** 1.5) ** (1.0 / 12.0)
    return (f * length / diam + k_minor) * mdot * am / (2.0 * rho * area * area)


@jit
def _node_props(x, NI, NF, fl, W):
    model, par, tab, grid, ltab, lgrid, rhol, sat, satg = fl
    n = NI.shape[0]
    for i in range(n):
        kind = NI[i, N_KIND]
        f = NI[i, N_FLUID]
        if kind == NODE_BOUNDARY:
            rho = NF[i, NF_BRHO]
            u = NF[i, NF_BU]
            p, t, xq, g, mu = state(f, rho, u, model, par, tab, grid, ltab, lgrid, rhol)
            W[i, W_P] = NF[i, NF_BP]
            W[i, W_T] = NF[i, NF_BT]
            W[i, W_V] = 0.0
            W[i, W_TW] = NF[i, NF_BT]
            W[i, W_S] = 1.0
        else:
            sx = NI[i, N_SX]
            m = max(x[sx], 1e-12)
            u = x[sx + 1] / m
            if kind == NODE_TANK_LIQUID:
                rho = NF[i, NF_RHOREF]
                vol = m / rho
                # fraction of an outflow that is still liquid; the rest is ullage gas
                W[i, W_S] = min(max(m / NF[i, NF_MREF], 0.0), 1.0)
            else:
                vol = NF[i, NF_VOL]
                pt = NI[i, N_PARTNER]
                if pt >= 0:
                    vol = vol - max(x[NI[pt, N_SX]], 0.0) / NF[pt, NF_RHOREF]
                    vol = max(vol, 1e-4 * NF[i, NF_VOL])
                rho = m / vol
                # outflow throttle near vacuum (see NF_MREF)
                W[i, W_S] = m / (m + NF[i, NF_MREF])
            p, t, xq, g, mu = state(f, rho, u, model, par, tab, grid, ltab, lgrid, rhol)
            W[i, W_P] = p
            W[i, W_T] = t
            W[i, W_V] = vol
            W[i, W_TW] = x[sx + 2] if NI[i, N_WALL] else NF[i, NF_TAMB]
        W[i, W_X] = xq
        W[i, W_RHO] = rho
        W[i, W_U] = u
        W[i, W_G] = g
        W[i, W_MU] = mu
    # tank liquid takes its pressure from the ullage above it, plus liquid head
    for i in range(n):
        if NI[i, N_KIND] == NODE_TANK_LIQUID:
            pt = NI[i, N_PARTNER]
            head = 0.0
            if NF[i, NF_AREA] > 0.0:
                head = W[i, W_RHO] * G0 * W[i, W_V] / NF[i, NF_AREA]
            W[i, W_P] = W[pt, W_P] + head
    for i in range(n):
        W[i, W_H] = W[i, W_U] + W[i, W_P] / W[i, W_RHO]


@jit
def _draw(i, d, NI, W, fl):
    """Properties of what flows out of node i through a port that draws ``d``.

    Returns (rho, h, mu, gamma, liquid_like, psat).
    """
    model, par, tab, grid, ltab, lgrid, rhol, sat, satg = fl
    f = NI[i, N_FLUID]
    mdl = model[f]
    t = W[i, W_T]
    xq = W[i, W_X]
    if mdl == MODEL_LIQUID:
        return W[i, W_RHO], W[i, W_H], W[i, W_MU], 1.0, True, par[f, L_PSAT]
    if mdl == MODEL_IDEAL:
        return W[i, W_RHO], W[i, W_H], W[i, W_MU], W[i, W_G], False, 0.0
    tc = satg[f, 2]
    if NI[i, N_KIND] == NODE_TANK_LIQUID:
        psat = 0.0
        if t < tc:
            psat = saturation(f, t, model, par, sat, satg)[0]
        return W[i, W_RHO], W[i, W_H], W[i, W_MU], W[i, W_G], True, psat
    if 0.0 < xq < 1.0 and d != DRAW_MIX:
        ps, rl, rv, hl, hv, mul, muv, gv, sub = saturation(f, t, model, par, sat, satg)
        if d == DRAW_LIQUID:
            return rl, hl, mul, gv, True, ps
        return rv, hv, muv, gv, False, 0.0
    liquid = t < tc and xq < 0.95
    psat = 0.0
    if liquid:
        psat = saturation(f, t, model, par, sat, satg)[0]
    return W[i, W_RHO], W[i, W_H], W[i, W_MU], W[i, W_G], liquid, psat


@jit
def _opening(k, LI, LF, LV, pa, pb):
    c = LI[k, L_CTRL]
    if c == CTRL_VALVE:
        return LV[k, V_POS]
    if c == CTRL_CHECK or c == CTRL_RELIEF:
        o = (pa - pb - LF[k, LF_P1]) / LF[k, LF_P2]
        return min(max(o, 0.0), 1.0)
    if c == CTRL_REGULATOR:
        if pa <= pb:
            return 0.0
        o = (LF[k, LF_P1] - pb) / LF[k, LF_P2]
        return min(max(o, 0.0), 1.0)
    return 1.0


@jit
def rhs(x, xdot, NI, NF, LI, LF, LV, fl, W, nout, lout):
    """Time derivative of the state vector. Also fills node and link outputs."""
    model, par, tab, grid, ltab, lgrid, rhol, sat, satg = fl
    xdot[:] = 0.0
    _node_props(x, NI, NF, fl, W)
    nl = LI.shape[0]
    for k in range(nl):
        a = LI[k, L_A]
        b = LI[k, L_B]
        pa = W[a, W_P]
        pb = W[b, W_P]
        o = _opening(k, LI, LF, LV, pa, pb)
        if LI[k, L_KIND] == LINK_PIPE:
            sx = LI[k, L_SX]
            mdot = x[sx]
            fwd = mdot > 0.0 or (mdot == 0.0 and pa >= pb)
        else:
            fwd = pa >= pb
            mdot = 0.0
        up = a if fwd else b
        dn = b if fwd else a
        d_up = LI[k, L_DRAWA] if fwd else LI[k, L_DRAWB]
        rho, h, mu, gam, liquid, psat = _draw(up, d_up, NI, W, fl)
        s_up = W[up, W_S]
        pt = NI[up, N_PARTNER]
        # an emptying tank liquid region hands its outflow over to the ullage gas
        if NI[up, N_KIND] == NODE_TANK_LIQUID and s_up < 0.5:
            rho2, h2, mu2, gam2, liq2, ps2 = _draw(pt, DRAW_VAPOR, NI, W, fl)
            rho, mu, gam, liquid, psat = rho2, mu2, gam2, liq2, ps2
        p_up = W[up, W_P]
        p_dn = W[dn, W_P]
        if LI[k, L_KIND] == LINK_PIPE:
            area = LF[k, LF_AREA]
            r = _friction_r(mdot, rho, mu, area, LF[k, LF_DIAM], LF[k, LF_LEN], LF[k, LF_ROUGH], LF[k, LF_K])
            acc = area / LF[k, LF_LEN] * (pa - pb - r)
            if not liquid:
                m_ch = _choked_flux(p_up, rho, gam) * area
                if abs(mdot) > m_ch:
                    acc -= (abs(mdot) - m_ch) * (1.0 if mdot > 0 else -1.0) / CHOKE_TAU
            xdot[sx] = acc
            q = abs(mdot)
            if NI[up, N_KIND] == NODE_VOLUME:
                q *= s_up
                mdot = q if fwd else -q
        else:
            q = _orifice_flow(LF[k, LF_CDA] * o, p_up, p_dn, rho, gam, liquid, psat)
            if NI[up, N_KIND] == NODE_VOLUME:
                q *= s_up
            mdot = q if fwd else -q
        lout[k, LO_MDOT] = mdot
        lout[k, LO_OPEN] = o
        if q == 0.0:
            continue
        # energy carried: liquid share at the liquid enthalpy, the rest as ullage gas
        if NI[up, N_KIND] == NODE_TANK_LIQUID:
            h_g = _draw(pt, DRAW_VAPOR, NI, W, fl)[1]
            sx_up = NI[up, N_SX]
            sx_pt = NI[pt, N_SX]
            xdot[sx_up] -= s_up * q
            xdot[sx_up + 1] -= s_up * q * h
            xdot[sx_pt] -= (1.0 - s_up) * q
            xdot[sx_pt + 1] -= (1.0 - s_up) * q * h_g
            e = s_up * q * h + (1.0 - s_up) * q * h_g
        else:
            if NI[up, N_KIND] != NODE_BOUNDARY:
                sx_up = NI[up, N_SX]
                xdot[sx_up] -= q
                xdot[sx_up + 1] -= q * h
            e = q * h
        if NI[dn, N_KIND] != NODE_BOUNDARY:
            sx_dn = NI[dn, N_SX]
            xdot[sx_dn] += q
            xdot[sx_dn + 1] += e

    n = NI.shape[0]
    for i in range(n):
        kind = NI[i, N_KIND]
        if kind == NODE_BOUNDARY:
            continue
        sx = NI[i, N_SX]
        t = W[i, W_T]
        if NI[i, N_WALL]:
            tw = x[sx + 2]
            q_in = NF[i, NF_HA] * (tw - t)
            xdot[sx + 1] += q_in
            xdot[sx + 2] = (-q_in + NF[i, NF_HAAMB] * (NF[i, NF_TAMB] - tw)) / NF[i, NF_WALLMC]
        elif NF[i, NF_HA] > 0.0:
            xdot[sx + 1] += NF[i, NF_HA] * (NF[i, NF_TAMB] - t)
        if kind == NODE_TANK_LIQUID:
            # liquid volume change does pressure work on the ullage gas
            pt = NI[i, N_PARTNER]
            w = W[pt, W_P] * xdot[sx] / NF[i, NF_RHOREF]
            xdot[NI[pt, N_SX] + 1] += w
            xdot[sx + 1] -= w

    for i in range(n):
        nout[i, O_P] = W[i, W_P]
        nout[i, O_T] = W[i, W_T]
        nout[i, O_X] = W[i, W_X]
        nout[i, O_RHO] = W[i, W_RHO]
        nout[i, O_H] = W[i, W_H]
        nout[i, O_M] = x[NI[i, N_SX]] if NI[i, N_KIND] != NODE_BOUNDARY else 0.0
        nout[i, O_TW] = W[i, W_TW]


# ---------------------------------------------------------------------------
# Jacobian and banded solve


@jit
def _jacobian(x, f0, NI, NF, LI, LF, LV, fl, W, nscr, lscr,
              xscale, col_ptr, col_rows, color_ptr, color_cols, jvals, xp, fp):
    n = x.shape[0]
    ncol = color_ptr.shape[0] - 1
    for c in range(ncol):
        for j in range(n):
            xp[j] = x[j]
        for q in range(color_ptr[c], color_ptr[c + 1]):
            j = color_cols[q]
            xp[j] = x[j] + 1e-7 * max(abs(x[j]), xscale[j])
        rhs(xp, fp, NI, NF, LI, LF, LV, fl, W, nscr, lscr)
        for q in range(color_ptr[c], color_ptr[c + 1]):
            j = color_cols[q]
            h = xp[j] - x[j]
            for idx in range(col_ptr[j], col_ptr[j + 1]):
                i = col_rows[idx]
                jvals[idx] = (fp[i] - f0[i]) / h


@jit
def _band_factor(ab, piv, kl, ku, n):
    """LU-factor a banded matrix in place with partial pivoting (like LAPACK gbtrf).

    ``ab[i, j - i + kl]`` holds entry (i, j) for j - i in [-kl, kl + ku]; the extra
    kl upper diagonals take the fill-in from row swaps. Multipliers stay in the
    lower band and the pivot rows go to ``piv``. Returns False if singular.
    """
    w = kl + ku
    for k in range(n):
        p = k
        best = abs(ab[k, kl])
        last = min(n - 1, k + kl)
        for i in range(k + 1, last + 1):
            v = abs(ab[i, k - i + kl])
            if v > best:
                best = v
                p = i
        piv[k] = p
        if best == 0.0 or not math.isfinite(best):
            return False
        jmax = min(n - 1, k + w)
        if p != k:
            for j in range(k, jmax + 1):
                t = ab[k, j - k + kl]
                ab[k, j - k + kl] = ab[p, j - p + kl]
                ab[p, j - p + kl] = t
        dkk = ab[k, kl]
        for i in range(k + 1, last + 1):
            lik = ab[i, k - i + kl] / dkk
            ab[i, k - i + kl] = lik
            if lik == 0.0:
                continue
            for j in range(k + 1, jmax + 1):
                ab[i, j - i + kl] -= lik * ab[k, j - k + kl]
    return True


@jit
def _band_solve(ab, piv, b, kl, ku, n):
    """Solve with factors from ``_band_factor``; ``b`` is overwritten with x."""
    w = kl + ku
    for k in range(n):
        p = piv[k]
        if p != k:
            t = b[k]
            b[k] = b[p]
            b[p] = t
        bk = b[k]
        if bk != 0.0:
            for i in range(k + 1, min(n - 1, k + kl) + 1):
                b[i] -= ab[i, k - i + kl] * bk
    for i in range(n - 1, -1, -1):
        s = b[i]
        for j in range(i + 1, min(n - 1, i + w) + 1):
            s -= ab[i, j - i + kl] * b[j]
        b[i] = s / ab[i, kl]


@jit
def _factor(h, NI, jvals, col_ptr, col_rows, iperm, kl, ku, ab, piv):
    n = iperm.shape[0]
    ab[:, :] = 0.0
    for i in range(n):
        ab[iperm[i], kl] = 1.0
    for j in range(n):
        pj = iperm[j]
        for idx in range(col_ptr[j], col_ptr[j + 1]):
            pi = iperm[col_rows[idx]]
            ab[pi, pj - pi + kl] -= h * jvals[idx]
    return _band_factor(ab, piv, kl, ku, n)


# Newton settings for each backward-Euler step. Most steps converge in one or two
# iterations; filling or flashing in small volumes can take more. A step that has
# not converged after NEWTON_MAX iterations is taken anyway (and counted), so the
# real-time budget is never blown by one hard step.
NEWTON_MAX = 8
NEWTON_TOL = 1e-4  # scaled Newton update that counts as converged
NEWTON_REJAC = 1e-2  # update size above which the Jacobian is rebuilt at the iterate
CLAMP = False
ACCEPT_UNCONVERGED = False
JAC_MAX_AGE = 10  # steps a Jacobian may be reused while Newton converges quickly


@jit
def _newton(x, h, xk, NI, NF, LI, LF, LV, fl, W, nout, lout, nscr, lscr, xscale,
            col_ptr, col_rows, color_ptr, color_cols, iperm, kl, ku, jvals, xp, fp, fk, ab, piv, bvec, ctl):
    """One backward-Euler step of size h from x into xk by damped modified Newton.

    ``ctl``: [jac_fresh, lu_h, iterations of last solve, jac_age, need_jac, clamped mass]. Returns success.
    """
    n = x.shape[0]
    nn = NI.shape[0]
    if ctl[4] > 0.0:
        rhs(x, fk, NI, NF, LI, LF, LV, fl, W, nout, lout)
        _jacobian(x, fk, NI, NF, LI, LF, LV, fl, W, nscr, lscr,
                  xscale, col_ptr, col_rows, color_ptr, color_cols, jvals, xp, fp)
        ctl[0] = 1.0
        ctl[3] = 0.0
        ctl[4] = 0.0
        ctl[1] = -1.0
    if ctl[1] != h:
        if not _factor(h, NI, jvals, col_ptr, col_rows, iperm, kl, ku, ab, piv):
            return False
        ctl[1] = h
    for i in range(n):
        xk[i] = x[i]
    for it in range(NEWTON_MAX):
        rhs(xk, fk, NI, NF, LI, LF, LV, fl, W, nout, lout)
        for i in range(n):
            bvec[iperm[i]] = -(xk[i] - x[i] - h * fk[i])
        _band_solve(ab, piv, bvec, kl, ku, n)
        # a volume may lose at most 95% of its mass in one iteration; a larger
        # predicted loss is clamped (the clamped mass is counted in stats)
        for i in range(nn):
            if NI[i, N_KIND] != NODE_VOLUME:
                continue
            sx = NI[i, N_SX]
            m = xk[sx]
            dm = bvec[iperm[sx]]
            if m + dm < 0.05 * m:
                if CLAMP:
                    u_old = xk[sx + 1] / max(m, 1e-30)
                    bvec[iperm[sx]] = -0.95 * m
                    bvec[iperm[sx + 1]] = 0.05 * m * u_old - xk[sx + 1]
                    ctl[5] += 0.05 * m - (m + dm)
                else:
                    return False
        err = 0.0
        for i in range(n):
            d = bvec[iperm[i]]
            if not math.isfinite(d):
                return False
            xk[i] += d
            e = abs(d) / xscale[i]
            if e > err:
                err = e
        ctl[2] = it + 1
        if err < NEWTON_TOL:
            break
        if it + 1 < NEWTON_MAX and (err > NEWTON_REJAC or (it >= 1 and ctl[0] == 0.0)):
            # large or slow correction: re-linearize at the current iterate
            rhs(xk, fk, NI, NF, LI, LF, LV, fl, W, nout, lout)
            _jacobian(xk, fk, NI, NF, LI, LF, LV, fl, W, nscr, lscr,
                      xscale, col_ptr, col_rows, color_ptr, color_cols, jvals, xp, fp)
            ctl[0] = 1.0
            ctl[3] = 0.0
            if not _factor(h, NI, jvals, col_ptr, col_rows, iperm, kl, ku, ab, piv):
                return False
    else:
        if not ACCEPT_UNCONVERGED:
            return False
        if NEWTON_MAX > 1:
            ctl[6] += 1.0
    # tank liquid regions may run dry; clamp tiny negatives from round-off
    for i in range(nn):
        if NI[i, N_KIND] == NODE_TANK_LIQUID:
            sx = NI[i, N_SX]
            if xk[sx] < 1e-12:
                xk[sx + 1] = xk[sx + 1] / max(abs(xk[sx]), 1e-30) * 1e-12 if xk[sx] != 0.0 else 0.0
                xk[sx] = 1e-12
        elif NI[i, N_KIND] != NODE_BOUNDARY and NI[i, N_WALL] and xk[NI[i, N_SX] + 2] <= 1.0:
            return False
    return True


@jit
def advance(x, n_steps, dt, NI, NF, LI, LF, LV, fl, W, nout, lout, nscr, lscr,
            xscale, col_ptr, col_rows, color_ptr, color_cols, iperm, kl, ku,
            jvals, xp, fp, fk, ab, piv, bvec, xnew, ctl, stats, hist_p, hist_t, hist_m, hist_every):
    """Advance ``n_steps`` fixed steps of ``dt``.

    A step whose Newton iteration does not converge is retried as smaller
    sub-steps (down to dt/256) with a fresh Jacobian; the outer step never
    changes, so simulated time stays on the fixed grid. ``stats`` accumulates
    [sub-steps, Newton iterations, Jacobians, failed sub-steps, clamped mass,
    steps taken without converging].
    History rows are written every ``hist_every`` steps (none if 0).
    """
    nl = LI.shape[0]
    hrow = 0
    for s in range(n_steps):
        moving = False
        for k in range(nl):
            if LI[k, L_CTRL] == CTRL_VALVE:
                d = LV[k, V_CMD] - LV[k, V_POS]
                if d != 0.0:
                    moving = True
                if d > 0:
                    d = min(d, LV[k, V_RATE_OPEN] * dt)
                else:
                    d = max(d, -LV[k, V_RATE_CLOSE] * dt)
                LV[k, V_POS] += d
        ctl[3] += 1.0
        if moving or ctl[3] >= JAC_MAX_AGE or ctl[2] > 3:
            ctl[4] = 1.0
        ctl[0] = 0.0
        remaining = dt
        h = dt
        while remaining > 1e-15:
            h = min(h, remaining)
            ctl[5] = 0.0
            ok = _newton(x, h, xnew, NI, NF, LI, LF, LV, fl, W, nout, lout, nscr, lscr, xscale,
                         col_ptr, col_rows, color_ptr, color_cols, iperm, kl, ku, jvals, xp, fp, fk, ab, piv,
                         bvec, ctl)
            stats[1] += ctl[2]
            if ctl[0] > 0.0:
                stats[2] += 1.0
            if not ok:
                stats[3] += 1.0
                if h > dt / 64.0:
                    h *= 0.5
                    ctl[4] = 1.0
                    ctl[0] = 0.0
                    continue
                # even the smallest sub-step did not converge: keep the damped
                # iterate (masses stay positive) so simulated time does not stall
                finite = True
                for i in range(x.shape[0]):
                    if not math.isfinite(xnew[i]):
                        finite = False
                if not finite:
                    for i in range(x.shape[0]):
                        xnew[i] = x[i]
            for i in range(x.shape[0]):
                x[i] = xnew[i]
            stats[4] += ctl[5]
            stats[5] += ctl[6]
            ctl[6] = 0.0
            remaining -= h
            stats[0] += 1.0
            if h < dt:
                h = min(2.0 * h, dt)
        if hist_every > 0 and (s + 1) % hist_every == 0 and hrow < hist_p.shape[0]:
            rhs(x, fk, NI, NF, LI, LF, LV, fl, W, nout, lout)
            for i in range(NI.shape[0]):
                hist_p[hrow, i] = nout[i, O_P]
                hist_t[hrow, i] = nout[i, O_T]
            for k in range(nl):
                hist_m[hrow, k] = lout[k, LO_MDOT]
            hrow += 1
    rhs(x, fk, NI, NF, LI, LF, LV, fl, W, nout, lout)
