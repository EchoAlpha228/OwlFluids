"""The compiled time-stepping kernel.

State vector ``x``: for every non-boundary node its mass ``m`` and total internal
energy ``U`` (plus wall temperature when the node has a wall), and for every
pipe link its mass flow. Orifices, valves and fittings are quasi-steady: their
flow is an algebraic function of the node states on either side.

Each step is backward Euler, solved by chord Newton iterations

    (I - dt J) dx = -(x_k - x_n - dt f(x_k))

on a Jacobian ``J`` that is reused while iterations converge quickly. ``J`` is
built by finite differences, perturbing groups of structurally independent
states together (graph colouring) and re-evaluating only the nodes and links
they touch. The system is solved with a banded LU after the unknowns were
reordered with reverse Cuthill-McKee. Every Newton iterate conserves mass and
energy exactly, because each link moves what it takes from one node into another.
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

from ..fluids.props import psat as _psat
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

# Compiled without Numba's reference counting: nothing here allocates, and the
# counting otherwise costs more than the physics in the small per-node and
# per-link helpers.
jit = nb.njit(cache=True, nogil=True, _nrt=False)
inl = jit

G0 = 9.80665
DP_LIN = 2000.0  # Pa; below this pressure difference flow is linear in dP (keeps the Jacobian finite)
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


@inl
def _node_one(i, x, NI, NF, fl, W):
    """Fill row i of the scratch array W with the state of node i."""
    model, par, tab, grid, ltab, lgrid, rhol, sat, satg = fl
    kind = NI[i, N_KIND]
    f = NI[i, N_FLUID]
    if kind == NODE_BOUNDARY:
        rho = NF[i, NF_BRHO]
        u = NF[i, NF_BU]
        if W[i, W_RHO] == rho and W[i, W_U] == u and W[i, W_P] == NF[i, NF_BP]:
            return  # unchanged since the last call
        p, t, xq, g, mu = state(f, rho, u, model, par, tab, grid, ltab, lgrid, rhol)
        W[i, W_P] = NF[i, NF_BP]
        W[i, W_T] = NF[i, NF_BT]
        W[i, W_V] = 0.0
        W[i, W_TW] = NF[i, NF_BT]
        W[i, W_S] = 1.0
        W[i, W_X] = xq
        W[i, W_RHO] = rho
        W[i, W_U] = u
        W[i, W_G] = g
        W[i, W_MU] = mu
        W[i, W_H] = u + NF[i, NF_BP] / rho
        return
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
    W[i, W_H] = u + p / rho


@inl
def _tank_liquid_p(i, NI, NF, W):
    """Tank liquid takes its pressure from the ullage above it, plus liquid head."""
    pt = NI[i, N_PARTNER]
    head = 0.0
    if NF[i, NF_AREA] > 0.0:
        head = W[i, W_RHO] * G0 * W[i, W_V] / NF[i, NF_AREA]
    W[i, W_P] = W[pt, W_P] + head
    W[i, W_H] = W[i, W_U] + W[i, W_P] / W[i, W_RHO]


@jit
def _node_props(x, NI, NF, fl, W):
    n = NI.shape[0]
    for i in range(n):
        _node_one(i, x, NI, NF, fl, W)
    for i in range(n):
        if NI[i, N_KIND] == NODE_TANK_LIQUID:
            _tank_liquid_p(i, NI, NF, W)


@inl
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
            psat = _psat(f, t, model, par, sat, satg)
        return W[i, W_RHO], W[i, W_H], W[i, W_MU], W[i, W_G], True, psat
    if 0.0 < xq < 1.0 and d != DRAW_MIX:
        ps, rl, rv, hl, hv, mul, muv, gv, sub = saturation(f, t, model, par, sat, satg)
        if d == DRAW_LIQUID:
            return rl, hl, mul, gv, True, ps
        return rv, hv, muv, gv, False, 0.0
    liquid = t < tc and xq < 0.95
    psat = 0.0
    if liquid:
        psat = _psat(f, t, model, par, sat, satg)
    return W[i, W_RHO], W[i, W_H], W[i, W_MU], W[i, W_G], liquid, psat


@inl
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


@inl
def _link_flux(k, x, NI, NF, LI, LF, LV, fl, W, fn, fv):
    """Flows carried by link k.

    ``fn[0..3]`` are the nodes it changes (upstream, downstream, and the ullage
    partners of either when it is tank liquid; -1 for none) and ``fv[2s]``,
    ``fv[2s + 1]`` their mass and energy rates; ``fv[8]`` is d(mdot)/dt of a pipe.
    Returns (signed mass flow, opening).
    """
    for s in range(4):
        fn[s] = -1
    for s in range(9):
        fv[s] = 0.0
    a = LI[k, L_A]
    b = LI[k, L_B]
    pa = W[a, W_P]
    pb = W[b, W_P]
    o = _opening(k, LI, LF, LV, pa, pb)
    pipe = LI[k, L_KIND] == LINK_PIPE
    if pipe:
        mdot = x[LI[k, L_SX]]
        fwd = mdot > 0.0 or (mdot == 0.0 and pa >= pb)
    else:
        fwd = pa >= pb
        mdot = 0.0
    up = a if fwd else b
    dn = b if fwd else a
    d_up = LI[k, L_DRAWA] if fwd else LI[k, L_DRAWB]
    rho, h, mu, gam, liquid, psat = _draw(up, d_up, NI, W, fl)
    s_up = W[up, W_S]
    up_kind = NI[up, N_KIND]
    pt = NI[up, N_PARTNER]
    # an emptying tank liquid region hands its outflow over to the ullage gas
    if up_kind == NODE_TANK_LIQUID and s_up < 0.5:
        rho2, h2, mu2, gam2, liq2, ps2 = _draw(pt, DRAW_VAPOR, NI, W, fl)
        rho, mu, gam, liquid, psat = rho2, mu2, gam2, liq2, ps2
    p_up = W[up, W_P]
    p_dn = W[dn, W_P]
    if pipe:
        area = LF[k, LF_AREA]
        r = _friction_r(mdot, rho, mu, area, LF[k, LF_DIAM], LF[k, LF_LEN], LF[k, LF_ROUGH], LF[k, LF_K])
        acc = area / LF[k, LF_LEN] * (pa - pb - r)
        if not liquid:
            m_ch = _choked_flux(p_up, rho, gam) * area
            if abs(mdot) > m_ch:
                acc -= (abs(mdot) - m_ch) * (1.0 if mdot > 0 else -1.0) / CHOKE_TAU
        fv[8] = acc
        q = abs(mdot)
        if up_kind == NODE_VOLUME:
            q *= s_up
            mdot = q if fwd else -q
    else:
        q = _orifice_flow(LF[k, LF_CDA] * o, p_up, p_dn, rho, gam, liquid, psat)
        if up_kind == NODE_VOLUME:
            q *= s_up
        mdot = q if fwd else -q
    if q == 0.0:
        return mdot, o
    if up_kind == NODE_TANK_LIQUID:
        # liquid share at the liquid enthalpy, the rest as ullage gas
        h_g = _draw(pt, DRAW_VAPOR, NI, W, fl)[1]
        fn[0] = up
        fv[0] = -s_up * q
        fv[1] = -s_up * q * h
        fn[2] = pt
        fv[4] = -(1.0 - s_up) * q
        fv[5] = -(1.0 - s_up) * q * h_g
        e = s_up * q * h + (1.0 - s_up) * q * h_g
        # the ullage does the work of pushing the liquid out
        w = W[pt, W_P] * fv[0] / NF[up, NF_RHOREF]
        fv[5] += w
        fv[1] -= w
    else:
        if up_kind != NODE_BOUNDARY:
            fn[0] = up
            fv[0] = -q
            fv[1] = -q * h
        e = q * h
    if NI[dn, N_KIND] != NODE_BOUNDARY:
        fn[1] = dn
        fv[2] = q
        fv[3] = e
        if NI[dn, N_KIND] == NODE_TANK_LIQUID:
            pd = NI[dn, N_PARTNER]
            fn[3] = pd
            w = W[pd, W_P] * q / NF[dn, NF_RHOREF]
            fv[7] += w
            fv[3] -= w
    return mdot, o


@inl
def _node_source(i, x, NI, NF, W, ns):
    """Heat flows of node i: ns[0] into the fluid, ns[1] d(Tw)/dt of its wall."""
    ns[0] = 0.0
    ns[1] = 0.0
    if NI[i, N_KIND] == NODE_BOUNDARY:
        return
    t = W[i, W_T]
    if NI[i, N_WALL]:
        tw = x[NI[i, N_SX] + 2]
        q_in = NF[i, NF_HA] * (tw - t)
        ns[0] = q_in
        ns[1] = (-q_in + NF[i, NF_HAAMB] * (NF[i, NF_TAMB] - tw)) / NF[i, NF_WALLMC]
    elif NF[i, NF_HA] > 0.0:
        ns[0] = NF[i, NF_HA] * (NF[i, NF_TAMB] - t)


@inl
def _scatter_link(k, NI, LI, fn, fv, out, sign):
    for s in range(4):
        i = fn[s]
        if i >= 0:
            sx = NI[i, N_SX]
            out[sx] += sign * fv[2 * s]
            out[sx + 1] += sign * fv[2 * s + 1]
    if LI[k, L_KIND] == LINK_PIPE:
        out[LI[k, L_SX]] += sign * fv[8]


@inl
def _scatter_node(i, NI, ns, out, sign):
    if NI[i, N_KIND] == NODE_BOUNDARY:
        return
    sx = NI[i, N_SX]
    out[sx + 1] += sign * ns[0]
    if NI[i, N_WALL]:
        out[sx + 2] += sign * ns[1]


@jit
def rhs(x, xdot, NI, NF, LI, LF, LV, fl, W, nout, lout, FN, FV, NS):
    """Time derivative of the state vector. Also fills node and link outputs.

    The per-link and per-node contributions are kept in FN, FV and NS so the
    Jacobian can update them incrementally.
    """
    xdot[:] = 0.0
    _node_props(x, NI, NF, fl, W)
    for k in range(LI.shape[0]):
        mdot, o = _link_flux(k, x, NI, NF, LI, LF, LV, fl, W, FN[k], FV[k])
        lout[k, LO_MDOT] = mdot
        lout[k, LO_OPEN] = o
        _scatter_link(k, NI, LI, FN[k], FV[k], xdot, 1.0)
    n = NI.shape[0]
    for i in range(n):
        _node_source(i, x, NI, NF, W, NS[i])
        _scatter_node(i, NI, NS[i], xdot, 1.0)
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
def _jacobian(x, NI, NF, LI, LF, LV, fl, W, FN, FV, NS, xscale, st, sc, jvals):
    """Finite-difference Jacobian at x, right after ``rhs(x)`` filled W, FN, FV, NS.

    Columns of one colour are perturbed together. Only the nodes they own (and
    their tank partners) are re-evaluated, and only the links touching those
    nodes; the change of each contribution gives the Jacobian entries, so a
    colour costs a small fraction of a full right-hand side.
    """
    col_ptr, col_rows, color_ptr, color_cols, sowner, nl_ptr, nl_idx = st
    xp, dfp, stn, stl, dl, ll, wsave, fnt, fvt, nst, gen = sc
    n = x.shape[0]
    nw = W.shape[1]
    for j in range(n):
        xp[j] = x[j]
        dfp[j] = 0.0
    ncol = color_ptr.shape[0] - 1
    for c in range(ncol):
        gen[0] += 1
        g = gen[0]
        nd = 0
        nk = 0
        for q in range(color_ptr[c], color_ptr[c + 1]):
            j = color_cols[q]
            xp[j] = x[j] + 1e-7 * max(abs(x[j]), xscale[j])
            o = sowner[j]
            if o >= 0:
                for i in (o, NI[o, N_PARTNER]):
                    if i >= 0 and stn[i] != g:
                        stn[i] = g
                        dl[nd] = i
                        nd += 1
            else:
                k = -o - 1
                if stl[k] != g:
                    stl[k] = g
                    ll[nk] = k
                    nk += 1
        for d in range(nd):
            i = dl[d]
            for col in range(nw):
                wsave[d, col] = W[i, col]
            _node_one(i, xp, NI, NF, fl, W)
            for idx in range(nl_ptr[i], nl_ptr[i + 1]):
                k = nl_idx[idx]
                if stl[k] != g:
                    stl[k] = g
                    ll[nk] = k
                    nk += 1
        for d in range(nd):
            i = dl[d]
            if NI[i, N_KIND] == NODE_TANK_LIQUID:
                _tank_liquid_p(i, NI, NF, W)
        for d in range(nk):
            k = ll[d]
            _scatter_link(k, NI, LI, FN[k], FV[k], dfp, -1.0)
            _link_flux(k, xp, NI, NF, LI, LF, LV, fl, W, fnt, fvt)
            _scatter_link(k, NI, LI, fnt, fvt, dfp, 1.0)
        for d in range(nd):
            i = dl[d]
            _scatter_node(i, NI, NS[i], dfp, -1.0)
            _node_source(i, xp, NI, NF, W, nst)
            _scatter_node(i, NI, nst, dfp, 1.0)
        for q in range(color_ptr[c], color_ptr[c + 1]):
            j = color_cols[q]
            h = xp[j] - x[j]
            for idx in range(col_ptr[j], col_ptr[j + 1]):
                i = col_rows[idx]
                jvals[idx] = dfp[i] / h
            xp[j] = x[j]
        # undo: scratch rows and the derivative entries (all inside these columns' rows)
        for d in range(nd):
            i = dl[d]
            for col in range(nw):
                W[i, col] = wsave[d, col]
        for q in range(color_ptr[c], color_ptr[c + 1]):
            j = color_cols[q]
            for idx in range(col_ptr[j], col_ptr[j + 1]):
                dfp[col_rows[idx]] = 0.0


@jit
def _liquid_margin(f, rho, u, model, grid, lgrid, rhol):
    """Relative distance of (rho, u) past the saturated-liquid boundary (> 0: compressed liquid)."""
    if model[f] == MODEL_LIQUID or model[f] == MODEL_IDEAL:
        return -1.0
    u = min(max(u, grid[f, 3]), grid[f, 4])
    u_lo = lgrid[f, 0]
    u_hi = lgrid[f, 1]
    if u >= u_hi:
        return -1.0
    su = (u - u_lo) / (u_hi - u_lo) * (rhol.shape[1] - 1)
    k0 = min(max(int(math.floor(su)), 0), rhol.shape[1] - 2)
    c = su - k0
    r_l = (1 - c) * rhol[f, k0] + c * rhol[f, k0 + 1]
    return rho / r_l - 1.0


@jit
def _chop(xk, bvec, iperm, NI, NF, fl):
    """Limit each volume node's Newton update (Appleyard-style chop).

    A node may not lose more than 80% of its mass in one iteration, and an update
    that crosses the saturated-liquid boundary stops a little past it, so the next
    iteration is linearized on the right side of the kink. Scaling m and U together
    keeps the update's direction. Returns True if anything was chopped.
    """
    model, par, tab, grid, ltab, lgrid, rhol, sat, satg = fl
    chopped = False
    for i in range(NI.shape[0]):
        if NI[i, N_KIND] != NODE_VOLUME or NI[i, N_PARTNER] >= 0:
            continue
        sx = NI[i, N_SX]
        m0 = xk[sx]
        dm = bvec[iperm[sx]]
        du = bvec[iperm[sx + 1]]
        a = 1.0
        if m0 + dm < 0.2 * m0:
            a = 0.8 * m0 / -dm
        f = NI[i, N_FLUID]
        vol = NF[i, NF_VOL]
        u0 = xk[sx + 1] / m0
        m1 = m0 + a * dm
        u1 = (xk[sx + 1] + a * du) / m1
        l0 = _liquid_margin(f, m0 / vol, u0, model, grid, lgrid, rhol)
        l1 = _liquid_margin(f, m1 / vol, u1, model, grid, lgrid, rhol)
        if (l0 <= 0.0) != (l1 <= 0.0) and abs(l1) > 1e-3:
            c = l0 / (l0 - l1)
            c = c + 0.1 * (1.0 - c)
            a *= c
        if a < 1.0:
            bvec[iperm[sx]] = a * dm
            bvec[iperm[sx + 1]] = a * du
            chopped = True
    return chopped


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
# chord iterations on a reused Jacobian; filling or flashing in small volumes can
# take more, and a step that cannot converge is retried as sub-steps within a
# fixed work budget (see ``advance``).
NEWTON_MAX = 8
NEWTON_TOL = 1e-4  # scaled Newton update that counts as converged
NEWTON_REJAC = 1e-2  # update size above which the Jacobian is rebuilt at the iterate
ACCEPT_TOL = 1e-2  # an unconverged step is still taken if its last update was this small
JAC_MAX_AGE = 10  # steps a Jacobian may be reused while Newton converges quickly
STEP_BUDGET = 40.0  # work one step may spend (in right-hand-side evaluations) before it is forced
JAC_COST = 5.0  # a Jacobian plus factorization, in right-hand-side evaluations
MIN_SUBSTEP = 1.0 / 16.0  # smallest retry, as a fraction of dt


@jit
def _newton(x, h, xk, NI, NF, LI, LF, LV, fl, W, nout, lout, FN, FV, NS, xscale,
            st, sc, iperm, kl, ku, jvals, fk, ab, piv, bvec, ctl):
    """One backward-Euler step of size h from x into xk by damped modified Newton.

    ``ctl``: [jac_fresh, lu_h, iterations of last solve, jac_age, need_jac, clamped mass]. Returns success.
    """
    n = x.shape[0]
    nn = NI.shape[0]
    if ctl[4] > 0.0:
        ctl[8] += JAC_COST + 1.0
        rhs(x, fk, NI, NF, LI, LF, LV, fl, W, nout, lout, FN, FV, NS)
        _jacobian(x, NI, NF, LI, LF, LV, fl, W, FN, FV, NS, xscale, st, sc, jvals)
        ctl[0] = 1.0
        ctl[3] = 0.0
        ctl[4] = 0.0
        ctl[1] = -1.0
    if ctl[1] != h:
        if not _factor(h, NI, jvals, st[0], st[1], iperm, kl, ku, ab, piv):
            return False
        ctl[1] = h
    for i in range(n):
        xk[i] = x[i]
    err_prev = 1e30
    for it in range(NEWTON_MAX):
        ctl[8] += 1.0
        if ctl[8] > STEP_BUDGET:
            return False
        rhs(xk, fk, NI, NF, LI, LF, LV, fl, W, nout, lout, FN, FV, NS)
        for i in range(n):
            bvec[iperm[i]] = -(xk[i] - x[i] - h * fk[i])
        _band_solve(ab, piv, bvec, kl, ku, n)
        chopped = _chop(xk, bvec, iperm, NI, NF, fl)
        err = 0.0
        for i in range(n):
            d = bvec[iperm[i]]
            if not math.isfinite(d):
                return False
            xk[i] += d
            e = abs(d) / xscale[i]
            if e > err:
                err = e
                ctl[7] = i
        ctl[2] = it + 1
        if err < NEWTON_TOL and not chopped:
            break
        if it >= 1 and not chopped and err < ACCEPT_TOL and err > 0.5 * err_prev:
            # stalled on a kink of the interpolated tables, already close: take it
            ctl[6] += 1.0
            break
        if it + 1 < NEWTON_MAX and (err > NEWTON_REJAC or (it >= 1 and ctl[0] == 0.0 and err > 0.1 * err_prev)):
            # large or slow correction: re-linearize at the current iterate
            ctl[8] += JAC_COST + 1.0
            rhs(xk, fk, NI, NF, LI, LF, LV, fl, W, nout, lout, FN, FV, NS)
            _jacobian(xk, NI, NF, LI, LF, LV, fl, W, FN, FV, NS, xscale, st, sc, jvals)
            ctl[0] = 1.0
            ctl[3] = 0.0
            if not _factor(h, NI, jvals, st[0], st[1], iperm, kl, ku, ab, piv):
                return False
        err_prev = err
    else:
        if chopped or err > ACCEPT_TOL:
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
def _physical(x, NI):
    for i in range(x.shape[0]):
        if not math.isfinite(x[i]):
            return False
    for i in range(NI.shape[0]):
        if NI[i, N_KIND] == NODE_VOLUME and x[NI[i, N_SX]] <= 0.0:
            return False
    return True


@jit
def advance(x, n_steps, dt, NI, NF, LI, LF, LV, fl, W, nout, lout, FN, FV, NS,
            xscale, st, sc, iperm, kl, ku,
            jvals, fk, ab, piv, bvec, xnew, ctl, stats, hist_p, hist_t, hist_m, hist_every):
    """Advance ``n_steps`` fixed steps of ``dt``.

    A step whose Newton iteration does not converge is retried as smaller
    sub-steps with a fresh Jacobian; simulated time stays on the fixed grid.
    Each step has a work budget (STEP_BUDGET Newton iterations) so one hard step
    cannot stall real time: when it runs out, the last Newton iterate is taken if
    it is physical (it conserves mass and energy like a converged one), otherwise
    the state is held for the rest of the step. ``stats`` accumulates [sub-steps,
    Newton iterations, Jacobians, failed sub-steps, forced steps, steps taken
    without converging]. History rows are written every ``hist_every`` steps.
    """
    nl = LI.shape[0]
    hrow = 0
    for s in range(n_steps):
        for k in range(nl):
            if LI[k, L_CTRL] == CTRL_VALVE:
                d = LV[k, V_CMD] - LV[k, V_POS]
                if d > 0:
                    d = min(d, LV[k, V_RATE_OPEN] * dt)
                else:
                    d = max(d, -LV[k, V_RATE_CLOSE] * dt)
                LV[k, V_POS] += d
        ctl[3] += 1.0
        if ctl[3] >= JAC_MAX_AGE or ctl[2] > 3:
            ctl[4] = 1.0
        ctl[0] = 0.0
        remaining = dt
        h = dt
        ctl[8] = 0.0
        while remaining > 1e-15:
            h = min(h, remaining)
            ok = _newton(x, h, xnew, NI, NF, LI, LF, LV, fl, W, nout, lout, FN, FV, NS, xscale,
                         st, sc, iperm, kl, ku, jvals, fk, ab, piv, bvec, ctl)
            stats[1] += ctl[2]
            if ctl[0] > 0.0:
                stats[2] += 1.0
            if not ok:
                stats[3] += 1.0
                if h > dt * MIN_SUBSTEP and ctl[8] < STEP_BUDGET:
                    h *= 0.5
                    ctl[4] = 1.0
                    ctl[0] = 0.0
                    continue
                # out of budget: take the iterate if it is physical, else hold
                stats[4] += 1.0
                if _physical(xnew, NI):
                    for i in range(x.shape[0]):
                        x[i] = xnew[i]
                ctl[4] = 1.0
                break
            for i in range(x.shape[0]):
                x[i] = xnew[i]
            stats[5] += ctl[6]
            ctl[6] = 0.0
            remaining -= h
            stats[0] += 1.0
            if h < dt:
                h = min(2.0 * h, dt)
        if hist_every > 0 and (s + 1) % hist_every == 0 and hrow < hist_p.shape[0]:
            rhs(x, fk, NI, NF, LI, LF, LV, fl, W, nout, lout, FN, FV, NS)
            for i in range(NI.shape[0]):
                hist_p[hrow, i] = nout[i, O_P]
                hist_t[hrow, i] = nout[i, O_T]
            for k in range(nl):
                hist_m[hrow, k] = lout[k, LO_MDOT]
            hrow += 1
    rhs(x, fk, NI, NF, LI, LF, LV, fl, W, nout, lout, FN, FV, NS)
