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

Purges: a node that a purge gas can reach as well as its own fluid also carries the
gas's mass (the state after its others); its energy state is that of both. The two
share the volume at one pressure and temperature, each in its own part of it.
Outflow is homogeneous: each species leaves in proportion to its mass, carrying its
own enthalpy. The purge gas is an ideal gas on its fluid's energy reference (see
``registry.P_OFF``).
"""

from __future__ import annotations

import math

import numba as nb

from ..fluids.props import psat as _psat
from ..fluids.props import saturation, state
from ..fluids.registry import (
    I_CV,
    I_GAMMA,
    I_MU,
    I_R,
    L_CV,
    L_MU,
    L_PREF,
    L_PSAT,
    L_RHO0,
    L_T0,
    L_U0,
    MODEL_IDEAL,
    MODEL_LIQUID,
    P_CPA,
    P_KL,
    P_OFF,
)
from .layout import (
    CTRL_CHECK,
    CTRL_REGULATOR,
    CTRL_RELIEF,
    CTRL_VALVE,
    DRAW_LIQUID,
    DRAW_MIX,
    DRAW_VAPOR,
    L_A,
    L_B,
    L_CTRL,
    L_DRAWA,
    L_DRAWB,
    L_KIND,
    L_SX,
    LF_AREA,
    LF_CDA,
    LF_DIAM,
    LF_K,
    LF_LEN,
    LF_P1,
    LF_P2,
    LF_ROUGH,
    LINK_PIPE,
    LO_MDOT,
    LO_OPEN,
    N_FLUID,
    N_GAS,
    N_GX,
    N_KIND,
    N_PARTNER,
    N_SX,
    N_WALL,
    NF_AREA,
    NF_BP,
    NF_BRHO,
    NF_BT,
    NF_BU,
    NF_HA,
    NF_HAAMB,
    NF_MREF,
    NF_RHOREF,
    NF_TAMB,
    NF_VOL,
    NF_WALLMC,
    NODE_BOUNDARY,
    NODE_TANK_LIQUID,
    NODE_VOLUME,
    O_GAS,
    O_H,
    O_M,
    O_P,
    O_RHO,
    O_T,
    O_TW,
    O_X,
    V_CMD,
    V_POS,
    V_RATE_CLOSE,
    V_RATE_OPEN,
)

# Compiled without Numba's reference counting: nothing here allocates, and the
# counting otherwise costs more than the physics in the small per-node and
# per-link helpers.
# error_model="numpy": a division by zero in a wild Newton iterate gives inf/nan,
# which the finiteness checks reject, instead of raising out of the time loop
jit = nb.njit(cache=True, nogil=True, _nrt=False, error_model="numpy")
inl = jit

G0 = 9.80665
EXCH = 0.1  # exchange stream at zero flow, as a share of the flow at the linear band's edge
DP_LIN = 2000.0  # Pa; below this pressure difference flow is linear in dP (keeps the Jacobian finite)
R_LIN = 1e-3  # gas pressure ratio band near 1 treated the same way
CHOKE_TAU = 1e-3  # s; relaxation time that holds pipe gas flow at the choked limit

MIX_ITER = 40  # most iterations of each solve in a mixed node's state

# scratch columns. For a mixed node the first ones describe its own fluid (W_H: the
# enthalpy it leaves with), then: purge gas enthalpy, the share of the volume not taken
# by liquid, the own fluid's mass fraction, the mixture density, the log partial
# pressure of the gas (liquid present) or its fluid's energy (no liquid), whether liquid
# is present (0) or not (1), its vapor pressure, the purge gas share of the volume
# (shown), and the last three at the last accepted step, where each solve starts
(W_P, W_T, W_X, W_RHO, W_U, W_H, W_G, W_MU, W_V, W_TW, W_S, W_HG, W_SG, W_Y, W_RHOM, W_LT,
 W_REG, W_PS, W_GF, W_U0, W_LT0, W_T0, W_REG0) = range(23)
N_W = 23

# per-link flows: for each of 4 nodes, own-fluid mass, energy and purge-gas mass;
# then d(mdot)/dt of a pipe
N_FV = 13
FV_ACC = 12


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
def _sat_ref(f, t, fl):
    """Liquid of fluid f at t, for a node it shares with a purge gas.

    Returns (vapor pressure, reference density, its pressure, saturated vapor density,
    liquid and vapor internal energy, liquid viscosity, vapor gamma); the liquid
    density at pressure p is ref * (1 + (p - p_ref) / P_KL).
    """
    model, par, tab, grid, ltab, lgrid, rhol, sat, satg = fl
    if model[f] == MODEL_LIQUID:
        u_l = par[f, L_U0] + par[f, L_CV] * (t - par[f, L_T0])
        return par[f, L_PSAT], par[f, L_RHO0], par[f, L_PREF], 0.0, u_l, 0.0, par[f, L_MU], 1.3
    ps, rl, rv, hl, hv, mul, muv, gv, sub = saturation(f, t, model, par, sat, satg)
    return ps, rl, ps, rv, hl - ps / rl, hv - ps / rv, mul, gv


@inl
def _gas_space(vol, m_f, m_g, rgt, ps, rho_ref, p_ref, k_l, rho_v, w0):
    """Liquid beside its vapor and a purge gas, at one temperature (Dalton).

    The liquid, compressed by the total pressure ps + p_g, fills what the gas space
    leaves; vapor at its vapor pressure and the purge gas (``rgt``: R T) share the gas
    space. Solved for w = ln(p_g) by safeguarded Newton (the volume residual falls
    monotonically with it), from ``w0``. Returns (w, gas space, pressure, liquid mass,
    liquid density); the liquid mass is negative when the vapor alone outweighs m_f.
    """
    if m_f - rho_v * vol <= 0.0:
        return w0, vol, ps + m_g * rgt / vol, m_f - rho_v * vol, rho_ref
    if m_g <= 0.0:
        # no gas: the liquid fills the node above its vapor pressure, or boils
        p = p_ref + k_l * (m_f / (vol * rho_ref) - 1.0)
        if p >= ps:
            return w0, 0.0, p, m_f, m_f / vol
        rho_l = rho_ref * (1.0 + (ps - p_ref) / k_l)
        y = (vol * rho_l - m_f) / (rho_l - rho_v)
        return w0, y, ps, m_f - rho_v * y, rho_l
    lo = -10.0
    hi = 30.0
    w = min(max(w0, lo + 1e-3), hi - 1e-3)
    for _ in range(MIX_ITER):
        p_g = math.exp(w)
        y = m_g * rgt / p_g
        rho_liq = rho_ref * (1.0 + (ps + p_g - p_ref) / k_l)
        m_liq = m_f - rho_v * y
        r = y + m_liq / rho_liq - vol
        if abs(r) <= 1e-15 * vol:
            break
        if r > 0.0:
            lo = w
        else:
            hi = w
        d = -y * (1.0 - rho_v / rho_liq) - m_liq * rho_ref * p_g / (rho_liq * rho_liq * k_l)
        nw = w - r / d if d < 0.0 else 0.5 * (lo + hi)
        if not (lo <= nw <= hi):
            nw = 0.5 * (lo + hi)
        if abs(nw - w) < 1e-13:
            w = nw
            break
        w = nw
    p_g = math.exp(w)
    y = m_g * rgt / p_g
    rho_liq = rho_ref * (1.0 + (ps + p_g - p_ref) / k_l)
    return w, y, ps + p_g, m_f - rho_v * y, rho_liq


@inl
def _mixed_wet(f, t, vol, m_f, m_g, e, w0, fl, gpar):
    """Energy residual of a node holding liquid at temperature t, with its state."""
    cv, rg, off, k_l = gpar
    ps, rho_ref, p_ref, rho_v, u_l, u_v, mu_l, gam_v = _sat_ref(f, t, fl)
    w, y, p, m_liq, rho_liq = _gas_space(vol, m_f, m_g, rg * t, ps, rho_ref, p_ref, k_l, rho_v, w0)
    m_vap = m_f - m_liq
    res = m_liq * u_l + m_vap * u_v + m_g * (cv * t + off) - e
    return res, w, y, p, m_liq, rho_liq, ps, mu_l, gam_v, u_l, u_v


@inl
def _mixed_one(i, x, NI, NF, fl, W):
    """Node i holding its own fluid and a purge gas, at one temperature.

    With liquid present, the liquid fills its part of the node and its vapor and the
    purge gas share the rest (pressure: vapor pressure plus the gas's partial
    pressure), so boiling under a purge is smooth. Without liquid both gases fill the
    node and their partial pressures add. Each solve starts from the node's state at
    the last accepted step, so the result depends on x alone.
    """
    model, par, tab, grid, ltab, lgrid, rhol, sat, satg = fl
    f = NI[i, N_FLUID]
    g = NI[i, N_GAS]
    sx = NI[i, N_SX]
    vol = NF[i, NF_VOL]
    # a trace driven slightly negative by Newton counts as none
    m_f = max(x[sx], 0.0)
    m_g = max(x[NI[i, N_GX]], 0.0)
    e = x[sx + 1]
    cv = par[g, I_CV]
    rg = par[g, I_R]
    off = par[g, P_OFF]
    gpar = (cv, rg, off, par[f, P_KL])
    mdl = model[f]
    reg = W[i, W_REG0]
    if mdl == MODEL_LIQUID:
        reg = 0.0
    elif mdl == MODEL_IDEAL or m_f <= 0.0:
        reg = 1.0
    t = W[i, W_T0]
    u = W[i, W_U0]
    w = W[i, W_LT0]
    p = 0.0
    y = vol
    m_liq = 0.0
    rho_liq = 1.0
    ps = 0.0
    mu = 1e-5
    gam = 1.4
    xq = 2.0
    tried_wet = False
    for _ in range(3):
        if reg == 0.0:
            tried_wet = True
            ok = True
            if mdl == MODEL_LIQUID:
                # the liquid's energy is linear in t and it has no vapor: t is explicit
                c = m_f * par[f, L_CV] + m_g * cv
                t = (e - m_f * (par[f, L_U0] - par[f, L_CV] * par[f, L_T0]) - m_g * off) / max(c, 1e-30)
                t = min(max(t, 1.0), 5000.0)
                res, w, y, p, m_liq, rho_liq, ps, mu, gam, u_l, u_v = _mixed_wet(f, t, vol, m_f, m_g, e, w, fl, gpar)
            else:
                # liquid present: solve the temperature by secant steps (the residual
                # rises with t), kept inside a bracket
                t_lo = satg[f, 0]
                t_top = min(satg[f, 1], satg[f, 2] - 1e-3)
                t_hi = t_top
                t = min(max(t, t_lo), t_hi)
                res, w, y, p, m_liq, rho_liq, ps, mu, gam, u_l, u_v = _mixed_wet(f, t, vol, m_f, m_g, e, w, fl, gpar)
                dt = 1e-4 * t if t + 1e-4 * t <= t_hi else -1e-4 * t
                d = (_mixed_wet(f, t + dt, vol, m_f, m_g, e, w, fl, gpar)[0] - res) / dt
                for _ in range(MIX_ITER):
                    if res > 0.0:
                        t_hi = t
                    else:
                        t_lo = t
                    nt = t - res / d if d > 0.0 else 0.5 * (t_lo + t_hi)
                    if not (t_lo <= nt <= t_hi):
                        nt = 0.5 * (t_lo + t_hi)
                    if abs(nt - t) < 1e-10 * t:
                        break
                    r_n, w, y, p, m_liq, rho_liq, ps, mu, gam, u_l, u_v = _mixed_wet(
                        f, nt, vol, m_f, m_g, e, w, fl, gpar)
                    d = (r_n - res) / (nt - t)
                    t = nt
                    res = r_n
                # too much energy for any liquid to stay, or too little mass
                if (res < 0.0 and t >= t_top * (1.0 - 1e-9)) or m_liq < 0.0:
                    ok = False
            if ok:
                u = (m_liq * u_l + (m_f - m_liq) * u_v) / max(m_f, 1e-30)
                xq = -1.0 if mdl == MODEL_LIQUID else (m_f - m_liq) / max(m_f, 1e-30)
                break
            reg = 1.0
            u = e / max(m_f, 1e-30)
        else:
            # no liquid: both gases fill the node
            if mdl == MODEL_IDEAL or m_f <= 0.0:
                c = m_g * cv
                if mdl == MODEL_IDEAL:
                    c += m_f * par[f, I_CV]
                t = min(max((e - m_g * off) / max(c, 1e-30), 1.0), 5000.0)
                u = par[f, I_CV] * t if mdl == MODEL_IDEAL else W[i, W_U0]
                p_f = m_f * par[f, I_R] * t / vol if mdl == MODEL_IDEAL else 0.0
                mu = par[f, I_MU] if mdl == MODEL_IDEAL else par[g, I_MU]
                gam = par[f, I_GAMMA] if mdl == MODEL_IDEAL else par[g, I_GAMMA]
                xq = 2.0
            else:
                rho = m_f / vol
                # secant steps on u; the gas term is small beside m_f u unless m_f is
                # a trace, so start from dT/du ~ 1/cv of the fluid as a gas
                p_f, t, xq, gam, mu = state(f, rho, u, model, par, tab, grid, ltab, lgrid, rhol)
                res = m_f * u + m_g * (cv * t + off) - e
                d = m_f + m_g * cv / par[f, I_CV]
                for _ in range(MIX_ITER):
                    nu = u - res / d
                    nu = min(max(nu, u - 2e5), u + 2e5)
                    if abs(nu - u) < 1e-10 * max(abs(u), 1e5):
                        break
                    p_f, t, xq, gam, mu = state(f, rho, nu, model, par, tab, grid, ltab, lgrid, rhol)
                    r_n = m_f * nu + m_g * (cv * t + off) - e
                    if r_n != res:
                        d = (r_n - res) / (nu - u)
                    if d < m_f:
                        d = m_f  # t never falls as u rises
                    u = nu
                    res = r_n
                if not tried_wet and t < satg[f, 2] and xq <= 1.0:
                    reg = 0.0  # it would condense: there is liquid
                    continue
            p = p_f + m_g * rg * t / vol
            y = vol
            m_liq = 0.0
            rho_liq = m_f / vol
            ps = 0.0
            break
    m = m_f + m_g
    p_g = m_g * rg * t / max(y, 1e-300)
    W[i, W_P] = p
    W[i, W_T] = t
    W[i, W_V] = vol
    W[i, W_TW] = x[sx + 2] if NI[i, N_WALL] else NF[i, NF_TAMB]
    W[i, W_X] = xq
    W[i, W_RHO] = rho_liq
    W[i, W_U] = u
    W[i, W_G] = gam
    W[i, W_MU] = mu
    u_g = cv * t + off
    # what the own fluid leaves with: its energy plus its share of the flow work
    # (the liquid at the total pressure, its vapor at the vapor pressure)
    W[i, W_H] = (e - m_g * u_g + p * vol - p_g * y) / max(m_f, 1e-30)
    r = m / NF[i, NF_MREF]
    W[i, W_S] = r * r / (1.0 + r * r)
    W[i, W_HG] = u_g + rg * t
    W[i, W_SG] = y / vol
    W[i, W_Y] = m_f / m if m > 0.0 else 1.0
    W[i, W_RHOM] = m / vol
    W[i, W_LT] = w
    W[i, W_REG] = reg
    W[i, W_PS] = ps
    W[i, W_GF] = p_g / p * y / vol if p > 0.0 else 0.0


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
        W[i, W_SG] = 0.0
        W[i, W_Y] = 1.0
        W[i, W_RHOM] = rho
        return
    if NI[i, N_GX] >= 0:
        _mixed_one(i, x, NI, NF, fl, W)
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
        # outflow throttle near vacuum (see NF_MREF); the squared ratio keeps it off
        # (under 0.01 %) at ordinary pressures and only bites close to vacuum
        r = max(m, 0.0) / NF[i, NF_MREF]
        W[i, W_S] = r * r / (1.0 + r * r)
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
    W[i, W_SG] = 0.0
    W[i, W_Y] = 1.0
    W[i, W_RHOM] = rho


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
def _soft_unit(o):
    """Clamp to [0, 1] with rounded corners (C1), so pressure-actuated valves don't chatter in Newton."""
    w = 0.02
    if o <= -w:
        return 0.0
    if o < w:
        return (o + w) * (o + w) / (4.0 * w)
    if o <= 1.0 - w:
        return o
    if o < 1.0 + w:
        return 1.0 - (1.0 + w - o) * (1.0 + w - o) / (4.0 * w)
    return 1.0


@inl
def _opening(k, LI, LF, LV, pa, pb):
    c = LI[k, L_CTRL]
    if c == CTRL_VALVE:
        return LV[k, V_POS]
    if c == CTRL_CHECK or c == CTRL_RELIEF:
        return _soft_unit((pa - pb - LF[k, LF_P1]) / LF[k, LF_P2])
    if c == CTRL_REGULATOR:
        if pa <= pb:
            return 0.0
        return _soft_unit((LF[k, LF_P1] - pb) / LF[k, LF_P2])
    return 1.0


@inl
def _deliver(dn, sp, q, h, t, NI, NF, W, fl, fn, fv, slot):
    """Add ``q`` of fluid ``sp`` carrying enthalpy ``h`` (at temperature ``t``) into node dn,
    whose rates go in ``slot`` of fn/fv (its tank ullage, if any, in slot + 2).

    It goes to dn's own fluid or its purge gas, whichever it is. Tank liquid passes a
    gas on to its ullage. A fluid the node does not hold is counted as the node's own
    fluid, at the same temperature (the network builder warns where this can happen).
    """
    model, par, tab, grid, ltab, lgrid, rhol, sat, satg = fl
    kind = NI[dn, N_KIND]
    if kind == NODE_BOUNDARY:
        return
    tgt = dn
    if kind == NODE_TANK_LIQUID:
        pd = NI[dn, N_PARTNER]
        fn[slot + 2] = pd
        if sp == NI[dn, N_FLUID]:
            # the incoming liquid pushes the ullage gas up
            w = W[pd, W_P] * q / NF[dn, NF_RHOREF]
            fv[3 * slot + 7] += w
            fv[3 * slot + 1] -= w
        else:
            slot += 2
            tgt = pd
    fn[slot] = tgt
    if sp == NI[tgt, N_FLUID]:
        fv[3 * slot] += q
        fv[3 * slot + 1] += q * h
    elif sp == NI[tgt, N_GAS]:
        fv[3 * slot + 2] += q
        fv[3 * slot + 1] += q * h
    else:
        f = NI[tgt, N_FLUID]
        fv[3 * slot] += q
        fv[3 * slot + 1] += q * (par[f, P_CPA] * t + par[f, P_OFF])


@inl
def _smooth01(z):
    if z <= 0.0:
        return 0.0
    if z >= 1.0:
        return 1.0
    return z * z * (3.0 - 2.0 * z)


@inl
def _q_side(n, d, NI, W, fl):
    """What sets the flow rate out of node n through a port drawing ``d``.

    Returns (density, gamma, liquid-like weight, vapor pressure, flow multiplier).
    """
    par = fl[1]
    if NI[n, N_GX] >= 0:
        g = NI[n, N_GAS]
        liquid = W[n, W_REG] == 0.0
        y = W[n, W_Y]
        # liquid-like by mass, as a single-fluid node: the gas share of what flows
        xg = 1.0 - y * (1.0 - min(max(W[n, W_X], 0.0), 1.0))
        w_l = _smooth01((0.95 - xg) / 0.1) if liquid else 0.0
        gam = par[g, I_GAMMA] if liquid else y * W[n, W_G] + (1.0 - y) * par[g, I_GAMMA]
        rho, psat = W[n, W_RHOM], W[n, W_PS]
    else:
        rho, h, mu, gam, liquid, psat = _draw(n, d, NI, W, fl)
        # an emptying tank liquid region hands its outflow over to the ullage gas
        if NI[n, N_KIND] == NODE_TANK_LIQUID and W[n, W_S] < 0.5:
            rho, h, mu, gam, liquid, psat = _draw(NI[n, N_PARTNER], DRAW_VAPOR, NI, W, fl)
        w_l = 1.0 if liquid else 0.0
    s = W[n, W_S] if NI[n, N_KIND] == NODE_VOLUME else 1.0
    return rho, gam, w_l, psat, s


@inl
def _q_orifice(cda, p_up, p_dn, side):
    rho, gam, w_l, psat, s = side
    q = 0.0
    if w_l > 0.0:
        q += w_l * _orifice_flow(cda, p_up, p_dn, rho, gam, True, psat)
    if w_l < 1.0:
        q += (1.0 - w_l) * _orifice_flow(cda, p_up, p_dn, rho, gam, False, 0.0)
    return q * s


@inl
def _exchange(src, d_src, dst, q, NI, NF, W, fl, fn, fv):
    """A stream ``q`` from volume src (slot 1) back into volume dst (slot 0)."""
    fn[0] = dst
    fn[1] = src
    t = W[src, W_T]
    if NI[src, N_GX] >= 0:
        y = W[src, W_Y]
        h = W[src, W_H]
        h_g = W[src, W_HG]
        fv[3] -= y * q
        fv[4] -= y * q * h + (1.0 - y) * q * h_g
        fv[5] -= (1.0 - y) * q
        _deliver(dst, NI[src, N_FLUID], y * q, h, t, NI, NF, W, fl, fn, fv, 0)
        _deliver(dst, NI[src, N_GAS], (1.0 - y) * q, h_g, t, NI, NF, W, fl, fn, fv, 0)
    else:
        h = _draw(src, d_src, NI, W, fl)[1]
        fv[3] -= q
        fv[4] -= q * h
        _deliver(dst, NI[src, N_FLUID], q, h, t, NI, NF, W, fl, fn, fv, 0)


@inl
def _link_flux(k, x, NI, NF, LI, LF, LV, fl, W, fn, fv):
    """Flows carried by link k.

    ``fn[0..3]`` are the nodes it changes (upstream, downstream, and the ullage
    partners of either when it is tank liquid; -1 for none) and ``fv[3s..3s + 2]``
    their own-fluid mass, energy and purge-gas mass rates;
    ``fv[FV_ACC]`` is d(mdot)/dt of a pipe.
    Returns (signed mass flow, opening).
    """
    model, par, tab, grid, ltab, lgrid, rhol, sat, satg = fl
    for s in range(4):
        fn[s] = -1
    for s in range(N_FV):
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
    mixed = NI[up, N_GX] >= 0
    d_up = LI[k, L_DRAWA] if fwd else LI[k, L_DRAWB]
    side = _q_side(up, d_up, NI, W, fl)
    rho_q, gam_q, w_l = side[0], side[1], side[2]
    s_up = W[up, W_S]
    up_kind = NI[up, N_KIND]
    pt = NI[up, N_PARTNER]
    y = 1.0
    if mixed:
        h = W[up, W_H]
        y = W[up, W_Y]
        mu_q = 1.0 / (y / max(W[up, W_MU], 1e-9) + (1.0 - y) / par[NI[up, N_GAS], I_MU])
    else:
        rho, h, mu_q, gam, liquid, psat = _draw(up, d_up, NI, W, fl)
        if up_kind == NODE_TANK_LIQUID and s_up < 0.5:
            mu_q = _draw(pt, DRAW_VAPOR, NI, W, fl)[2]
    p_up = W[up, W_P]
    p_dn = W[dn, W_P]
    back = 0.0
    if pipe:
        area = LF[k, LF_AREA]
        r = _friction_r(mdot, rho_q, mu_q, area, LF[k, LF_DIAM], LF[k, LF_LEN], LF[k, LF_ROUGH], LF[k, LF_K])
        acc = area / LF[k, LF_LEN] * (pa - pb - r)
        if w_l < 0.5:
            m_ch = _choked_flux(p_up, rho_q, gam_q) * area
            if abs(mdot) > m_ch:
                acc -= (abs(mdot) - m_ch) * (1.0 if mdot > 0 else -1.0) / CHOKE_TAU
        fv[FV_ACC] = acc
        q = abs(mdot)
        if up_kind == NODE_VOLUME:
            q *= s_up
            mdot = q if fwd else -q
    else:
        cda = LF[k, LF_CDA] * o
        q = _q_orifice(cda, p_up, p_dn, side)
        z = (p_up - p_dn) / max(DP_LIN, R_LIN * p_up)
        if z < 1.0 and cda > 0.0:
            # near zero flow, blend toward the rate the other side would give: the
            # upstream properties switch sides at zero flow, and an unblended switch
            # puts a kink in the flow law that Newton circles across
            d_dn = LI[k, L_DRAWB] if fwd else LI[k, L_DRAWA]
            q_dn = _q_orifice(cda, p_up, p_dn, _q_side(dn, d_dn, NI, W, fl)) * side[4]
            a_up = 0.5 + 0.5 * z
            q = a_up * q + (1.0 - a_up) * q_dn
            if (mixed or NI[dn, N_GX] >= 0) and up_kind == NODE_VOLUME and NI[dn, N_KIND] == NODE_VOLUME:
                # smooth upwinding where a purge gas meets another fluid: near zero flow
                # the flow is two opposite streams, each carrying its own side's fluid,
                # so what crosses does not jump from one side's to the other's as the
                # flow reverses (Newton circles such a jump). It fades out by z = 1.
                eps = EXCH * _q_orifice(cda, p_up, p_up - max(DP_LIN, R_LIN * p_up), side) * (1.0 - _smooth01(z))
                back = 0.5 * (math.sqrt(q * q + eps * eps) - q) * W[dn, W_S]
        mdot = q if fwd else -q
    if back > 0.0:
        _exchange(dn, LI[k, L_DRAWB] if fwd else LI[k, L_DRAWA], up, back, NI, NF, W, fl, fn, fv)
        q += back
    if q == 0.0:
        return mdot, o
    f_up = NI[up, N_FLUID]
    t_up = W[up, W_T]
    if up_kind == NODE_TANK_LIQUID:
        # liquid share at the liquid enthalpy, the rest as ullage gas
        h_g = _draw(pt, DRAW_VAPOR, NI, W, fl)[1]
        fn[0] = up
        fv[0] -= s_up * q
        fv[1] -= s_up * q * h
        fn[2] = pt
        fv[6] -= (1.0 - s_up) * q
        fv[7] -= (1.0 - s_up) * q * h_g
        # the ullage does the work of pushing the liquid out
        w = W[pt, W_P] * fv[0] / NF[up, NF_RHOREF]
        fv[7] += w
        fv[1] -= w
        _deliver(dn, f_up, s_up * q, h, t_up, NI, NF, W, fl, fn, fv, 1)
        if s_up < 1.0:
            _deliver(dn, NI[pt, N_FLUID], (1.0 - s_up) * q, h_g, W[pt, W_T], NI, NF, W, fl, fn, fv, 1)
    elif mixed:
        q_g = (1.0 - y) * q
        h_g = W[up, W_HG]
        fn[0] = up
        fv[0] -= y * q
        fv[1] -= y * q * h + q_g * h_g
        fv[2] -= q_g
        _deliver(dn, f_up, y * q, h, t_up, NI, NF, W, fl, fn, fv, 1)
        _deliver(dn, NI[up, N_GAS], q_g, h_g, t_up, NI, NF, W, fl, fn, fv, 1)
    else:
        if up_kind != NODE_BOUNDARY:
            fn[0] = up
            fv[0] -= q
            fv[1] -= q * h
        _deliver(dn, f_up, q, h, t_up, NI, NF, W, fl, fn, fv, 1)
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
            out[sx] += sign * fv[3 * s]
            out[sx + 1] += sign * fv[3 * s + 1]
            gx = NI[i, N_GX]
            if gx >= 0:
                out[gx] += sign * fv[3 * s + 2]
    if LI[k, L_KIND] == LINK_PIPE:
        out[LI[k, L_SX]] += sign * fv[FV_ACC]


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
        nout[i, O_X] = W[i, W_X]
        nout[i, O_TW] = W[i, W_TW]
        gx = NI[i, N_GX]
        if gx >= 0:
            y = W[i, W_Y]
            nout[i, O_T] = W[i, W_T]
            nout[i, O_RHO] = W[i, W_RHOM]
            nout[i, O_H] = y * W[i, W_H] + (1.0 - y) * W[i, W_HG]
            nout[i, O_M] = max(x[NI[i, N_SX]], 0.0) + max(x[gx], 0.0)
            nout[i, O_GAS] = W[i, W_GF]
        else:
            nout[i, O_T] = W[i, W_T]
            nout[i, O_RHO] = W[i, W_RHO]
            nout[i, O_H] = W[i, W_H]
            nout[i, O_M] = x[NI[i, N_SX]] if NI[i, N_KIND] != NODE_BOUNDARY else 0.0
            nout[i, O_GAS] = 0.0


@jit
def rhs_repeat(n, x, xdot, NI, NF, LI, LF, LV, fl, W, nout, lout, FN, FV, NS):
    """``n`` evaluations of ``rhs``, to time it from Python."""
    for _ in range(n):
        rhs(x, xdot, NI, NF, LI, LF, LV, fl, W, nout, lout, FN, FV, NS)


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


TRACE = 1e-3  # a mixed node's fluid or gas below this share of its scale is a trace: never chopped for


@jit
def _chop(xk, bvec, iperm, NI, NF, fl, xscale):
    """Damp the Newton update (Appleyard-style chop).

    A node may not lose more than 80% of its mass in one iteration, and an update
    that crosses the saturated-liquid boundary stops a little past it, so the next
    iteration is linearized on the right side of the kink. The most restrictive
    node sets one factor for the whole update: scaling every state together keeps
    each flux's mass and energy leaving one node equal to what enters the next,
    so a damped iterate conserves like a full one. In a mixed node the same holds for
    its fluid and its purge gas, except for a trace of either, which may dip slightly
    below zero instead (it then counts as none). Returns True if anything was chopped.
    """
    model, par, tab, grid, ltab, lgrid, rhol, sat, satg = fl
    amin = 1.0
    for i in range(NI.shape[0]):
        if NI[i, N_KIND] != NODE_VOLUME or NI[i, N_PARTNER] >= 0:
            continue
        sx = NI[i, N_SX]
        gx = NI[i, N_GX]
        m0 = xk[sx]
        dm = bvec[iperm[sx]]
        du = bvec[iperm[sx + 1]]
        a = 1.0
        vol = NF[i, NF_VOL]
        if gx >= 0:
            mg0 = xk[gx]
            dmg = bvec[iperm[gx]]
            if mg0 > TRACE * xscale[gx] and mg0 + dmg < 0.2 * mg0:
                a = 0.8 * mg0 / -dmg
            if m0 > TRACE * xscale[sx]:
                if m0 + a * dm < 0.2 * m0:
                    a = 0.8 * m0 / -dm
            if a < amin:
                amin = a
            continue
        if not m0 > 0.0:
            continue  # the step will be rejected as non-physical
        if m0 + a * dm < 0.2 * m0:
            a = 0.8 * m0 / -dm
        f = NI[i, N_FLUID]
        u0 = xk[sx + 1] / m0
        m1 = m0 + a * dm
        u1 = (xk[sx + 1] + a * du) / m1
        l0 = _liquid_margin(f, m0 / vol, u0, model, grid, lgrid, rhol)
        l1 = _liquid_margin(f, m1 / vol, u1, model, grid, lgrid, rhol)
        if (l0 <= 0.0) != (l1 <= 0.0) and abs(l1) > 1e-3:
            c = l0 / (l0 - l1)
            c = c + 0.1 * (1.0 - c)
            a *= c
        if a < amin:
            amin = a
    if amin >= 1.0:
        return False
    for i in range(bvec.shape[0]):
        bvec[i] *= amin
    return True


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
NEWTON_REJAC = 1.0  # first-update size above which the Jacobian is rebuilt at the iterate
ACCEPT_TOL = 1e-2  # an unconverged step is still taken if its last update was this small
JAC_MAX_AGE = 10  # steps a Jacobian may be reused while Newton converges quickly
JAC_COST = 5.0  # a Jacobian plus factorization, in right-hand-side evaluations
MIN_SUBSTEP = 1.0 / 16.0  # smallest retry, as a fraction of dt
WORK_BANK = 4.0  # unspent budget carried to later steps, in step budgets
DAMP_MIN = 0.125  # smallest damping of a Newton update that keeps overshooting
DAMP_EXTRA = 6  # extra iterations a damped solve may take


@jit
def _newton(x, h, xk, NI, NF, LI, LF, LV, fl, W, nout, lout, FN, FV, NS, xscale,
            st, sc, iperm, kl, ku, jvals, fk, ab, piv, bvec, ctl):
    """One backward-Euler step of size h from x into xk by damped modified Newton.

    ``ctl``: [jac_fresh, lu_h, iterations of last solve, jac_age, need_jac, Jacobians made, stalls,
    argmax state, work used, work budget, banked work, this step's allowance]. Every iterate is
    a whole (possibly uniformly damped) Newton update, so each flux's mass and energy leave one
    node and enter the next: any iterate conserves like a converged one and is safe to force
    when the budget runs out.
    Returns success.
    """
    n = x.shape[0]
    nn = NI.shape[0]
    if ctl[4] > 0.0:
        ctl[8] += JAC_COST + 1.0
        rhs(x, fk, NI, NF, LI, LF, LV, fl, W, nout, lout, FN, FV, NS)
        _jacobian(x, NI, NF, LI, LF, LV, fl, W, FN, FV, NS, xscale, st, sc, jvals)
        ctl[0] = 1.0
        ctl[5] += 1.0
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
    damp = 1.0
    limit = NEWTON_MAX
    converged = False
    it = 0
    while it < limit:
        ctl[8] += 1.0
        if ctl[8] > ctl[11]:
            return False
        rhs(xk, fk, NI, NF, LI, LF, LV, fl, W, nout, lout, FN, FV, NS)
        for i in range(n):
            bvec[iperm[i]] = -(xk[i] - x[i] - h * fk[i])
        _band_solve(ab, piv, bvec, kl, ku, n)
        for i in range(n):
            if not math.isfinite(bvec[i]):
                return False  # before touching xk, so it is never left half-updated
        # the size of the full update measures how Newton converges, before any chop
        raw = 0.0
        for i in range(n):
            e = abs(bvec[iperm[i]]) / xscale[i]
            if e > raw:
                raw = e
        chopped = _chop(xk, bvec, iperm, NI, NF, fl, xscale)
        if it >= 1 and raw >= ACCEPT_TOL:
            if raw > 0.9 * err_prev:
                # the update is not shrinking: Newton is circling a switch (a check
                # valve snapping open and shut against a stiff liquid line). Shorter
                # steps settle where full ones overshoot; uniform, so it still conserves
                damp = max(0.5 * damp, DAMP_MIN)
                limit = NEWTON_MAX + DAMP_EXTRA
            elif raw < 0.5 * err_prev:
                damp = min(2.0 * damp, 1.0)
        if damp < 1.0:
            for i in range(n):
                bvec[i] *= damp
        err = 0.0
        for i in range(n):
            d = bvec[iperm[i]]
            xk[i] += d
            e = abs(d) / xscale[i]
            if e > err:
                err = e
                ctl[7] = i
        ctl[2] = it + 1
        if err < NEWTON_TOL and not chopped and damp == 1.0:
            converged = True
            break
        if it >= 1 and not chopped and damp == 1.0 and err < ACCEPT_TOL and err > 0.5 * err_prev:
            # stalled on a kink of the interpolated tables, already close: take it
            ctl[6] += 1.0
            converged = True
            break
        if it + 1 < limit and ((it == 0 and raw > NEWTON_REJAC) or (it >= 1 and raw > 0.25 * err_prev)):
            # a huge first update, or slow convergence: re-linearize at the current iterate
            ctl[8] += JAC_COST + 1.0
            rhs(xk, fk, NI, NF, LI, LF, LV, fl, W, nout, lout, FN, FV, NS)
            _jacobian(xk, NI, NF, LI, LF, LV, fl, W, FN, FV, NS, xscale, st, sc, jvals)
            ctl[0] = 1.0
            ctl[5] += 1.0
            ctl[3] = 0.0
            if not _factor(h, NI, jvals, st[0], st[1], iperm, kl, ku, ab, piv):
                return False
        err_prev = raw
        it += 1
    if not converged:
        if chopped or damp < 1.0 or err > ACCEPT_TOL:
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
        if NI[i, N_KIND] != NODE_VOLUME:
            continue
        m = x[NI[i, N_SX]]
        gx = NI[i, N_GX]
        if gx >= 0:
            m = max(m, 0.0) + max(x[gx], 0.0)
        if m <= 0.0:
            return False
    return True


@jit
def _keep_split(NI, W):
    """Remember mixed nodes' state at an accepted step, to start the next solves from."""
    for i in range(NI.shape[0]):
        if NI[i, N_GX] >= 0:
            W[i, W_U0] = W[i, W_U]
            W[i, W_LT0] = W[i, W_LT]
            W[i, W_T0] = W[i, W_T]
            W[i, W_REG0] = W[i, W_REG]


@jit
def advance(x, n_steps, dt, NI, NF, LI, LF, LV, fl, W, nout, lout, FN, FV, NS,
            xscale, st, sc, iperm, kl, ku,
            jvals, fk, ab, piv, bvec, xnew, ctl, stats, hist_p, hist_t, hist_m, hist_every):
    """Advance ``n_steps`` fixed steps of ``dt``.

    A step whose Newton iteration does not converge is retried as smaller
    sub-steps with a fresh Jacobian; simulated time stays on the fixed grid.
    Each step has a work budget (``ctl[9]``, in right-hand-side evaluations, plus up to
    ``WORK_BANK`` budgets that earlier easy steps left unspent, banked in ``ctl[10]``)
    so one hard step cannot stall real time: when it runs out, the last Newton iterate is taken if
    it is physical (every iterate is a whole, uniformly damped update, so it
    conserves mass and energy like a converged one), otherwise the state is held
    for the rest of the step. ``stats`` accumulates [sub-steps,
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
        # this step may spend its own budget plus what easy steps saved: a hard
        # transient gets a few times the budget while the average stays bounded
        ctl[11] = ctl[9] + ctl[10]
        while remaining > 1e-15:
            h = min(h, remaining)
            ok = _newton(x, h, xnew, NI, NF, LI, LF, LV, fl, W, nout, lout, FN, FV, NS, xscale,
                         st, sc, iperm, kl, ku, jvals, fk, ab, piv, bvec, ctl)
            stats[1] += ctl[2]
            stats[2] += ctl[5]
            ctl[5] = 0.0
            if not ok:
                stats[3] += 1.0
                if h > dt * MIN_SUBSTEP and ctl[8] < ctl[11]:
                    h *= 0.5
                    if ctl[0] == 0.0:  # retry on a fresh Jacobian unless this step already made one
                        ctl[4] = 1.0
                    continue
                # out of budget: take the iterate if it is physical, else hold. The
                # Jacobian is kept: a forced step must not make the next one pay for
                # a new Jacobian too, or a small budget never catches up
                stats[4] += 1.0
                if _physical(xnew, NI):
                    for i in range(x.shape[0]):
                        x[i] = xnew[i]
                    _keep_split(NI, W)
                break
            if not _physical(xnew, NI):  # converged onto a non-physical state: retry smaller
                stats[3] += 1.0
                if h > dt * MIN_SUBSTEP and ctl[8] < ctl[11]:
                    h *= 0.5
                    if ctl[0] == 0.0:
                        ctl[4] = 1.0
                    continue
                stats[4] += 1.0
                break
            for i in range(x.shape[0]):
                x[i] = xnew[i]
            _keep_split(NI, W)
            stats[5] += ctl[6]
            ctl[6] = 0.0
            remaining -= h
            stats[0] += 1.0
            if h < dt:
                h = min(2.0 * h, dt)
        ctl[10] = min(max(ctl[10] + ctl[9] - ctl[8], 0.0), WORK_BANK * ctl[9])
        if hist_every > 0 and (s + 1) % hist_every == 0 and hrow < hist_p.shape[0]:
            rhs(x, fk, NI, NF, LI, LF, LV, fl, W, nout, lout, FN, FV, NS)
            for i in range(NI.shape[0]):
                hist_p[hrow, i] = nout[i, O_P]
                hist_t[hrow, i] = nout[i, O_T]
            for k in range(nl):
                hist_m[hrow, k] = lout[k, LO_MDOT]
            hrow += 1
    rhs(x, fk, NI, NF, LI, LF, LV, fl, W, nout, lout, FN, FV, NS)
