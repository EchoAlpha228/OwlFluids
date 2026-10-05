"""Run-time property evaluation, compiled with Numba.

Every function takes the packed arrays from ``FluidSet`` so the solver kernel can
call them without touching Python objects.
"""

from __future__ import annotations

import math

import numba as nb

from .registry import (
    I_CV,
    I_GAMMA,
    I_MU,
    I_R,
    L_CV,
    L_K,
    L_MU,
    L_PREF,
    L_PSAT,
    L_RHO0,
    L_T0,
    L_U0,
    MODEL_IDEAL,
    MODEL_LIQUID,
)
from .tables import (
    GAMMA,
    MU,
    N_ETA,
    N_LIN,
    N_LOG,
    N_SAT,
    N_U,
    N_UL,
    S_GAMV,
    S_HL,
    S_HV,
    S_MUL,
    S_MUV,
    S_PSAT,
    S_RHOL,
    S_RHOV,
    P,
    T,
    X,
)

jit = nb.njit(cache=True, fastmath=False, nogil=True, _nrt=False, error_model="numpy")


@jit
def _rho_index(rho, rho_min, rho_split, rho_max):
    """Fractional grid index for a density (inverse of ``tables.rho_grid``)."""
    if rho <= rho_split:
        r = max(rho, 1e-12)
        return (math.log(r) - math.log(rho_min)) / (math.log(rho_split) - math.log(rho_min)) * (N_LOG - 1)
    return (N_LOG - 1) + (rho - rho_split) / (rho_max - rho_split) * N_LIN


@jit
def _bilinear(tab, f, prop, i0, a, j0, b):
    v00 = tab[f, prop, i0, j0]
    v01 = tab[f, prop, i0, j0 + 1]
    v10 = tab[f, prop, i0 + 1, j0]
    v11 = tab[f, prop, i0 + 1, j0 + 1]
    return (1 - a) * ((1 - b) * v00 + b * v01) + a * ((1 - b) * v10 + b * v11)


@jit
def _quality(tab, f, i0, a, j0, b, dense):
    """Quality code at a point: interpolated, except across the single-phase code jump.

    Single-phase states carry -1 above the critical density and 2 below it, so a
    cell whose corners are all single-phase but straddle that jump (supercritical
    states near the critical density) must not interpolate to a two-phase value.
    """
    x00 = tab[f, X, i0, j0]
    x01 = tab[f, X, i0, j0 + 1]
    x10 = tab[f, X, i0 + 1, j0]
    x11 = tab[f, X, i0 + 1, j0 + 1]
    single = 0
    for v in (x00, x01, x10, x11):
        if v < 0.0 or v > 1.0:
            single += 1
    if single == 4:
        return -1.0 if dense else 2.0
    return (1 - a) * ((1 - b) * x00 + b * x01) + a * ((1 - b) * x10 + b * x11)


@jit
def state(f, rho, u, model, par, tab, grid, ltab, lgrid, rhol):
    """(P, T, quality code, gamma, viscosity) of fluid ``f`` at (rho, u).

    Quality code: 0..1 two-phase, below 0 liquid or dense supercritical,
    above 1 vapor or light supercritical.
    """
    m = model[f]
    if m == MODEL_IDEAL:
        t = min(max(u / par[f, I_CV], 1.0), 5000.0)
        return rho * par[f, I_R] * t, t, 2.0, par[f, I_GAMMA], par[f, I_MU]
    if m == MODEL_LIQUID:
        p = par[f, L_PREF] + par[f, L_K] * (rho / par[f, L_RHO0] - 1.0)
        # a liquid line pulled below vapor pressure cavitates and holds near vapor
        # pressure; the exponential blend keeps dP/drho continuous for Newton
        ps = par[f, L_PSAT]
        d = 0.5 * ps + 1000.0
        if p < ps + d:
            p = ps + d * math.exp((p - ps - d) / d)
        t = par[f, L_T0] + (u - par[f, L_U0]) / par[f, L_CV]
        return p, t, -1.0, 1.0, par[f, L_MU]

    # keep wild Newton iterates inside the tables instead of extrapolating
    if u < grid[f, 3]:
        u = grid[f, 3]
    elif u > grid[f, 4]:
        u = grid[f, 4]
    # compressed liquid: boundary-fitted table on (u, eta)
    u_lo = lgrid[f, 0]
    u_hi = lgrid[f, 1]
    if u < u_hi:
        su = (u - u_lo) / (u_hi - u_lo) * (N_UL - 1)
        k0 = min(max(int(math.floor(su)), 0), N_UL - 2)
        c = su - k0
        r_l = (1 - c) * rhol[f, k0] + c * rhol[f, k0 + 1]
        if rho > r_l:
            rho_max = lgrid[f, 2]
            se = (rho - r_l) / (rho_max - r_l) * (N_ETA - 1)
            e0 = min(max(int(math.floor(se)), 0), N_ETA - 2)
            d = se - e0
            p = _bilinear(ltab, f, P, k0, c, e0, d)
            t = max(_bilinear(ltab, f, T, k0, c, e0, d), 1.0)
            g = _bilinear(ltab, f, GAMMA, k0, c, e0, d)
            mu = max(_bilinear(ltab, f, MU, k0, c, e0, d), 1e-7)
            return p, t, -1.0, min(max(g, 1.0), 3.0), mu

    # below the table's lowest density the fluid is a dilute gas: scale pressure
    # with density so a pumped-down volume approaches vacuum instead of extrapolating
    scale = 1.0
    if rho < grid[f, 0]:
        scale = max(rho, 0.0) / grid[f, 0]
        rho = grid[f, 0]
    s = _rho_index(rho, grid[f, 0], grid[f, 1], grid[f, 2])
    n_rho = N_LOG + N_LIN
    i0 = min(max(int(math.floor(s)), 0), n_rho - 2)
    a = s - i0
    su = (u - grid[f, 3]) / (grid[f, 4] - grid[f, 3]) * (N_U - 1)
    j0 = min(max(int(math.floor(su)), 0), N_U - 2)
    b = su - j0
    p = _bilinear(tab, f, P, i0, a, j0, b)
    t = max(_bilinear(tab, f, T, i0, a, j0, b), 1.0)
    x = _quality(tab, f, i0, a, j0, b, s >= N_LOG - 1)
    g = _bilinear(tab, f, GAMMA, i0, a, j0, b)
    mu = max(_bilinear(tab, f, MU, i0, a, j0, b), 1e-7)
    if scale < 1.0:
        x = 2.0
    elif t >= grid[f, 5] and 0.0 <= x <= 1.0:
        x = -1.0 if s >= N_LOG - 1 else 2.0  # no two-phase states above the critical temperature
    return max(p, 1.0) * scale, t, x, min(max(g, 1.0), 3.0), mu


@jit
def saturation(f, t, model, par, sat, satg):
    """Saturated (Psat, rho_l, rho_v, h_l, h_v, mu_l, mu_v, gamma_v, subcritical)."""
    if model[f] == MODEL_LIQUID:
        return par[f, L_PSAT], par[f, L_RHO0], 1.0, 0.0, 0.0, par[f, L_MU], 1e-5, 1.3, False
    if model[f] == MODEL_IDEAL:
        return 0.0, 1.0, 1.0, 0.0, 0.0, 1e-5, 1e-5, 1.4, False
    t_min = satg[f, 0]
    t_max = satg[f, 1]
    sub = t < satg[f, 2]
    s = (t - t_min) / (t_max - t_min) * (N_SAT - 1)
    k = min(max(int(math.floor(s)), 0), N_SAT - 2)
    a = min(max(s - k, 0.0), 1.0)
    out0 = (1 - a) * sat[f, S_PSAT, k] + a * sat[f, S_PSAT, k + 1]
    out1 = (1 - a) * sat[f, S_RHOL, k] + a * sat[f, S_RHOL, k + 1]
    out2 = (1 - a) * sat[f, S_RHOV, k] + a * sat[f, S_RHOV, k + 1]
    out3 = (1 - a) * sat[f, S_HL, k] + a * sat[f, S_HL, k + 1]
    out4 = (1 - a) * sat[f, S_HV, k] + a * sat[f, S_HV, k + 1]
    out5 = (1 - a) * sat[f, S_MUL, k] + a * sat[f, S_MUL, k + 1]
    out6 = (1 - a) * sat[f, S_MUV, k] + a * sat[f, S_MUV, k + 1]
    out7 = (1 - a) * sat[f, S_GAMV, k] + a * sat[f, S_GAMV, k + 1]
    return out0, out1, out2, out3, out4, out5, out6, out7, sub


@jit
def psat(f, t, model, par, sat, satg):
    """Saturation pressure only (the cheap part of ``saturation``)."""
    if model[f] == MODEL_LIQUID:
        return par[f, L_PSAT]
    if model[f] == MODEL_IDEAL:
        return 0.0
    t_min = satg[f, 0]
    s = (t - t_min) / (satg[f, 1] - t_min) * (N_SAT - 1)
    k = min(max(int(math.floor(s)), 0), N_SAT - 2)
    a = min(max(s - k, 0.0), 1.0)
    return (1 - a) * sat[f, S_PSAT, k] + a * sat[f, S_PSAT, k + 1]
