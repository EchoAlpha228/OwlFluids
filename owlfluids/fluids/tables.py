"""Builds the property lookup tables the solver reads at run time.

Each table-backed fluid gets a 2-D table on a (density, specific internal energy)
grid and a 1-D saturation table on temperature. Tables are built through
CoolProp's AbstractState, so the same code uses CoolProp's own equations of state
("HEOS") or NIST REFPROP ("REFPROP", when REFPROP is installed and CoolProp can
find it). Nothing here runs inside the time loop.

Build tables from the command line:

    python -m owlfluids.fluids.tables --backend REFPROP
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

# Grid sizes are shared by every fluid so the tables pack into one array.
N_LOG = 110  # log-spaced density points from the lowest density up to critical
N_LIN = 170  # linear density points from critical up to compressed liquid
N_RHO = N_LOG + N_LIN
N_U = 260
N_SAT = 200
N_UL = 200  # liquid table: internal energy points along the saturated-liquid line
N_ETA = 120  # liquid table: points from saturated liquid out to maximum compression
N_T_COLUMN = 420  # temperatures sampled per density column before re-gridding on u

# 2-D table property rows
P, T, X, GAMMA, MU = range(5)
N_PROP = 5
# saturation table rows
S_PSAT, S_RHOL, S_RHOV, S_HL, S_HV, S_MUL, S_MUV, S_GAMV = range(8)
N_SATPROP = 8

TABLE_VERSION = 3
P_MIN = 500.0  # Pa, lowest pressure the table must reach (vacuum-ish vent lines)
P_MAX = 400e5  # Pa, about 5,800 psi, covers 3,000 psi with margin
T_MAX = 500.0  # K

REFPROP_NAMES = {
    "Oxygen": "OXYGEN",
    "Methane": "METHANE",
    "Nitrogen": "NITROGEN",
    "NitrousOxide": "N2O",
    "Water": "WATER",
    "Ethanol": "ETHANOL",
}


def table_dir() -> Path:
    env = os.environ.get("OWLFLUIDS_TABLE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".owlfluids" / "tables"


def table_path(coolprop_name: str, backend: str) -> Path:
    return table_dir() / f"{coolprop_name}_{backend}_v{TABLE_VERSION}.npz"


def _state(coolprop_name: str, backend: str):
    from CoolProp import AbstractState

    if backend == "REFPROP":
        _point_coolprop_at_refprop()
    name = REFPROP_NAMES.get(coolprop_name, coolprop_name) if backend == "REFPROP" else coolprop_name
    try:
        return AbstractState(backend, name)
    except ValueError as e:
        if backend == "REFPROP":
            raise RuntimeError(
                f"REFPROP could not load {name}: {e}. Set RPPREFIX (or OWLFLUIDS_REFPROP_PATH) to the REFPROP "
                "folder, e.g. C:\\Program Files (x86)\\REFPROP, or use the HEOS backend.") from None
        raise


def _point_coolprop_at_refprop() -> None:
    """Tell CoolProp where REFPROP lives when an environment variable says so."""
    path = os.environ.get("OWLFLUIDS_REFPROP_PATH") or os.environ.get("RPPREFIX")
    if path:
        import CoolProp.CoolProp as CP

        CP.set_config_string(CP.ALTERNATIVE_REFPROP_PATH, path)


def rho_grid(rho_min: float, rho_split: float, rho_max: float) -> np.ndarray:
    """Density grid: log spaced in the gas range, linear in the liquid range.

    Liquids are stiff (a fraction of a percent of compression is tens of bar), so
    the liquid side gets dense linear spacing. Must match ``props._rho_index``.
    """
    lo = np.exp(np.linspace(np.log(rho_min), np.log(rho_split), N_LOG))
    hi = rho_split + (rho_max - rho_split) * np.arange(1, N_LIN + 1) / N_LIN
    return np.concatenate([lo, hi])


def build_table(coolprop_name: str, backend: str = "HEOS") -> dict[str, np.ndarray]:
    """Build the three tables for one fluid.

    * ``tab``: main table on (density, u) for gas, two-phase and supercritical states.
    * ``ltab``: compressed-liquid table on (u, eta), where eta runs from the
      saturated-liquid density at that u (eta = 0) to the maximum density (eta = 1).
      A liquid's pressure climbs tens of bar per kg/m3, so a plain (density, u) grid
      smears the liquid/two-phase boundary badly; fitting the grid to the boundary
      keeps saturated and subcooled liquids accurate.
    * ``sat``: saturation properties on temperature.
    """
    import CoolProp.CoolProp as CP
    from scipy.optimize import brentq

    st = _state(coolprop_name, backend)
    t_min = st.Ttriple() + 0.5
    t_crit = st.T_critical()
    rho_crit = st.rhomass_critical()

    st.update(CP.PT_INPUTS, P_MIN, T_MAX)
    rho_min = st.rhomass()
    # densest state needed: maximum pressure at the coldest temperature the
    # equation of state allows there (the melting line rises with pressure)
    t_dense = t_min
    while True:
        try:
            st.update(CP.PT_INPUTS, P_MAX, t_dense)
            break
        except ValueError:
            t_dense += 0.5
    rho_max = st.rhomass() * 1.01
    rhos = rho_grid(rho_min, rho_crit, rho_max)

    def dt_flash(rho, t):
        st.update(CP.DmassT_INPUTS, rho, t)
        return st

    # ---- saturation curves (fine grid for generation, coarse copy for run time)
    t_top = t_crit - 0.2
    t_fine = np.linspace(t_min, t_top, 2000)
    fine = np.zeros((6, len(t_fine)))  # rho_l, u_l, rho_v, u_v, p, h_l
    for k, tk in enumerate(t_fine):
        st.update(CP.QT_INPUTS, 0.0, tk)
        fine[0, k], fine[1, k], fine[4, k] = st.rhomass(), st.umass(), st.p()
        st.update(CP.QT_INPUTS, 1.0, tk)
        fine[2, k], fine[3, k] = st.rhomass(), st.umass()
    rho_l_f, u_l_f, rho_v_f, u_v_f = fine[0], fine[1], fine[2], fine[3]

    t_sat = np.linspace(t_min, t_top, N_SAT)
    sat = np.zeros((N_SATPROP, N_SAT))
    for k, tk in enumerate(t_sat):
        st.update(CP.QT_INPUTS, 0.0, tk)
        sat[S_PSAT, k] = st.p()
        sat[S_RHOL, k] = st.rhomass()
        sat[S_HL, k] = st.hmass()
        sat[S_MUL, k] = _safe(st.viscosity, 1e-4)
        st.update(CP.QT_INPUTS, 1.0, tk)
        sat[S_RHOV, k] = st.rhomass()
        sat[S_HV, k] = st.hmass()
        sat[S_MUV, k] = _safe(st.viscosity, 1e-5)
        sat[S_GAMV, k] = _safe(lambda: st.cpmass() / st.cvmass(), 1.4)

    def sat_at(t):
        """Interpolated saturation state at T (generation only)."""
        return (np.interp(t, t_fine, rho_l_f), np.interp(t, t_fine, u_l_f),
                np.interp(t, t_fine, rho_v_f), np.interp(t, t_fine, u_v_f))

    # saturated-liquid density as a function of u (u_l rises monotonically with T)
    u_l_lo, u_l_hi = u_l_f[0], u_l_f[-1]
    u_l_grid = np.linspace(u_l_lo, u_l_hi, N_UL)
    t_of_ul = np.interp(u_l_grid, u_l_f, t_fine)
    rho_l_of_u = np.interp(t_of_ul, t_fine, rho_l_f)

    def rho_l_u(u):
        return np.interp(u, u_l_grid, rho_l_of_u)

    def props_at(rho, t):
        s_ = dt_flash(rho, t)
        q = s_.Q()
        two = 0.0 <= q <= 1.0
        if two:
            ks = min(max(np.searchsorted(t_sat, t), 0), N_SAT - 1)
            gam = sat[S_GAMV, ks]
            mu = (1 - q) * sat[S_MUL, ks] + q * sat[S_MUV, ks]
            x = q
        else:
            gam = _safe(lambda: s_.cpmass() / s_.cvmass(), np.nan)
            mu = _safe(s_.viscosity, np.nan)
            x = -1.0 if rho > rho_crit else 2.0
        return s_.p(), t, x, gam, mu, s_.umass()

    # ---- main table: sample each density column on T, then re-grid on u
    t_col = np.linspace(t_min, T_MAX, N_T_COLUMN)
    cols = np.full((len(rhos), N_PROP + 1, N_T_COLUMN), np.nan)
    for i, rho in enumerate(rhos):
        for k, tk in enumerate(t_col):
            try:
                cols[i, :, k] = props_at(rho, tk)
            except ValueError:
                continue
        _fill_nans(cols[i])

    u_min = min(np.nanmin(cols[:, N_PROP, 0]), u_l_lo)
    u_max = np.nanmax(cols[:, N_PROP, -1])
    u_grid = np.linspace(u_min, u_max, N_U)
    tab = np.zeros((N_PROP, len(rhos), N_U))
    for i in range(len(rhos)):
        u_c = cols[i, N_PROP]
        order = np.argsort(u_c)
        for prop in range(N_PROP):
            tab[prop, i] = _interp_extrap(u_grid, u_c[order], cols[i, prop][order])

    # Quality at two-phase points comes straight from the saturation relations.
    # Re-gridding a column on u blends the last two-phase sample with the first
    # compressed-liquid one, which would give near-full saturated tanks a liquid
    # code (x < 0), so they would vent liquid from their top port.
    #
    # Pressure and temperature there keep the blended values on purpose. Within
    # about one grid cell of saturated liquid (tank fills above ~98 %) pressure
    # reads a few percent high, and that slope is what keeps Newton stable in
    # small flashing junctions; exact, flat two-phase pressures there made the
    # LOX feed of the test stand fail most steps.
    #
    # Main-table points inside the compressed-liquid region are only used when a
    # two-phase state sits in a cell next to the boundary, so they get the
    # two-phase relations continued past saturated liquid; those cells then
    # interpolate smoothly instead of blending in liquid pressures.
    rho_v_min = rho_v_f.min()
    for i, rho in enumerate(rhos):
        if rho < rho_v_min:
            continue  # thinner than any saturated vapor: never two-phase
        for j, u in enumerate(u_grid):
            if u >= u_l_hi and rho > rho_l_u(u_l_hi):
                continue

            def g(t, rho=rho, u=u):
                rl, ul, rv, uv = sat_at(t)
                x = (1 / rho - 1 / rl) / (1 / rv - 1 / rl)
                return ul + x * (uv - ul) - u

            try:
                t_c = brentq(g, t_min, t_top, xtol=1e-6)
            except ValueError:
                continue
            rl, ul, rv, uv = sat_at(t_c)
            x = (1 / rho - 1 / rl) / (1 / rv - 1 / rl)
            if x > 1.0 or (x < 0.0 and rho <= rho_l_u(u)):
                continue  # vapor, or a liquid-side root on the wrong side of the boundary
            if x >= 0.0:
                tab[X, i, j] = x
                continue
            ks = min(max(np.searchsorted(t_sat, t_c), 0), N_SAT - 1)
            tab[P, i, j] = np.interp(t_c, t_fine, fine[4])
            tab[T, i, j] = t_c
            tab[X, i, j] = x
            tab[GAMMA, i, j] = sat[S_GAMV, ks]
            tab[MU, i, j] = sat[S_MUL, ks]

    # ---- compressed-liquid table on (u, eta)
    ltab = np.full((N_PROP, N_UL, N_ETA), np.nan)
    etas = np.linspace(0.0, 1.0, N_ETA)
    for k, u in enumerate(u_l_grid):
        rl = rho_l_of_u[k]
        for j, eta in enumerate(etas):
            rho = rl + eta * (rho_max - rl)
            if j == 0:
                t_k = t_of_ul[k]
                rho = rl
                ks = min(max(np.searchsorted(t_sat, t_k), 0), N_SAT - 1)
                ltab[:, k, j] = (np.interp(t_k, t_fine, fine[4]), t_k, 0.0, sat[S_GAMV, ks], sat[S_MUL, ks])
                continue

            def gu(t, rho=rho, u=u):
                return dt_flash(rho, t).umass() - u

            try:
                lo = t_min
                if gu(lo) > 0:  # colder than the triple point would be needed
                    continue
                t_k = brentq(gu, lo, T_MAX, xtol=1e-6)
                p_, t_, _, gam, mu, _ = props_at(rho, t_k)
                ltab[:, k, j] = (p_, t_, -1.0, gam, mu)
            except ValueError:
                continue
    for prop in range(N_PROP):
        _fill_nans_2d(ltab[prop])

    for arr in (tab, ltab):
        arr[T] = np.maximum(arr[T], 1.0)
        arr[GAMMA] = np.nan_to_num(np.clip(arr[GAMMA], 1.0, 3.0), nan=1.4)
        arr[MU] = np.maximum(np.nan_to_num(arr[MU], nan=1e-4), 1e-7)

    return {
        "tab": tab,
        "grid": np.array([rho_min, rho_crit, rho_max, u_min, u_max]),
        "ltab": ltab,
        "lgrid": np.array([u_l_lo, u_l_hi, rho_max]),
        "rho_l_of_u": rho_l_of_u,
        "sat": sat,
        "sat_grid": np.array([t_min, t_top, t_crit, rho_crit]),
        "molar_mass": np.array(st.molar_mass()),
        "backend": np.array(backend),
    }


def _fill_nans_2d(a: np.ndarray) -> None:
    """Fill holes in a 2-D table: along eta first, then along u."""
    for row in a:
        _fill_nans(row[None, :])
    for col in a.T:
        _fill_nans(col[None, :])


def _safe(fn, default):
    try:
        v = fn()
        return v if np.isfinite(v) else default
    except ValueError:
        return default


def _fill_nans(col: np.ndarray) -> None:
    """Fill failed samples in one density column from their neighbours along T."""
    for row in col:
        bad = ~np.isfinite(row)
        if bad.all():
            row[:] = 0.0
        elif bad.any():
            idx = np.arange(len(row))
            row[bad] = np.interp(idx[bad], idx[~bad], row[~bad])


def _interp_extrap(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    y = np.interp(x, xp, fp)
    lo = x < xp[0]
    hi = x > xp[-1]
    if lo.any():
        s = (fp[1] - fp[0]) / max(xp[1] - xp[0], 1e-12)
        y[lo] = fp[0] + s * (x[lo] - xp[0])
    if hi.any():
        s = (fp[-1] - fp[-2]) / max(xp[-1] - xp[-2], 1e-12)
        y[hi] = fp[-1] + s * (x[hi] - xp[-1])
    return y


def load_or_build(coolprop_name: str, backend: str = "HEOS", verbose: bool = False) -> dict:
    path = table_path(coolprop_name, backend)
    if path.exists():
        with np.load(path) as data:
            return {k: data[k] for k in data.files}
    if verbose:
        print(f"building {coolprop_name} property table with {backend} (one time, cached in {path.parent})")
    data = build_table(coolprop_name, backend)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **data)
    os.replace(tmp, path)
    return data


def main(argv: list[str] | None = None) -> None:
    from .registry import TABLE_FLUIDS

    ap = argparse.ArgumentParser(description="Build OwlFluids property tables")
    ap.add_argument("--backend", default="HEOS", help="HEOS (CoolProp) or REFPROP")
    ap.add_argument("--force", action="store_true", help="rebuild even if cached")
    ap.add_argument("fluids", nargs="*", help="CoolProp fluid names (default: all)")
    args = ap.parse_args(argv)
    for name in args.fluids or TABLE_FLUIDS:
        path = table_path(name, args.backend)
        if args.force and path.exists():
            path.unlink()
        load_or_build(name, args.backend, verbose=True)
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
