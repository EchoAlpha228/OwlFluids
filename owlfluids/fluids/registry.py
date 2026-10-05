"""The fluids OwlFluids knows, and how each one is modelled at run time.

Three runtime models keep property evaluation cheap:

* ``liquid``: nearly incompressible liquid with a constant bulk modulus
  (water and ethanol by default).
* ``ideal``: ideal gas with constant heat capacities (opt-in for gases).
* ``table``: real-fluid lookup table on (density, internal energy) covering gas,
  liquid, supercritical and two-phase states (default for oxygen, methane,
  nitrogen and nitrous oxide, so GOX/LOX and GN2/LN2 are the same fluid).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import tables

MODEL_IDEAL = 0
MODEL_LIQUID = 1
MODEL_TABLE = 2
MODEL_CODES = {"ideal": MODEL_IDEAL, "liquid": MODEL_LIQUID, "table": MODEL_TABLE}

# ideal-gas params
I_R, I_CV, I_GAMMA, I_MU = range(4)
# liquid params
L_RHO0, L_K, L_PREF, L_CV, L_T0, L_U0, L_PSAT, L_MU = range(8)
N_PAR = 8

R_UNIVERSAL = 8.314462618

# canonical key -> CoolProp name, default model, models allowed
FLUIDS = {
    "water": ("Water", "liquid", ("liquid",)),
    "ethanol": ("Ethanol", "liquid", ("liquid",)),
    "oxygen": ("Oxygen", "table", ("table", "ideal")),
    "methane": ("Methane", "table", ("table", "ideal")),
    "nitrogen": ("Nitrogen", "table", ("table", "ideal")),
    "nitrous": ("NitrousOxide", "table", ("table", "ideal")),
}
TABLE_FLUIDS = [v[0] for v in FLUIDS.values() if "table" in v[2]]

ALIASES = {
    "gox": "oxygen", "lox": "oxygen", "o2": "oxygen", "oxygen": "oxygen",
    "gch4": "methane", "lch4": "methane", "ch4": "methane", "methane": "methane",
    "gn2": "nitrogen", "ln2": "nitrogen", "n2": "nitrogen", "nitrogen": "nitrogen",
    "n2o": "nitrous", "nitrous": "nitrous", "nitrous oxide": "nitrous", "nitrousoxide": "nitrous",
    "water": "water", "h2o": "water",
    "ethanol": "ethanol", "etoh": "ethanol", "c2h5oh": "ethanol",
}


def canonical(name: str) -> str:
    key = name.strip().lower().replace("_", " ")
    if key not in ALIASES:
        known = ", ".join(sorted(FLUIDS))
        raise ValueError(f"unknown fluid {name!r}; known fluids: {known}")
    return ALIASES[key]


@dataclass
class FluidSet:
    """The fluids one system uses, packed into arrays for the solver kernel."""

    names: list[str]
    models: list[str]
    backend: str = "HEOS"
    model: np.ndarray = field(init=False)
    par: np.ndarray = field(init=False)
    tab: np.ndarray = field(init=False)
    grid: np.ndarray = field(init=False)
    sat: np.ndarray = field(init=False)
    satg: np.ndarray = field(init=False)
    ltab: np.ndarray = field(init=False)
    lgrid: np.ndarray = field(init=False)
    rhol: np.ndarray = field(init=False)

    def __post_init__(self):
        nf = max(len(self.names), 1)
        self.model = np.zeros(nf, dtype=np.int64)
        self.par = np.zeros((nf, N_PAR))
        n_tab = sum(1 for m in self.models if m == "table")
        # non-table fluids get a 1x1 placeholder slice to keep the array small
        shape_r, shape_u = (tables.N_RHO, tables.N_U) if n_tab else (2, 2)
        shape_ul, shape_eta = (tables.N_UL, tables.N_ETA) if n_tab else (2, 2)
        self.tab = np.zeros((nf, tables.N_PROP, shape_r, shape_u))
        self.ltab = np.zeros((nf, tables.N_PROP, shape_ul, shape_eta))
        self.lgrid = np.ones((nf, 3))
        self.rhol = np.ones((nf, tables.N_UL))
        self.grid = np.ones((nf, 5))
        self.sat = np.zeros((nf, tables.N_SATPROP, tables.N_SAT))
        self.satg = np.ones((nf, 4))
        self._states = {}
        for i, (name, mdl) in enumerate(zip(self.names, self.models)):
            cp_name = FLUIDS[name][0]
            self.model[i] = MODEL_CODES[mdl]
            st = self._backend_state(cp_name)
            if mdl == "table":
                data = tables.load_or_build(cp_name, self.backend, verbose=True)
                self.tab[i] = data["tab"]
                self.grid[i] = data["grid"]
                self.sat[i] = data["sat"]
                self.satg[i] = data["sat_grid"]
                self.ltab[i] = data["ltab"]
                self.lgrid[i] = data["lgrid"]
                self.rhol[i] = data["rho_l_of_u"]
            elif mdl == "ideal":
                import CoolProp.CoolProp as CP

                st.update(CP.PT_INPUTS, 101325.0, 300.0)
                r = R_UNIVERSAL / st.molar_mass()
                cp0 = st.cp0mass()
                self.par[i, I_R] = r
                self.par[i, I_CV] = cp0 - r
                self.par[i, I_GAMMA] = cp0 / (cp0 - r)
                self.par[i, I_MU] = st.viscosity()
            else:
                import CoolProp.CoolProp as CP

                t0 = 293.15
                st.update(CP.PT_INPUTS, 101325.0, t0)
                rho0 = st.rhomass()
                drho_dp = st.first_partial_deriv(CP.iDmass, CP.iP, CP.iT)
                self.par[i, L_RHO0] = rho0
                self.par[i, L_K] = rho0 / drho_dp
                self.par[i, L_PREF] = 101325.0
                self.par[i, L_CV] = st.cpmass()
                self.par[i, L_T0] = t0
                self.par[i, L_U0] = st.umass()
                self.par[i, L_MU] = st.viscosity()
                st.update(CP.QT_INPUTS, 0.0, t0)
                self.par[i, L_PSAT] = st.p()

    def arrays(self) -> tuple:
        """The packed arrays in the order the kernel expects them."""
        return (self.model, self.par, self.tab, self.grid, self.ltab, self.lgrid, self.rhol, self.sat, self.satg)

    def index(self, name: str) -> int:
        return self.names.index(canonical(name))

    def _backend_state(self, cp_name: str):
        if cp_name not in self._states:
            self._states[cp_name] = tables._state(cp_name, self.backend)
        return self._states[cp_name]

    # ---- set-up time helpers (never called in the time loop) ----

    def rho_u_from_pt(self, f: int, p: float, t: float) -> tuple[float, float]:
        """Density and internal energy for a single-phase state at (P, T)."""
        mdl = self.model[f]
        if mdl == MODEL_IDEAL:
            return p / (self.par[f, I_R] * t), self.par[f, I_CV] * t
        if mdl == MODEL_LIQUID:
            rho = self.par[f, L_RHO0] * (1.0 + (p - self.par[f, L_PREF]) / self.par[f, L_K])
            u = self.par[f, L_U0] + self.par[f, L_CV] * (t - self.par[f, L_T0])
            return rho, u
        import CoolProp.CoolProp as CP

        st = self._backend_state(FLUIDS[self.names[f]][0])
        try:
            st.update(CP.PT_INPUTS, p, t)
        except ValueError:
            # (P, T) sits on the saturation line; take saturated vapor
            st.update(CP.QT_INPUTS, 1.0, t)
        return st.rhomass(), st.umass()

    def rho_u_saturated(self, f: int, t: float, liquid_volume_fraction: float) -> tuple[float, float]:
        """Mixture density and internal energy of a saturated liquid/vapor fill at T."""
        if self.model[f] != MODEL_TABLE:
            raise ValueError(f"{self.names[f]} uses the {self.models[f]} model, which has no saturation state")
        import CoolProp.CoolProp as CP

        st = self._backend_state(FLUIDS[self.names[f]][0])
        st.update(CP.QT_INPUTS, 0.0, t)
        rl, ul = st.rhomass(), st.umass()
        st.update(CP.QT_INPUTS, 1.0, t)
        rv, uv = st.rhomass(), st.umass()
        a = liquid_volume_fraction
        rho = a * rl + (1 - a) * rv
        u = (a * rl * ul + (1 - a) * rv * uv) / rho
        return rho, u

    def saturation_temperature(self, f: int, p: float) -> float:
        import CoolProp.CoolProp as CP

        st = self._backend_state(FLUIDS[self.names[f]][0])
        st.update(CP.PQ_INPUTS, p, 0.0)
        return st.T()

    def liquid_density(self, f: int, t: float) -> float:
        """Reference liquid density used for the liquid region of pressurized tanks."""
        mdl = self.model[f]
        if mdl == MODEL_LIQUID:
            return float(self.par[f, L_RHO0])
        if mdl == MODEL_IDEAL:
            raise ValueError(f"{self.names[f]} is modelled as an ideal gas and cannot be a tank liquid")
        import CoolProp.CoolProp as CP

        st = self._backend_state(FLUIDS[self.names[f]][0])
        st.update(CP.QT_INPUTS, 0.0, min(t, st.T_critical() - 1.0))
        return st.rhomass()
