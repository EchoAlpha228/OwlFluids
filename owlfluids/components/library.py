"""Component definitions.

Each component type declares its ports and parameters and knows how to turn
itself into solver nodes and links through a ``NetworkBuilder``. Adding a new
component means adding a class here; the solver never changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .. import units
from ..core.layout import (
    CTRL_CHECK, CTRL_FIXED, CTRL_REGULATOR, CTRL_RELIEF, CTRL_VALVE,
    DRAW_LIQUID, DRAW_MIX, DRAW_VAPOR,
)

CV_TO_CDA = 1.6987e-5  # m^2 of effective flow area per unit of valve Cv
STEEL = {"density": 8000.0, "cp": 500.0}
MATERIALS = {
    "steel": STEEL,
    "stainless": STEEL,
    "aluminum": {"density": 2700.0, "cp": 900.0},
    "aluminium": {"density": 2700.0, "cp": 900.0},
    "copper": {"density": 8960.0, "cp": 385.0},
    "brass": {"density": 8500.0, "cp": 380.0},
}
FITTING_K = {
    "elbow90": 0.9, "elbow45": 0.4, "tee_run": 0.4, "tee_branch": 1.8,
    "union": 0.08, "reducer": 0.5, "expander": 1.0, "bulkhead": 0.2, "quick_disconnect": 1.5,
}


@dataclass(frozen=True)
class Param:
    name: str
    dimension: str | None  # units.parse dimension, None for plain numbers or text
    default: object = None
    help: str = ""
    choices: tuple = ()


class ComponentType:
    """Base class. Subclasses set ``ports``, ``params`` and implement ``build``."""

    type_name = ""
    label = ""
    category = ""
    ports: tuple[str, ...] = ()
    params: tuple[Param, ...] = ()
    is_valve = False  # appears in the valve control panel
    merges_ports = False  # all ports are the same point (tees, crosses)

    @classmethod
    def param(cls, comp, name):
        """A parameter in SI, or its default."""
        spec = next((p for p in cls.params if p.name == name), None)
        if spec is None:
            raise KeyError(f"{cls.type_name} has no parameter {name!r}")
        raw = comp.params.get(name, spec.default)
        if raw is None:
            return None
        if spec.dimension in (None, "text", "choice", "bool"):
            return raw
        try:
            return units.parse(raw, spec.dimension)
        except ValueError as e:
            raise ValueError(f"{comp.id}.{name}: {e}") from None

    @classmethod
    def require(cls, comp, name):
        v = cls.param(comp, name)
        if v is None:
            raise ValueError(f"{comp.id} ({cls.type_name}) needs '{name}'")
        return v

    @classmethod
    def build(cls, b, comp):  # pragma: no cover - abstract
        raise NotImplementedError


def _flow_area(cls, comp):
    """Effective flow area (Cd*A) from cv, cda, or diameter and cd."""
    cv = cls.param(comp, "cv") if any(p.name == "cv" for p in cls.params) else None
    if cv is not None:
        return cv * CV_TO_CDA
    cda = cls.param(comp, "cda") if any(p.name == "cda" for p in cls.params) else None
    if cda is not None:
        return cda
    d = cls.param(comp, "diameter")
    if d is None:
        raise ValueError(f"{comp.id} needs cv, cda or diameter")
    return cls.param(comp, "cd") * math.pi * d * d / 4.0


AREA_PARAMS = (
    Param("cv", None, None, "flow coefficient (US gpm/sqrt(psi)); or give cda or diameter"),
    Param("cda", "area", None, "effective flow area Cd*A"),
    Param("diameter", "length", None, "flow diameter (used with cd)"),
    Param("cd", None, 0.7, "discharge coefficient"),
)

HEAT_PARAMS = (
    Param("heat_transfer", None, 0.0, "fluid-to-ambient conductance h*A in W/K (0 = adiabatic)"),
    Param("ambient_temperature", "temperature", "293.15 K", "ambient temperature for heat transfer"),
)


# ---------------------------------------------------------------- volumes


class Tank(ComponentType):
    type_name = "tank"
    label = "Tank"
    category = "Volumes"
    ports = ("top", "bottom", "port")
    params = (
        Param("fluid", "text", None, "fluid in the tank"),
        Param("volume", "volume", None, "internal volume"),
        Param("pressure", "pressure", None, "initial pressure (gas, or saturated if fill is given)"),
        Param("temperature", "temperature", None, "initial temperature"),
        Param("fill", None, None, "liquid volume fraction 0-1 for a saturated liquid/vapor fill"),
    ) + HEAT_PARAMS
    help = ("Single-fluid tank. With 'fill' it holds saturated liquid and vapor (LOX, LN2, nitrous); "
            "'top' draws vapor, 'bottom' draws liquid, 'port' draws the mixture.")

    @classmethod
    def build(cls, b, comp):
        fluid = cls.require(comp, "fluid")
        vol = cls.require(comp, "volume")
        fill = cls.param(comp, "fill")
        p = cls.param(comp, "pressure")
        t = cls.param(comp, "temperature")
        if fill is not None:
            init = ("saturated", t, p, float(fill))
        else:
            if p is None or t is None:
                raise ValueError(f"{comp.id} needs pressure and temperature (or temperature and fill)")
            init = ("pt", p, t)
        n = b.new_node(comp, "", fluid=fluid, volume=vol, init=init,
                       heat=(cls.param(comp, "heat_transfer"), cls.param(comp, "ambient_temperature")))
        b.bind_port(comp, "top", n, DRAW_VAPOR)
        b.bind_port(comp, "bottom", n, DRAW_LIQUID)
        b.bind_port(comp, "port", n, DRAW_MIX)


class PressurizedTank(ComponentType):
    type_name = "pressurized_tank"
    label = "Pressurized tank"
    category = "Volumes"
    ports = ("gas", "liquid")
    params = (
        Param("liquid", "text", None, "propellant or liquid fluid"),
        Param("pressurant", "text", "nitrogen", "ullage gas fluid"),
        Param("volume", "volume", None, "internal volume"),
        Param("fill", None, 0.8, "initial liquid volume fraction 0-1"),
        Param("pressure", "pressure", None, "initial ullage pressure"),
        Param("liquid_temperature", "temperature", "293.15 K", "initial liquid temperature"),
        Param("gas_temperature", "temperature", None, "initial ullage temperature (default: system initial temperature)"),
        Param("diameter", "length", None, "inner diameter, for liquid head at the outlet (optional)"),
    ) + HEAT_PARAMS
    help = ("Liquid with a separate pressurant gas above it at one shared pressure. "
            "When the liquid runs out, the liquid port starts passing ullage gas.")

    @classmethod
    def build(cls, b, comp):
        vol = cls.require(comp, "volume")
        fill = float(cls.param(comp, "fill"))
        if not 0.0 <= fill < 1.0:
            raise ValueError(f"{comp.id}.fill must be between 0 and 1")
        t_l = cls.param(comp, "liquid_temperature")
        t_g = cls.param(comp, "gas_temperature") or b.t_init
        p = cls.require(comp, "pressure")
        d = cls.param(comp, "diameter")
        area = math.pi * d * d / 4.0 if d else 0.0
        b.new_pressurized_tank(comp, cls.require(comp, "liquid"), cls.param(comp, "pressurant"), vol, fill,
                               p, t_l, t_g, area,
                               heat=(cls.param(comp, "heat_transfer"), cls.param(comp, "ambient_temperature")))


class Boundary(ComponentType):
    type_name = "boundary"
    label = "Pressure boundary"
    category = "Volumes"
    ports = ("port",)
    params = (
        Param("fluid", "text", None, "fluid supplied or vented"),
        Param("pressure", "pressure", "1 atm", "fixed pressure"),
        Param("temperature", "temperature", "293.15 K", "fixed temperature"),
    )
    help = "Fixed pressure and temperature: a regulated supply, the atmosphere, or a vent."

    @classmethod
    def build(cls, b, comp):
        n = b.new_boundary(comp, cls.require(comp, "fluid"), cls.param(comp, "pressure"),
                           cls.param(comp, "temperature"))
        b.bind_port(comp, "port", n, DRAW_MIX)


class Volume(ComponentType):
    type_name = "volume"
    label = "Manifold / volume"
    category = "Volumes"
    ports = ("a", "b", "c", "d")
    params = (
        Param("volume", "volume", None, "internal volume"),
        Param("fluid", "text", None, "fluid (default: taken from connected lines)"),
        Param("pressure", "pressure", None, "initial pressure (default: system initial pressure)"),
        Param("temperature", "temperature", None, "initial temperature"),
    ) + HEAT_PARAMS
    help = "A lumped volume with up to four ports, such as a manifold or a chamber."

    @classmethod
    def build(cls, b, comp):
        n = b.new_node(comp, "", fluid=cls.param(comp, "fluid"), volume=cls.require(comp, "volume"),
                       init=("pt", cls.param(comp, "pressure"), cls.param(comp, "temperature")),
                       heat=(cls.param(comp, "heat_transfer"), cls.param(comp, "ambient_temperature")))
        for port in cls.ports:
            b.bind_port(comp, port, n, DRAW_MIX)


# ---------------------------------------------------------------- lines


class Pipe(ComponentType):
    type_name = "pipe"
    label = "Tubing"
    category = "Lines"
    ports = ("in", "out")
    params = (
        Param("length", "length", None, "length"),
        Param("diameter", "length", None, "inner diameter"),
        Param("roughness", "length", "1.5 um", "wall roughness"),
        Param("segments", None, "auto", "number of lumped segments (auto: about one per metre, 1-10)"),
        Param("minor_loss", None, 0.0, "sum of loss coefficients K of bends and fittings along the line"),
        Param("fluid", "text", None, "fluid (default: taken from connected components)"),
        Param("pressure", "pressure", None, "initial pressure (default: system initial pressure)"),
        Param("temperature", "temperature", None, "initial temperature"),
        Param("wall_thickness", "length", None, "wall thickness; enables a wall temperature for chilldown"),
        Param("wall_material", "choice", "steel", "wall material", tuple(MATERIALS)),
        Param("film_coefficient", None, 500.0, "fluid-to-wall heat transfer coefficient, W/m2K"),
        Param("outer_coefficient", None, 5.0, "wall-to-ambient heat transfer coefficient, W/m2K"),
        Param("ambient_temperature", "temperature", "293.15 K", "ambient temperature"),
    )
    help = "Line with friction (Churchill), flow inertia and volume, split into lumped segments."

    @classmethod
    def build(cls, b, comp):
        length = cls.require(comp, "length")
        d = cls.require(comp, "diameter")
        seg = cls.param(comp, "segments")
        n = max(1, min(10, math.ceil(length / 1.0))) if seg in (None, "auto") else max(1, int(seg))
        area = math.pi * d * d / 4.0
        k_minor = float(cls.param(comp, "minor_loss"))
        rough = cls.param(comp, "roughness")
        wall = None
        thick = cls.param(comp, "wall_thickness")
        if thick:
            mat = MATERIALS[cls.param(comp, "wall_material")]
            seg_len = length / n
            d_out = d + 2 * thick
            m_wall = mat["density"] * math.pi / 4 * (d_out**2 - d**2) * seg_len
            wall = (m_wall * mat["cp"],
                    float(cls.param(comp, "film_coefficient")) * math.pi * d * seg_len,
                    float(cls.param(comp, "outer_coefficient")) * math.pi * d_out * seg_len,
                    cls.param(comp, "ambient_temperature"))
        init = ("pt", cls.param(comp, "pressure"), cls.param(comp, "temperature"))
        fluid = cls.param(comp, "fluid")
        # staggered lumping: n flow links of length L/n, n-1 interior volumes of A*L/n,
        # and half a segment of volume added to the connection point at each end
        seg_vol = area * length / n
        b.add_port_volume(comp, "in", seg_vol / 2)
        b.add_port_volume(comp, "out", seg_vol / 2)
        nodes = [b.new_node(comp, f"s{i + 1}", fluid=fluid, volume=seg_vol, init=init, wall=wall)
                 for i in range(n - 1)]
        ends = [b.port_node(comp, "in")] + nodes + [b.port_node(comp, "out")]
        b.set_port_init(comp, "in", init, fluid)
        b.set_port_init(comp, "out", init, fluid)
        for i in range(n):
            b.add_pipe(comp, f"f{i + 1}", ends[i], ends[i + 1], area=area, diameter=d, length=length / n,
                       roughness=rough, k_minor=k_minor / n, draw_a=b.port_draw(comp, "in") if i == 0 else 0,
                       draw_b=b.port_draw(comp, "out") if i == n - 1 else 0)


class Tee(ComponentType):
    type_name = "tee"
    label = "Tee / cross"
    category = "Lines"
    ports = ("a", "b", "c", "d")
    merges_ports = True
    params = (Param("volume", "volume", "1 cc", "internal volume"),)
    help = "Joins up to four lines at one point."

    @classmethod
    def build(cls, b, comp):
        b.add_port_volume(comp, "a", cls.param(comp, "volume"))


class Fitting(ComponentType):
    type_name = "fitting"
    label = "Fitting"
    category = "Lines"
    ports = ("in", "out")
    params = (
        Param("kind", "choice", "elbow90", "fitting type (sets the loss coefficient)", tuple(FITTING_K) + ("custom",)),
        Param("k", None, None, "loss coefficient K (overrides kind)"),
        Param("diameter", "length", None, "flow diameter"),
    )
    help = "A local pressure loss K*rho*v^2/2 (elbow, union, reducer, quick disconnect)."

    @classmethod
    def build(cls, b, comp):
        d = cls.require(comp, "diameter")
        k = cls.param(comp, "k")
        if k is None:
            kind = cls.param(comp, "kind")
            if kind not in FITTING_K:
                raise ValueError(f"{comp.id}: give k for a custom fitting")
            k = FITTING_K[kind]
        cda = math.pi * d * d / 4.0 / math.sqrt(max(float(k), 1e-3))
        b.add_orifice(comp, "", "in", "out", cda=cda)


# ---------------------------------------------------------------- flow control


class Orifice(ComponentType):
    type_name = "orifice"
    label = "Orifice / injector"
    category = "Flow control"
    ports = ("in", "out")
    params = AREA_PARAMS
    help = "Fixed restriction. Gas flow chokes; liquid flow uses Cd*A*sqrt(2*rho*dP) with flashing."

    @classmethod
    def build(cls, b, comp):
        b.add_orifice(comp, "", "in", "out", cda=_flow_area(cls, comp))


class Valve(ComponentType):
    type_name = "valve"
    label = "Valve (on/off)"
    category = "Flow control"
    ports = ("in", "out")
    is_valve = True
    params = AREA_PARAMS + (
        Param("open_time", "time", "50 ms", "time to travel fully open"),
        Param("close_time", "time", None, "time to travel fully closed (default: open_time)"),
        Param("normally", "choice", "closed", "state at the start of a run", ("closed", "open")),
    )
    help = "Solenoid or actuated ball valve. Its flow area ramps over the open and close times."

    @classmethod
    def build(cls, b, comp):
        t_open = max(cls.param(comp, "open_time"), 1e-6)
        t_close = max(cls.param(comp, "close_time") or t_open, 1e-6)
        start = 1.0 if cls.param(comp, "normally") == "open" else 0.0
        b.add_orifice(comp, "", "in", "out", cda=_flow_area(cls, comp), ctrl=CTRL_VALVE,
                      valve=(start, 1.0 / t_open, 1.0 / t_close))


class ThrottleValve(Valve):
    type_name = "throttle_valve"
    label = "Throttle / needle valve"
    params = AREA_PARAMS + (
        Param("open_time", "time", "2 s", "time to stroke from closed to fully open"),
        Param("close_time", "time", None, "time to stroke fully closed (default: open_time)"),
        Param("position", None, 0.0, "initial position 0-1"),
    )
    help = "Valve commanded to any position from 0 (closed) to 1 (open)."

    @classmethod
    def build(cls, b, comp):
        t_open = max(cls.param(comp, "open_time"), 1e-6)
        t_close = max(cls.param(comp, "close_time") or t_open, 1e-6)
        start = min(max(float(cls.param(comp, "position")), 0.0), 1.0)
        b.add_orifice(comp, "", "in", "out", cda=_flow_area(cls, comp), ctrl=CTRL_VALVE,
                      valve=(start, 1.0 / t_open, 1.0 / t_close))


class CheckValve(ComponentType):
    type_name = "check_valve"
    label = "Check valve"
    category = "Flow control"
    ports = ("in", "out")
    params = AREA_PARAMS + (
        Param("cracking_pressure", "pressure", "0.5 psi", "forward pressure difference where it starts to open"),
        Param("full_open_pressure", "pressure", "5 psi", "extra pressure difference to open fully"),
    )
    help = "Opens with forward pressure difference, never passes reverse flow."

    @classmethod
    def build(cls, b, comp):
        b.add_orifice(comp, "", "in", "out", cda=_flow_area(cls, comp), ctrl=CTRL_CHECK,
                      p1=cls.param(comp, "cracking_pressure"),
                      p2=max(cls.param(comp, "full_open_pressure"), 100.0))


class ReliefValve(ComponentType):
    type_name = "relief_valve"
    label = "Relief valve"
    category = "Flow control"
    ports = ("in", "out")
    params = AREA_PARAMS + (
        Param("set_pressure", "pressure", None, "inlet-minus-outlet pressure where it starts to open"),
        Param("overpressure", "pressure", "10 psi", "extra pressure to open fully"),
    )
    help = "Opens when the inlet exceeds the outlet by the set pressure."

    @classmethod
    def build(cls, b, comp):
        b.add_orifice(comp, "", "in", "out", cda=_flow_area(cls, comp), ctrl=CTRL_RELIEF,
                      p1=cls.require(comp, "set_pressure"), p2=max(cls.param(comp, "overpressure"), 100.0))


class Regulator(ComponentType):
    type_name = "regulator"
    label = "Regulator"
    category = "Flow control"
    ports = ("in", "out")
    params = AREA_PARAMS + (
        Param("setpoint", "pressure", None, "outlet pressure (absolute) where it closes"),
        Param("droop", "pressure", "20 psi", "outlet pressure drop below setpoint that opens it fully"),
    )
    help = "Pressure-reducing regulator: throttles to hold its outlet near the setpoint."

    @classmethod
    def build(cls, b, comp):
        b.add_orifice(comp, "", "in", "out", cda=_flow_area(cls, comp), ctrl=CTRL_REGULATOR,
                      p1=cls.require(comp, "setpoint"), p2=max(cls.param(comp, "droop"), 100.0))


TYPES: dict[str, type[ComponentType]] = {
    c.type_name: c
    for c in (Tank, PressurizedTank, Boundary, Volume, Pipe, Tee, Fitting, Orifice, Valve, ThrottleValve,
              CheckValve, ReliefValve, Regulator)
}
ALIASES = {"source": "boundary", "vent": "boundary", "ambient": "boundary", "injector": "orifice",
           "solenoid": "valve", "ball_valve": "valve", "needle_valve": "throttle_valve", "tubing": "pipe",
           "line": "pipe", "manifold": "volume", "cross": "tee", "junction": "tee"}


def get_type(name: str) -> type[ComponentType]:
    key = ALIASES.get(name, name)
    if key not in TYPES:
        raise ValueError(f"unknown component type {name!r}; known: {', '.join(sorted(TYPES))}")
    return TYPES[key]
