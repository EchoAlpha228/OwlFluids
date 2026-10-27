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
    CTRL_CHECK,
    CTRL_REGULATOR,
    CTRL_RELIEF,
    CTRL_VALVE,
    DRAW_LIQUID,
    DRAW_MIX,
    DRAW_VAPOR,
)
from ..io.system import PIPE_TYPES

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
        if spec.dimension is None and isinstance(raw, str):
            try:  # plain numbers written as text, e.g. YAML reads "5e-2" as a string
                return float(raw)
            except ValueError:
                return raw  # a keyword rather than a number
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

WALL_PARAMS = (
    Param("wall_mass", "mass", None,
          "mass of the shell; enables a wall that stores heat (a nitrous bottle stays warmer as it "
          "empties, a cold line or manifold chills down)"),
    Param("wall_material", "choice", "steel", "shell material", tuple(MATERIALS)),
    Param("film_coefficient", None, 1000.0,
          "fluid-to-wall heat transfer coefficient, W/m2K (boiling nitrous: about 1000-3000; tune to test data)"),
    Param("outer_coefficient", None, 5.0,
          "wall-to-ambient heat transfer coefficient, W/m2K (heat_transfer, if set, adds to it)"),
)


def _wall(cls, comp, vol):
    """A shell that starts at the fluid's temperature, or None without a wall_mass."""
    m_wall = cls.param(comp, "wall_mass")
    if not m_wall:
        return None
    # shell area of a cylinder four diameters long
    d = (vol / math.pi) ** (1.0 / 3.0)
    area = 4.5 * math.pi * d * d
    mat = MATERIALS[cls.param(comp, "wall_material")]
    return (float(m_wall) * mat["cp"], float(cls.param(comp, "film_coefficient")) * area,
            float(cls.param(comp, "outer_coefficient")) * area + float(cls.param(comp, "heat_transfer") or 0.0),
            cls.param(comp, "ambient_temperature"),
            True)


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
    ) + WALL_PARAMS + HEAT_PARAMS
    help = ("Single-fluid tank. With 'fill' it holds saturated liquid and vapor (LOX, LN2, nitrous) in "
            "equilibrium, so its pressure follows its temperature as it fills, vents or boils off; "
            "'top' draws vapor, 'bottom' draws liquid, 'port' draws the mixture. A 'wall_mass' gives "
            "it a shell that feeds heat into the boiling fluid (a nitrous bottle stays warmer as it empties).")

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
        wall = _wall(cls, comp, vol)
        n = b.new_node(comp, "", fluid=fluid, volume=vol, init=init,
                       heat=(cls.param(comp, "heat_transfer"), cls.param(comp, "ambient_temperature")), wall=wall)
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
        Param("liquid_temperature", "temperature", None,
              "initial liquid temperature (default: system initial temperature, or just below the boiling "
              "point at the tank pressure if that is colder)"),
        Param("gas_temperature", "temperature", None,
              "initial ullage temperature (default: system initial temperature)"),
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
    ) + WALL_PARAMS + HEAT_PARAMS
    help = ("A lumped volume with up to four ports, such as a manifold or a chamber. A 'wall_mass' "
            "gives it a shell, for example to chill down a cryogenic line.")

    @classmethod
    def build(cls, b, comp):
        vol = cls.require(comp, "volume")
        n = b.new_node(comp, "", fluid=cls.param(comp, "fluid"), volume=vol,
                       init=("pt", cls.param(comp, "pressure"), cls.param(comp, "temperature")),
                       heat=(cls.param(comp, "heat_transfer"), cls.param(comp, "ambient_temperature")),
                       wall=_wall(cls, comp, vol))
        for port in cls.ports:
            b.bind_port(comp, port, n, DRAW_MIX)


class PneumaticSupply(Volume):
    type_name = "pneumatic_supply"
    label = "Pneumatic supply"
    params = (
        Param("volume", "volume", "2 L", "internal volume of the header, lines and any receiver"),
        Param("fluid", "text", None, "gas (default: taken from what feeds it, such as a compressor or bottle)"),
        Param("pressure", "pressure", None, "initial pressure (default: system initial pressure)"),
        Param("temperature", "temperature", None, "initial temperature"),
    ) + HEAT_PARAMS
    help = ("The air (or nitrogen) header that pneumatic valves run on, the 'A' on a P&ID. Feed it from a "
            "compressor, or a bottle through a regulator. Every pneumatic valve draws on it as its actuator "
            "fills, so a weak supply strokes valves slowly or not at all.")

    @classmethod
    def build(cls, b, comp):
        n = b.new_node(comp, "", fluid=cls.param(comp, "fluid"), volume=cls.require(comp, "volume"),
                       init=("pt", cls.param(comp, "pressure"), cls.param(comp, "temperature")),
                       heat=(cls.param(comp, "heat_transfer"), cls.param(comp, "ambient_temperature")))
        for port in cls.ports:
            b.bind_port(comp, port, n, DRAW_MIX)


# ---------------------------------------------------------------- lines


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
    label = "Orifice"
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
        Param("actuator", "choice", "pneumatic",
              "pneumatic (spring return, draws on an air supply) or solenoid", ("pneumatic", "solenoid")),
        Param("air_supply", "text", None, "pneumatic supply it runs on (default: the only one in the system)"),
        Param("actuator_volume", "volume", "100 cc", "air the actuator takes per stroke, at its own pressure"),
        Param("stroke_pressure", "pressure", "30 psig", "actuator pressure where the valve starts to move"),
        Param("full_stroke_pressure", "pressure", "60 psig", "actuator pressure where it has fully stroked"),
        Param("pilot_cv", None, 0.1, "Cv of the pilot solenoid that fills and vents the actuator"),
    )
    help = ("Pneumatic or solenoid on/off valve. Its flow area ramps over the open and close times. A pneumatic "
            "valve draws on its air supply while held away from its normal state (a normally closed valve while "
            "open, a normally open one while closed) and its spring returns it when the air is vented; with too "
            "little supply pressure it strokes partway or not at all. Without a pneumatic supply in the system "
            "it moves on command alone, like a solenoid valve.")

    @classmethod
    def build(cls, b, comp):
        t_open = max(cls.param(comp, "open_time"), 1e-6)
        t_close = max(cls.param(comp, "close_time") or t_open, 1e-6)
        normally_open = cls.param(comp, "normally") == "open"
        start = 1.0 if normally_open else 0.0
        b.add_orifice(comp, "", "in", "out", cda=_flow_area(cls, comp), ctrl=CTRL_VALVE,
                      valve=(start, 1.0 / t_open, 1.0 / t_close))
        actuator = comp.params.get("actuator", "solenoid" if comp.type == "solenoid" else "pneumatic")
        if actuator not in ("pneumatic", "solenoid"):
            raise ValueError(f"{comp.id}.actuator must be pneumatic or solenoid")
        if actuator == "solenoid":
            return
        supply = cls.param(comp, "air_supply")
        supplies = [c.id for c in b.system.components if get_type(c.type) is PneumaticSupply]
        if supply is None:
            if not supplies:
                return  # nothing to draw on: it moves on command alone
            if len(supplies) > 1:
                raise ValueError(f"set air_supply to one of {', '.join(supplies)}")
            supply = supplies[0]
        elif supply not in supplies:
            raise ValueError(f"air_supply {supply!r} is not a pneumatic supply"
                             + (f"; pneumatic supplies: {', '.join(supplies)}" if supplies else ""))
        p_start = cls.param(comp, "stroke_pressure") - units.ATM
        p_full = cls.param(comp, "full_stroke_pressure") - units.ATM
        if p_full <= p_start:
            raise ValueError("full_stroke_pressure must be above stroke_pressure")
        b.add_actuator(comp, b.comp_nodes[supply][0], cls.require(comp, "actuator_volume"), p_start, p_full,
                       float(cls.param(comp, "pilot_cv")) * CV_TO_CDA, normally_open)


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


class AirPump(ComponentType):
    type_name = "air_pump"
    label = "Air pump / compressor"
    category = "Flow control"
    ports = ("out",)
    params = (
        Param("flow", "mass_flow", None, "mass flow while running"),
        Param("start_pressure", "pressure", None, "outlet pressure at or below which it switches on"),
        Param("stop_pressure", "pressure", None, "outlet pressure at which it switches off"),
        Param("fluid", "text", "air", "what it pumps (drawn from the atmosphere)"),
        Param("temperature", "temperature", "293.15 K", "delivery temperature (after its aftercooler)"),
        Param("spin_up", "time", "0.5 s", "time to reach full flow, and to stop"),
        Param("volume", "volume", "50 cc", "internal volume at its outlet"),
    )
    help = ("Compressor with a pressure switch: it starts when its outlet falls to the start pressure "
            "and runs at its rated flow until the outlet reaches the stop pressure.")

    @classmethod
    def build(cls, b, comp):
        p_on = cls.require(comp, "start_pressure")
        p_off = cls.require(comp, "stop_pressure")
        if p_off <= p_on:
            raise ValueError(f"{comp.id}: stop_pressure must be above start_pressure")
        b.add_pump(comp, "out", cls.param(comp, "fluid"), cls.param(comp, "temperature"), cls.require(comp, "flow"),
                   p_on, p_off, cls.param(comp, "spin_up"), cls.param(comp, "volume"))


TYPES: dict[str, type[ComponentType]] = {
    c.type_name: c
    for c in (Tank, PressurizedTank, Boundary, Volume, PneumaticSupply, Tee, Fitting, Orifice, Valve, ThrottleValve,
              CheckValve, ReliefValve, Regulator, AirPump)
}
ALIASES = {"source": "boundary", "vent": "boundary", "ambient": "boundary", "injector": "orifice",
           "solenoid": "valve", "ball_valve": "valve", "needle_valve": "throttle_valve",
           "manifold": "volume", "cross": "tee", "junction": "tee", "compressor": "air_pump",
           "pump": "air_pump", "pneumatic": "pneumatic_supply", "air_supply": "pneumatic_supply"}


def get_type(name: str) -> type[ComponentType]:
    key = ALIASES.get(name, name)
    if key in PIPE_TYPES:
        raise ValueError(f"{name!r} is no longer a component: lines are not modelled, so connect the ports "
                         "directly (draw a line as a 'volume' to keep its volume or steel)")
    if key not in TYPES:
        raise ValueError(f"unknown component type {name!r}; known: {', '.join(sorted(TYPES))}")
    return TYPES[key]
