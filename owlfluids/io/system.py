"""System description: components, connections, sensors and settings.

Systems are stored as YAML so they can be written by hand, diffed and kept in git:

    name: GN2 blowdown
    settings: {dt: 1 ms}
    components:
      - {id: TK-1, type: tank, fluid: nitrogen, volume: 50 L, pressure: 3000 psi, temperature: 20 C}
      - {id: V-1, type: valve, cv: 0.5}
      - {id: AMB, type: boundary, fluid: nitrogen}
    connections:
      - [TK-1.top, V-1.in]
      - [V-1.out, AMB.port]
    sensors:
      - {id: PT-1, type: pressure, line: [V-1.out, AMB.port], along: 0.5}
      - {id: LL-1, type: level, at: TK-1}

A pressure, temperature, flow, density, quality or purge sensor sits on a line (``line``, the
two ports it joins, and ``along``, how far from the first one, 0-1) and reads what is there;
see core/lines.py. Level, mass, valve position and wall temperature sensors belong to a part
(``at``). A line sensor can also read a tank's or volume's port with nothing wired to it
(``at: TK-1.top``), and one that is on no line keeps its place on the schematic (``pos``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_SETTINGS = {
    "dt": "1 ms",
    "initial_pressure": "1 atm",
    "initial_temperature": "293.15 K",
    "junction_volume": "1 cc",
    "prime_lines": True,
    "backend": "HEOS",
    "fluid_models": {},
    "sensor_noise": "realistic",  # realistic: noise and lag by sensor type and state; off: exact values
    "noise_seed": 0,  # the same seed gives the same noise run to run
}

# Drawing-only line types from before lines were removed. A file that still has them
# loads with each one folded into a direct connection between the ports it joined.
PIPE_TYPES = ("pipe", "tubing", "line")
# Throttle valves from before manual valves replaced them; they load as manual valves.
THROTTLE_TYPES = ("throttle_valve", "needle_valve")
# Pressurized tanks from before they became tanks with a pressurant, and their old port names.
PRESSURIZED_TANK = "pressurized_tank"
TANK_PORTS = {"gas": "top", "liquid": "bottom"}
# Tanks had a third, side port ("port") before they had only a top and a bottom.
OLD_TANK_PORT = "port"


def tank_side_port(params: dict) -> str:
    """Where an old tank's side port goes: the bottom if the tank holds liquid there, else the top.

    A tank with a pressurant drew its ullage gas from the side port, so that is the top. A saturated
    tank (with a fill) drew its bulk mixture, which is mostly liquid by mass, so that is the bottom;
    so is a tank of a fluid that is only ever liquid. A gas bottle's two ports are alike: the top.
    """
    from ..fluids.registry import is_liquid_name

    if params.get("pressurant"):
        return "top"
    if params.get("fill") is not None or is_liquid_name(params.get("fluid")):
        return "bottom"
    return "top"


def _tank_params(params: dict) -> dict:
    """An old pressurized tank's parameters as a tank's: the liquid is its fluid."""
    out = {}
    for k, v in params.items():
        out[{"liquid": "fluid", "liquid_temperature": "temperature"}.get(k, k)] = v
    out.setdefault("pressurant", "nitrogen")  # the old defaults
    out.setdefault("fill", 0.8)
    return out


@dataclass
class Component:
    id: str
    type: str
    params: dict = field(default_factory=dict)
    position: tuple[float, float] | None = None  # schematic position, ignored by the solver
    rotation: int = 0

    def to_dict(self) -> dict:
        d = {"id": self.id, "type": self.type, **self.params}
        if self.position is not None:
            d["at"] = [round(self.position[0], 1), round(self.position[1], 1)]
        if self.rotation:
            d["rotation"] = self.rotation
        return d


def _optional_float(v) -> float | None:
    """A sensor's lag or noise: omitted or "auto" means the realistic default, "off" means 0."""
    if v is None or (isinstance(v, str) and v.strip().lower() == "auto"):
        return None
    if isinstance(v, str) and v.strip().lower() in ("off", "none"):
        return 0.0
    return float(v)


@dataclass
class Sensor:
    id: str
    type: str  # pressure, temperature, mass_flow, valve_position, quality, density, mass, level...
    # what it reads without a line: "COMP" for a part's sensors (level, mass, valve position), or
    # "COMP.port" for a port with no line (and in scripts and files from before sensors had lines)
    at: str = ""
    lag: float | None = None  # s, first-order time constant; None = the realistic default for the type
    noise: float | None = None  # standard deviation in SI units; None = realistic model, 0 = none
    rate: float = 0.0  # Hz, 0 = every step
    units: str = ""  # display units, e.g. "psi"
    # on a line: the two ports it joins (a flow meter reads positive from the first toward the
    # second), and where along it the sensor sits, 0-1 from the first (None: spread out evenly)
    line: tuple[str, str] | None = None
    along: float | None = None
    pos: tuple[float, float] | None = None  # schematic position of a sensor on no line

    def to_dict(self) -> dict:
        d = {"id": self.id, "type": self.type}
        if self.line:
            d["line"] = list(self.line)
            if self.along is not None:
                d["along"] = round(float(self.along), 3)
        elif self.at:
            d["at"] = self.at
        elif self.pos is not None:
            d["pos"] = [round(self.pos[0], 1), round(self.pos[1], 1)]
        for k in ("lag", "noise"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        if self.rate:
            d["rate"] = self.rate
        if self.units:
            d["units"] = self.units
        return d


@dataclass
class System:
    name: str = "untitled"
    components: list[Component] = field(default_factory=list)
    connections: list[tuple[str, str]] = field(default_factory=list)
    sensors: list[Sensor] = field(default_factory=list)
    settings: dict = field(default_factory=lambda: dict(DEFAULT_SETTINGS))
    # wires shaped by hand on the schematic, ignored by the solver: "a -- b" ->
    # {"h": leaves a horizontally, "bends": [[x, y], ...]}
    routes: dict = field(default_factory=dict)
    # Synnax test settings that belong to this system (mode, channel names...); the
    # connection itself (host, login) is the user's, see drivers/synnax.py
    synnax: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)  # changes made on load, shown as build warnings
    _old_tanks: set = field(default_factory=set, repr=False, compare=False)  # added as pressurized tanks

    # ---- editing helpers
    def component(self, cid: str) -> Component:
        for c in self.components:
            if c.id == cid:
                return c
        raise KeyError(f"no component {cid!r}")

    def add(self, cid: str, ctype: str, **params) -> Component:
        if any(c.id == cid for c in self.components):
            raise ValueError(f"component id {cid!r} is already used")
        pos = params.pop("at", None)
        if ctype == PRESSURIZED_TANK:  # scripts written before tanks took a pressurant
            ctype, params = "tank", _tank_params(params)
            self._old_tanks.add(cid)
        c = Component(cid, ctype, params, tuple(pos) if pos else None)
        self.components.append(c)
        return c

    def renamed(self, old: str, new: str) -> None:
        """A part or sensor was renamed: its Synnax channels keep their names under the new id."""
        sx = self.synnax
        for key in ("sensors", "valves"):
            m = sx.get(key)
            if isinstance(m, dict) and old in m:
                sx[key] = {new if k == old else k: v for k, v in m.items()}
        if old in (sx.get("exclude") or []):
            sx["exclude"] = [new if x == old else x for x in sx["exclude"]]

    def _pressurized_port(self, ref: str) -> str:
        """An old pressurized tank's port (TK.gas, TK.liquid) by its tank name (TK.top, TK.bottom)."""
        cid, dot, port = ref.rpartition(".")
        return f"{cid}.{TANK_PORTS[port]}" if cid in self._old_tanks and port in TANK_PORTS else ref

    def _side_port(self, ref: str) -> str:
        """An old tank side port (TK.port) by the port it is now: TK.top or TK.bottom."""
        cid, dot, port = ref.rpartition(".")
        if dot and port == OLD_TANK_PORT:
            c = next((c for c in self.components if c.id == cid), None)
            if c is not None and c.type == "tank":
                return f"{cid}.{tank_side_port(c.params)}"
        return ref

    def current_port(self, ref: str) -> str:
        """An old tank port by its name now (scripts written before tanks changed)."""
        return self._side_port(self._pressurized_port(ref))

    def connect(self, a: str, b: str) -> None:
        self.connections.append((self.current_port(a), self.current_port(b)))

    def sense(self, sid: str, stype: str, at: str = "", line: tuple[str, str] | None = None, **kw) -> Sensor:
        """A sensor on ``line`` (the two ports it joins), or on part ``at`` (a level, mass or
        valve position sensor; or, in older scripts, the line at port ``at``)."""
        if line is not None:
            line = (self.current_port(line[0]), self.current_port(line[1]))
        s = Sensor(sid, stype, self.current_port(at) if at else "", line=line, **kw)
        self.sensors.append(s)
        return s

    def line_of(self, s: Sensor) -> tuple[str, str] | None:
        """The connection a sensor sits on, in the system's own order, or None."""
        if not s.line:
            return None
        a, b = s.line
        return next(((x, y) for x, y in self.connections if (x, y) in ((a, b), (b, a))), None)

    def setting(self, key: str):
        return self.settings.get(key, DEFAULT_SETTINGS[key])

    # ---- files
    @classmethod
    def from_dict(cls, d: dict) -> System:
        settings = dict(DEFAULT_SETTINGS)
        settings.update(d.get("settings") or {})
        comps = []
        for raw in d.get("components") or []:
            raw = dict(raw)
            cid = str(raw.pop("id"))
            ctype = raw.pop("type")
            pos = raw.pop("at", None)
            rot = int(raw.pop("rotation", 0))
            comps.append(Component(cid, ctype, raw, tuple(pos) if pos else None, rot))
        conns = []
        for c in d.get("connections") or []:
            if isinstance(c, str):
                a, b = (s.strip() for s in c.split("--"))
            else:
                a, b = c
            conns.append((str(a), str(b)))
        sensors = []
        for s in d.get("sensors") or []:
            s = dict(s)
            line = s.pop("line", None)
            if isinstance(line, str):
                line = [x.strip() for x in line.split("--")]
            along, pos = s.pop("along", None), s.pop("pos", None)
            sensors.append(Sensor(str(s.pop("id")), s.pop("type"), str(s.pop("at", "") or ""),
                                  _optional_float(s.pop("lag", None)), _optional_float(s.pop("noise", None)),
                                  float(s.pop("rate", 0) or 0), str(s.pop("units", "") or ""),
                                  (str(line[0]), str(line[1])) if line else None,
                                  None if along is None else min(max(float(along), 0.0), 1.0),
                                  (float(pos[0]), float(pos[1])) if pos else None))
        routes = {str(k): dict(v) for k, v in (d.get("routes") or {}).items()}
        system = cls(str(d.get("name", "untitled")), comps, conns, sensors, settings, routes,
                     dict(d.get("synnax") or {}))
        system._fold_pipes()
        system._throttles_to_manual()
        system._merge_pressurized_tanks()
        system._tank_side_ports()
        moved = system.sensors_onto_lines()
        if moved:
            system.notes.append("sensors sit on lines now, not on a part's port: " + ", ".join(moved)
                                + " moved onto the line they were drawn on (save the file to keep it that way)")
        return system

    def sensors_onto_lines(self) -> list[str]:
        """Put sensors that read the line at a part's port (files and scripts from before sensors
        were parts of their own) on that line, reading the same. Returns the ids moved.

        A flow meter at a flow element's port, or on the element itself, goes on the line at its
        outlet (or inlet), reading positive the way the element passes fluid; at a tank's or
        volume's port, on the line out of it. A port with two or more lines straight off it keeps
        its sensor where it is: no one line carries all its flow.
        """
        from ..components import get_type
        from ..core.lines import LINE_TYPES

        comps = {c.id: c for c in self.components}
        moved = []
        for s in self.sensors:
            if s.line or not s.at or s.type not in LINE_TYPES:
                continue
            ref = s.at
            cid, dot, port = ref.rpartition(".")
            if not dot:  # a part: a flow meter on a flow element reads the line at its outlet, else its inlet
                c = comps.get(ref)
                if s.type != "mass_flow" or c is None:
                    continue
                ct = get_type(c.type)
                ports = ct.ports
                if len(ports) != 2 or ct.category == "Volumes" or ct.merges_ports:
                    continue
                ref = next((f"{ref}.{p}" for p in (ports[-1], ports[0])
                            if sum(f"{ref}.{p}" in ab for ab in self.connections) == 1), None)
                if ref is None:
                    continue
                cid, port = ref.rpartition(".")[0], ref.rpartition(".")[2]
            on = [ab for ab in self.connections if ref in ab]
            if not on or (s.type == "mass_flow" and len(on) > 1) or cid not in comps:
                continue
            a, b = on[0]
            other = b if a == ref else a
            ct = get_type(comps[cid].type)
            if s.type == "mass_flow" and ct.merges_ports:  # never read: a tee's ports are all one point
                continue
            if s.type == "mass_flow":
                # positive the way the element passes fluid (in to out), or out of a tank or volume
                inlet = ct.category != "Volumes" and not ct.merges_ports and port == ct.ports[0] and len(ct.ports) > 1
                s.line = (other, ref) if inlet else (ref, other)
            else:
                s.line = (a, b)
            s.at = ""
            moved.append(s.id)
        return moved

    def _fix_refs(self, fix) -> None:
        """Rename port references everywhere they appear: connections, sensors and hand-shaped wires."""
        self.connections = [(fix(a), fix(b)) for a, b in self.connections]
        for s in self.sensors:
            s.at = fix(s.at) if s.at else s.at
            if s.line:
                s.line = (fix(s.line[0]), fix(s.line[1]))
        routes = {}
        for k, v in self.routes.items():
            a, _, b = k.partition(" -- ")
            routes[f"{fix(a)} -- {fix(b)}" if b else k] = v
        self.routes = routes

    def _tank_side_ports(self) -> None:
        """Move what was on an old tank's side port to its top or bottom (see ``tank_side_port``)."""
        moved = {}
        tanks = {c.id: c for c in self.components if c.type == "tank"}
        refs = [r for ab in self.connections for r in ab] + [s.at for s in self.sensors] + [
            r for s in self.sensors for r in (s.line or ())]
        for ref in refs:
            cid, _, port = ref.rpartition(".")
            if port == OLD_TANK_PORT and cid in tanks:
                moved[cid] = tank_side_port(tanks[cid].params)
        if not moved:
            return
        self._fix_refs(self._side_port)
        # a wire can now join a tank's port to itself or repeat another; keep one of each
        kept = []
        for a, b in self.connections:
            if a != b and (a, b) not in kept and (b, a) not in kept:
                kept.append((a, b))
        self.connections = kept
        self.notes.append("tanks have only a top and a bottom port now: "
                          + ", ".join(f"{cid}.port is {cid}.{p}" for cid, p in sorted(moved.items()))
                          + " (save the file to keep it that way)")

    def _merge_pressurized_tanks(self) -> None:
        """Turn old pressurized tanks into tanks with a pressurant: gas port on top, liquid at the bottom."""
        for c in self.components:
            if c.type == PRESSURIZED_TANK:
                c.type, c.params = "tank", _tank_params(c.params)
                self._old_tanks.add(c.id)
        if not self._old_tanks:
            return
        self._fix_refs(self._pressurized_port)
        self.notes.append(f"pressurized tanks are now tanks with a pressurant: {', '.join(sorted(self._old_tanks))} "
                          "(their gas port is 'top' and their liquid port 'bottom'; save the file to keep it that way)")

    def _throttles_to_manual(self) -> None:
        """Turn old throttle valves into manual valves, keeping their 2 s default stroke."""
        moved = []
        for c in self.components:
            if c.type in THROTTLE_TYPES:
                c.type = "manual_valve"
                c.params.setdefault("open_time", "2 s")
                moved.append(c.id)
        if moved:
            self.notes.append(f"throttle valves are now manual valves, worked from the valve panel and never by "
                              f"Synnax: {', '.join(moved)} (save the file to keep it that way)")

    def _fold_pipes(self) -> None:
        """Replace each old drawing-only pipe with direct connections between the ports it joined."""
        pipes = {c.id for c in self.components if c.type in PIPE_TYPES}
        if not pipes:
            return
        parent: dict[str, str] = {}

        def find(r: str) -> str:
            parent.setdefault(r, r)
            while parent[r] != r:
                parent[r] = parent[parent[r]]
                r = parent[r]
            return r

        def comp_of(ref: str) -> str:
            return ref.rsplit(".", 1)[0] if "." in ref else ref

        for pid in pipes:  # a pipe's two ends are one point
            parent[find(f"{pid}.in")] = find(f"{pid}.out")
            parent[find(pid)] = find(f"{pid}.out")
        kept = []
        for a, b in self.connections:
            if comp_of(a) in pipes or comp_of(b) in pipes:
                parent[find(a)] = find(b)
            else:
                kept.append((a, b))
        # every point a pipe touched: the component ports there, first one seen first
        points: dict[str, list[str]] = {find(pid): [] for pid in pipes}
        for a, b in self.connections:
            for ref in (a, b):
                refs = points.get(find(ref))
                if refs is not None and comp_of(ref) not in pipes and ref not in refs:
                    refs.append(ref)
        for refs in points.values():
            kept.extend((refs[0], r) for r in refs[1:] if (refs[0], r) not in kept and (r, refs[0]) not in kept)
        self.connections = kept
        self.notes.append(f"lines are no longer components: {', '.join(sorted(pipes))} became direct connections "
                          "(save the file to keep it that way)")
        # a line's id stands for all its points (in, out, and the segments of older files)
        for s in self.sensors:
            line = comp_of(s.at)
            if line in pipes:
                refs = points.get(find(line))
                if s.type in ("mass_flow", "valve_position"):
                    self.notes.append(f"sensor {s.id} reads {s.type} on line {line}, which has no flow element; "
                                      "move it to a valve, orifice or fitting")
                elif refs:
                    s.at = refs[0]
                    self.notes.append(f"sensor {s.id} was on line {line}; it now reads {s.at}")
                else:
                    self.notes.append(f"sensor {s.id} is on line {line}, which joins nothing; move it to a port")
        dropped = sorted(c.id for c in self.components if c.id in pipes and c.params)
        if dropped:
            self.notes.append(f"the settings of {', '.join(dropped)} (length, diameter, wall) are dropped: lines are "
                              "not modelled, so draw a line as a volume to keep its volume or wall")
        self.components = [c for c in self.components if c.id not in pipes]

    def to_dict(self) -> dict:
        settings = {k: v for k, v in self.settings.items() if DEFAULT_SETTINGS.get(k) != v}
        d = {"name": self.name}
        if settings:
            d["settings"] = settings
        d["components"] = [c.to_dict() for c in self.components]
        d["connections"] = [list(c) for c in self.connections]
        if self.sensors:
            d["sensors"] = [s.to_dict() for s in self.sensors]
        if self.routes:
            d["routes"] = self.routes
        if self.synnax:
            d["synnax"] = self.synnax
        return d

    @classmethod
    def load(cls, path: str | Path) -> System:
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(self.to_dict(), fh, Dumper=_FlowDumper, sort_keys=False, width=4096, allow_unicode=True)


class _FlowDumper(yaml.SafeDumper):
    """Writes each component, connection and sensor on one line."""


def _repr_dict(dumper, data):
    flow = all(not isinstance(v, (dict, list)) or (isinstance(v, list) and len(v) <= 2) for v in data.values())
    return dumper.represent_mapping("tag:yaml.org,2002:map", data, flow_style=flow and len(data) > 0)


def _repr_list(dumper, data):
    flow = len(data) <= 2 and all(not isinstance(v, (dict, list)) for v in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_FlowDumper.add_representer(dict, _repr_dict)
_FlowDumper.add_representer(list, _repr_list)
