"""System description: components, connections, sensors and settings.

Systems are stored as YAML so they can be written by hand, diffed and kept in git:

    name: GN2 blowdown
    settings: {dt: 1 ms}
    components:
      - {id: TK-1, type: tank, fluid: nitrogen, volume: 50 L, pressure: 3000 psi, temperature: 20 C}
      - {id: V-1, type: valve, cv: 0.5}
      - {id: AMB, type: boundary, fluid: nitrogen}
    connections:
      - [TK-1.port, V-1.in]
      - [V-1.out, AMB.port]
    sensors:
      - {id: PT-1, type: pressure, at: TK-1.port}
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
    type: str  # pressure, temperature, mass_flow, valve_position, quality, density, mass, level
    at: str  # "COMP.port" for node quantities, "COMP" for links and tanks
    lag: float | None = None  # s, first-order time constant; None = the realistic default for the type
    noise: float | None = None  # standard deviation in SI units; None = realistic model, 0 = none
    rate: float = 0.0  # Hz, 0 = every step
    units: str = ""  # display units, e.g. "psi"

    def to_dict(self) -> dict:
        d = {"id": self.id, "type": self.type, "at": self.at}
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
    notes: list[str] = field(default_factory=list)  # changes made on load, shown as build warnings

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
        c = Component(cid, ctype, params, tuple(pos) if pos else None)
        self.components.append(c)
        return c

    def connect(self, a: str, b: str) -> None:
        self.connections.append((a, b))

    def sense(self, sid: str, stype: str, at: str, **kw) -> Sensor:
        s = Sensor(sid, stype, at, **kw)
        self.sensors.append(s)
        return s

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
            sensors.append(Sensor(str(s.pop("id")), s.pop("type"), str(s.pop("at")),
                                  _optional_float(s.pop("lag", None)), _optional_float(s.pop("noise", None)),
                                  float(s.pop("rate", 0) or 0), str(s.pop("units", "") or "")))
        system = cls(str(d.get("name", "untitled")), comps, conns, sensors, settings)
        system._fold_pipes()
        return system

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
