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
    "backend": "HEOS",
    "fluid_models": {},
}


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


@dataclass
class Sensor:
    id: str
    type: str  # pressure, temperature, mass_flow, valve_position, quality, density, mass
    at: str  # "COMP.port" for node quantities, "COMP" for links
    lag: float = 0.0  # s, first-order time constant
    noise: float = 0.0  # standard deviation in SI units
    rate: float = 0.0  # Hz, 0 = every step
    units: str = ""  # display units, e.g. "psi"

    def to_dict(self) -> dict:
        d = {"id": self.id, "type": self.type, "at": self.at}
        for k in ("lag", "noise", "rate"):
            if getattr(self, k):
                d[k] = getattr(self, k)
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
                                  float(s.pop("lag", 0) or 0), float(s.pop("noise", 0) or 0),
                                  float(s.pop("rate", 0) or 0), str(s.pop("units", "") or "")))
        return cls(str(d.get("name", "untitled")), comps, conns, sensors, settings)

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
            yaml.dump(self.to_dict(), fh, Dumper=_FlowDumper, sort_keys=False, width=120, allow_unicode=True)


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
