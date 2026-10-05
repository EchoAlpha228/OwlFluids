"""Synnax hardware-in-the-loop driver.

The simulation stands in for the test stand: every ``1/rate`` seconds the driver
writes the simulated sensors to Synnax channels (in each sensor's display units,
so they look like the real transducers), and it streams valve channels back from
Synnax and applies them to the simulated valves. Those can be the command
channels that Arc sequences and operators write, or the valve state channels a
real controller reports in HITL tests.

Everything that touches the ``synnax`` package lives in ``SynnaxLink``; when the
client API changes, that class is the only one to update. The driver itself talks
to any object with the same five methods, which is how it is tested without a
Synnax server.

Config (a ``synnax:`` section in the system file, or its own YAML file)::

    synnax:
      host: localhost
      port: 9090
      username: synnax
      password: seldon
      secure: false
      rate: 100                   # Hz written to Synnax
      index: owl_time             # index channel for everything written
      create_channels: true       # create any missing channel
      sensors:                    # sim sensor -> channel (default: lower-case id)
        PT-OX: pt_ox
      valves:                     # sim valve -> channels (default: <valve>_cmd / <valve>_state)
        V-OXM: {read: ox_mpv_cmd, write: ox_mpv_state}
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .. import units


def _channel_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_").replace(".", "_")


@dataclass
class SynnaxConfig:
    host: str = "localhost"
    port: int = 9090
    username: str = "synnax"
    password: str = "seldon"
    secure: bool = False
    rate: float = 100.0
    index: str = "owl_time"
    create_channels: bool = True
    sensors: dict[str, str] = field(default_factory=dict)
    valves: dict[str, dict[str, str]] = field(default_factory=dict)
    valve_threshold: float = 0.5  # a read value above this opens an on/off valve

    @classmethod
    def from_dict(cls, d: dict | None) -> SynnaxConfig:
        d = dict(d or {})
        known = {k: d.pop(k) for k in list(d) if k in cls.__dataclass_fields__}
        if d:
            raise ValueError(f"unknown synnax settings: {', '.join(d)}")
        return cls(**known)

    @classmethod
    def load(cls, path: str | Path) -> SynnaxConfig:
        with open(path, encoding="utf-8") as fh:
            d = yaml.safe_load(fh) or {}
        return cls.from_dict(d.get("synnax", d))

    def resolve(self, sim) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
        """Sensor -> channel and valve -> (read channel, write channel), filling in defaults."""
        sensors = dict(self.sensors) or {s.id: _channel_name(s.id) for s in sim.system.sensors}
        for sid in sensors:
            sim.read(sid)  # unknown sensor -> KeyError now, not mid-run
        valves = {}
        names = list(self.valves) or sim.valves()
        for v in names:
            sim.valve_position(v)
            spec = self.valves.get(v) or {}
            valves[v] = (spec.get("read", f"{_channel_name(v)}_cmd"), spec.get("write", f"{_channel_name(v)}_state"))
        return sensors, valves


class SynnaxLink:
    """The only code that uses the ``synnax`` client package."""

    def __init__(self, cfg: SynnaxConfig):
        try:
            import synnax as sy
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise RuntimeError("the Synnax driver needs the synnax package (Python 3.12+): "
                               "pip install owlfluids[synnax]") from e
        self.sy = sy
        self.client = sy.Synnax(host=cfg.host, port=cfg.port, username=cfg.username, password=cfg.password,
                                secure=cfg.secure)
        self.writer = None
        self.streamer = None

    def ensure_channels(self, index: str, data: list[str], commands: list[str], create: bool) -> None:
        sy = self.sy
        existing = {c.name for c in self.client.channels.retrieve([index, *data, *commands])}
        missing = [n for n in [index, *data, *commands] if n not in existing]
        if missing and not create:
            raise RuntimeError(f"missing Synnax channels: {', '.join(missing)}")
        if index in missing:
            self.client.channels.create(name=index, is_index=True, data_type=sy.DataType.TIMESTAMP)
        idx = self.client.channels.retrieve(index)
        for n in data:
            if n in missing:
                self.client.channels.create(name=n, index=idx.key, data_type=sy.DataType.FLOAT32)
        for n in commands:
            if n in missing:
                self.client.channels.create(name=n, virtual=True, data_type=sy.DataType.FLOAT32)

    def open(self, write: list[str], read: list[str]) -> None:
        sy = self.sy
        self.writer = self.client.open_writer(start=sy.TimeStamp.now(), channels=write, name="OwlFluids",
                                              enable_auto_commit=True)
        self.streamer = self.client.open_streamer(read) if read else None

    def write(self, values: dict[str, float], index: str) -> None:
        self.writer.write({index: self.sy.TimeStamp.now(), **values})

    def read(self, timeout: float) -> dict[str, float]:
        """Latest value of each streamed channel that changed, by channel name."""
        if self.streamer is None:
            time.sleep(timeout)
            return {}
        frame = self.streamer.read(timeout=timeout)
        if frame is None:
            return {}
        out = {}
        for ch, series in frame.items():
            if len(series):
                out[self._name(ch)] = float(series[-1])
        return out

    def _name(self, ch) -> str:
        if isinstance(ch, str):
            return ch
        return self.client.channels.retrieve(ch).name

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        if self.streamer is not None:
            self.streamer.close()


class SynnaxDriver:
    """Publishes sensors and applies valve channels for a running ``Runner``."""

    def __init__(self, runner, cfg: SynnaxConfig, link_factory: Callable[[SynnaxConfig], object] = SynnaxLink):
        self.runner = runner
        self.cfg = cfg
        sim = runner.sim
        self.sensors, self.valves = cfg.resolve(sim)
        self.units = {}
        for s in sim.system.sensors:
            self.units[s.id] = s.units or units.DEFAULT_UNITS.get(s.type, "")
        self.read_to_valve = {r: v for v, (r, _w) in self.valves.items()}
        self.link = link_factory(cfg)
        self.link.ensure_channels(cfg.index, list(self.sensors.values()) + [w for _r, w in self.valves.values()],
                                  list(self.read_to_valve), cfg.create_channels)
        self.link.open([cfg.index, *self.sensors.values(), *(w for _r, w in self.valves.values())],
                       list(self.read_to_valve))
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.writes = 0
        self.commands = 0
        self.error: BaseException | None = None

    def start(self) -> None:
        self._stop.clear()
        self._threads = [threading.Thread(target=self._guard(self._write_loop), name="synnax-write", daemon=True),
                         threading.Thread(target=self._guard(self._read_loop), name="synnax-read", daemon=True)]
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(2.0)
        self.link.close()

    def _guard(self, fn):
        def run():
            try:
                fn()
            except BaseException as e:  # reported to the caller / GUI
                self.error = e
        return run

    def values(self, frame) -> dict[str, float]:
        out = {ch: units.convert(frame.sensors[sid], self.units.get(sid, "")) for sid, ch in self.sensors.items()}
        for v, (_r, w) in self.valves.items():
            out[w] = frame.valves[v]
        return out

    def _write_loop(self) -> None:
        period = 1.0 / self.cfg.rate
        nxt = time.perf_counter()
        last_t = None
        while not self._stop.is_set():
            frame = self.runner.frame
            if frame.t != last_t:  # nothing new while paused
                self.link.write(self.values(frame), self.cfg.index)
                self.writes += 1
                last_t = frame.t
            nxt += period
            delay = nxt - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = time.perf_counter()

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            for ch, value in self.link.read(0.1).items():
                v = self.read_to_valve.get(ch)
                if v is None:
                    continue
                self.apply(v, value)

    def apply(self, valve: str, value: float) -> None:
        """A valve channel changed: on/off valves open above the threshold, throttles follow the value."""
        sim = self.runner.sim
        from ..components import get_type

        throttle = get_type(sim.system.component(valve).type).type_name == "throttle_valve"
        pos = min(max(value, 0.0), 1.0) if throttle else (1.0 if value > self.cfg.valve_threshold else 0.0)
        self.runner.command(valve, pos)
        self.commands += 1
