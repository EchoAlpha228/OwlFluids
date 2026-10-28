"""Synnax driver for hardware-out-of-the-loop (HOOTL) and hardware-in-the-loop (HITL) tests.

The simulation stands in for the test stand: every ``1/rate`` seconds the driver
writes the simulated sensors to Synnax channels (in each sensor's display units,
so they look like the real transducers), and it streams valve channels back from
Synnax and applies them to the simulated valves. ``mode`` decides which:

- ``hootl`` (default): no hardware. The driver reads each valve's command channel
  (what Arc sequences and operators write) and fakes the valve's state channel,
  so schematics and sequences that wait on valve states work as on the stand.
- ``hitl``: real valve hardware. The driver reads the valve state channels the
  hardware reports and never writes them; only sensors are faked.

``valve_state`` says what a state channel means on the stand:

- ``command`` (default; a Synnax digital output task's state channel): it echoes
  the command. HOOTL fakes it from the command (1/0);
  HITL commands the sim valve with it, and the sim models the stroke.
- ``position``: it reports where the valve actually is (limit switches or position
  feedback). HOOTL fakes it like limit switches, 1 once fully open and 0 once
  fully closed, holding its value mid-stroke; HITL
  puts the sim valve at the reported position at once.

Manual valves and regulators are the operator's: they are worked from the
OwlFluids window only, so the driver never gives them channels.

Sensor and state channels that already exist keep their own index channel (the
driver stamps each index it writes), so the real stand's channels can be used.

Everything that touches the ``synnax`` package lives in ``SynnaxLink``; when the
client API changes, that class is the only one to update. The driver itself talks
to any object with the same methods, which is how it is tested without a
Synnax server.

Config (a ``synnax:`` section in the system file, or its own YAML file)::

    synnax:
      host: localhost
      port: 9090
      username: synnax
      password: seldon
      secure: false
      mode: hootl                 # hootl: fake valve states; hitl: read them from the hardware
      valve_state: command        # command: state echoes the command; position: limit switches
      rate: 100                   # Hz written to Synnax
      index: owl_time             # index channel for channels the driver creates
      create_channels: true       # create any missing channel
      check_version: warn         # Core and client major.minor differ: warn, strict (refuse) or off
      sensors:                    # sim sensor -> channel (default: lower-case id)
        PT-OX: pt_ox
      valves:                     # sim valve -> channels, see SynnaxConfig.resolve
        V-OXM: {cmd: ox_mpv_cmd, state: ox_mpv_state}
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .. import units

MODES = ("hootl", "hitl")
VALVE_STATES = ("command", "position")
VERSION_CHECKS = ("warn", "strict", "off")
LIMIT_OPEN, LIMIT_CLOSED = 0.99, 0.01  # faked limit switches (valve_state: position)


def version_mismatch(client: str, core: str) -> str | None:
    """How this client and Synnax Core differ, or None if their major.minor versions match (or are unknown)."""
    def mm(v: str):
        parts = str(v or "").lstrip("v").split(".")
        try:
            return int(parts[0]), int(parts[1])
        except (IndexError, ValueError):
            return None

    c, n = mm(client), mm(core)
    if c is None:
        return None
    if n is None:
        return f"the Synnax Core did not report its version (it is probably older than client {client})"
    if c == n or (0, 0) in (c, n):  # 0.0 marks a development build, which Synnax pairs with anything
        return None
    side = "Core" if c > n else "OwlFluids build"
    return (f"Synnax Core {core} and the Synnax client {client} in this OwlFluids build differ in major.minor, "
            f"which Synnax does not promise to support; upgrade the {side} if channels or writes misbehave")


def check_versions(client, mode: str) -> str | None:
    """Compare a connected client with its Core. Returns a warning, raises for ``strict``, or None.

    Clients older than about 0.50 can't report versions; that counts as unknown, never as a mismatch.
    A check that could not reach the Core raises a connection error, not a version complaint.
    """
    if mode == "off":
        return None
    try:
        state = client.connectivity.check()
    except Exception:  # noqa: BLE001 - no connectivity checker (old client): version unknown
        return None
    if getattr(state, "status", "connected") != "connected":
        raise RuntimeError(f"cannot reach the Synnax Core: {getattr(state, 'message', '') or state.status}")
    if getattr(state, "client_server_compatible", False):  # Synnax's own rule says they pair
        return None
    problem = version_mismatch(getattr(state, "client_version", ""), getattr(state, "node_version", ""))
    if problem and mode == "strict":
        raise RuntimeError(f"{problem} (refused because check_version is strict)")
    return problem


def _channel_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_").replace(".", "_")


@dataclass
class SynnaxConfig:
    host: str = "localhost"
    port: int = 9090
    username: str = "synnax"
    password: str = "seldon"
    secure: bool = False
    mode: str = "hootl"
    valve_state: str = "command"
    rate: float = 100.0
    index: str = "owl_time"
    create_channels: bool = True
    sensors: dict[str, str] = field(default_factory=dict)
    valves: dict[str, dict[str, str]] = field(default_factory=dict)
    valve_threshold: float = 0.5  # a read value above this opens an on/off valve
    check_version: str = "warn"  # Core and client major.minor differ: warn, strict (refuse) or off

    def __post_init__(self):
        self.mode = str(self.mode).strip().lower()
        if self.mode not in MODES:
            raise ValueError(f"synnax mode must be one of {', '.join(MODES)}, not {self.mode!r}")
        self.valve_state = str(self.valve_state).strip().lower()
        if isinstance(self.check_version, bool):
            self.check_version = "warn" if self.check_version else "off"
        self.check_version = str(self.check_version).strip().lower()
        if self.check_version not in VERSION_CHECKS:
            raise ValueError(f"synnax check_version must be one of {', '.join(VERSION_CHECKS)}, "
                             f"not {self.check_version!r}")
        if self.valve_state not in VALVE_STATES:
            raise ValueError(f"synnax valve_state must be one of {', '.join(VALVE_STATES)}, "
                             f"not {self.valve_state!r}")
        for v, spec in self.valves.items():
            bad = set(spec or {}) - {"cmd", "state", "read", "write"}
            if bad:
                raise ValueError(f"unknown settings for valve {v}: {', '.join(sorted(bad))} (use cmd and state)")

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

    def resolve(self, sim) -> tuple[dict[str, str], dict[str, tuple[str, str | None]]]:
        """Sensor -> channel and valve -> (channel read, channel written or None), filling in defaults.

        Each valve has a command channel (default ``<valve>_cmd``) and a state channel
        (default ``<valve>_state``). HOOTL reads the command and writes the state; HITL
        reads the state and writes nothing. ``read:``/``write:`` override either side
        (``read: <cmd channel>`` in HITL follows commands instead of the hardware).
        """
        sensors = dict(self.sensors) or {s.id: _channel_name(s.id) for s in sim.system.sensors}
        for sid in sensors:
            sim.read(sid)  # unknown sensor -> KeyError now, not mid-run
        valves = {}
        names = list(self.valves) or sim.automated_valves()
        manual = set(sim.manual_valves())
        for v in names:
            sim.valve_position(v)
            if v in manual:
                raise ValueError(f"{v} is a manual valve: only the operator works it, from the OwlFluids window, "
                                 "so it can't have Synnax channels")
            spec = self.valves.get(v) or {}
            cmd = spec.get("cmd", f"{_channel_name(v)}_cmd")
            state = spec.get("state", f"{_channel_name(v)}_state")
            if self.mode == "hitl":
                valves[v] = (spec.get("read", state), None)
            else:
                valves[v] = (spec.get("read", cmd), spec.get("write", state))
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
        self.warnings: list[str] = []
        try:
            problem = check_versions(self.client, cfg.check_version)
        except RuntimeError:
            self.client.close()
            raise
        if problem:
            self.warnings.append(f"Synnax: {problem}")
        self.writer = None
        self.streamer = None
        self._t0 = 0
        self._p0 = 0.0
        self._last = 0

    def ensure_channels(self, index: str, data: list[str], commands: list[str], create: bool) -> list[str]:
        """Check (and create) the channels; returns the index channels every write must stamp.

        Data channels that exist keep their own index, so a real stand's sensor and
        valve state channels can be written; missing ones are created on ``index``.
        An index that also stamps a channel the driver reads (a DAQ task's valve
        states, say) is refused: two writers on one index lose each other's data.
        """
        sy = self.sy
        names = [*data, *commands]
        found = {c.name: c for c in self.client.channels.retrieve(names)} if names else {}
        missing = [n for n in names if n not in found]
        new_data = [n for n in data if n in missing]
        idx = None
        if new_data:
            got = self.client.channels.retrieve([index])
            idx = got[0] if got else None
            if idx is None:
                missing.insert(0, index)
        if missing and not create:
            raise RuntimeError(f"missing Synnax channels: {', '.join(missing)}")
        if new_data and idx is None:
            idx = self.client.channels.create(name=index, is_index=True, data_type=sy.DataType.TIMESTAMP)
        if idx is not None and not idx.is_index:
            raise RuntimeError(f"Synnax channel {index} exists but is not an index channel")
        keys = []
        for n in data:
            c = found.get(n)
            if c is None:
                continue
            if c.is_index:
                raise RuntimeError(f"Synnax channel {n} is an index channel, not a data channel")
            if not c.virtual and c.index and c.index not in keys:
                keys.append(c.index)
        shared = {c.index for n in commands if (c := found.get(n)) and not c.virtual and c.index in keys}
        if shared:
            names = ", ".join(c.name for c in self.client.channels.retrieve(sorted(shared)))
            reads = ", ".join(n for n in commands if (c := found.get(n)) and c.index in shared)
            raise RuntimeError(f"Synnax index {names} stamps channels the driver writes and also {reads}, which "
                               "it reads from the hardware; put the sensors the sim fakes on another index "
                               "(or stop that task)")
        indexes = [c.name for c in self.client.channels.retrieve(keys)] if keys else []
        for n in new_data:
            self.client.channels.create(name=n, index=idx.key, data_type=sy.DataType.FLOAT32)
        if new_data and idx.name not in indexes:
            indexes.append(idx.name)
        for n in commands:
            if n in missing:
                self.client.channels.create(name=n, virtual=True, data_type=sy.DataType.FLOAT32)
        return indexes

    def open(self, write: list[str], read: list[str]) -> None:
        sy = self.sy
        self._t0 = int(sy.TimeStamp.now())
        self._p0 = time.perf_counter()
        self._last = self._t0 - 1
        self.writer = self.client.open_writer(start=sy.TimeStamp(self._t0), channels=write, name="OwlFluids",
                                              enable_auto_commit=True, err_on_unauthorized=True)
        self.streamer = self.client.open_streamer(read) if read else None

    def latest(self, names: list[str]) -> dict[str, float]:
        """Last stored value of each channel that has one (virtual channels have none)."""
        out = {}
        for n in names:
            try:
                series = self.client.read_latest(n, 1)
                if len(series):
                    out[n] = float(series[-1])
            except Exception:  # no data yet, or a virtual channel
                pass
        return out

    def write(self, values: dict[str, float], indexes: list[str]) -> None:
        # stamps from a monotonic clock, so a wall-clock step (NTP) never sends time backwards
        stamp = max(self._t0 + int((time.perf_counter() - self._p0) * 1e9), self._last + 1)
        self._last = stamp
        ts = self.sy.TimeStamp(stamp)
        self.writer.write({**{i: ts for i in indexes}, **values})

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
        """Close both ends; the writer re-raises any error it hit, after the streamer is closed."""
        err = None
        for obj in (self.writer, self.streamer):
            if obj is None:
                continue
            try:
                obj.close()
            except Exception as e:
                err = err or e
        if err is not None:
            raise err


class SynnaxDriver:
    """Publishes sensors (and, in HOOTL, valve states) and applies valve channels for a running ``Runner``."""

    def __init__(self, runner, cfg: SynnaxConfig, link_factory: Callable[[SynnaxConfig], object] = SynnaxLink):
        self.runner = runner
        self.cfg = cfg
        sim = runner.sim
        self.sensors, self.valves = cfg.resolve(sim)
        self.units = {}
        for s in sim.system.sensors:
            self.units[s.id] = s.units or units.DEFAULT_UNITS.get(s.type, "")
        self.read_to_valve = {r: v for v, (r, _w) in self.valves.items()}
        self._held: dict[str, float] = {}  # faked limit-switch states
        self.state_out = {v: w for v, (_r, w) in self.valves.items() if w}
        data = [*self.sensors.values(), *self.state_out.values()]
        clash = sorted(set(data) & set(self.read_to_valve)) or sorted({c for c in data if data.count(c) > 1})
        if clash:
            raise ValueError(f"Synnax channels used twice: {', '.join(clash)}")
        self.link = link_factory(cfg)
        for w in getattr(self.link, "warnings", ()):
            runner._warn(w)
        self.indexes = self.link.ensure_channels(cfg.index, data, list(self.read_to_valve), cfg.create_channels)
        if self.indexes is None:  # a link that doesn't report indexes writes everything on cfg.index
            self.indexes = [cfg.index]
        self.link.open([*self.indexes, *data], list(self.read_to_valve))
        # a streamer only sees new writes: start from what the valve channels already say
        latest = getattr(self.link, "latest", None)
        self.initial = latest(list(self.read_to_valve)) if latest else {}
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.writes = 0
        self.commands = 0
        self.error: BaseException | None = None

    def start(self) -> None:
        self._stop.clear()
        for ch, value in self.initial.items():
            self.apply(self.read_to_valve[ch], value)
        self._threads = [threading.Thread(target=self._guard(self._write_loop), name="synnax-write", daemon=True),
                         threading.Thread(target=self._guard(self._read_loop), name="synnax-read", daemon=True)]
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        """Stop and disconnect. Never raises: a close error is kept in ``error``."""
        self._stop.set()
        for t in self._threads:
            t.join(2.0)
        try:
            self.link.close()
        except Exception as e:
            self.error = self.error or e

    def _guard(self, fn):
        def run():
            try:
                fn()
            except BaseException as e:  # reported to the caller / GUI
                self.error = e
        return run

    def values(self, frame) -> dict[str, float]:
        out = {ch: units.convert(frame.sensors[sid], self.units.get(sid, "")) for sid, ch in self.sensors.items()}
        for v, w in self.state_out.items():  # HOOTL: the faked valve states
            out[w] = self._fake_state(v, frame)
        return out

    def _fake_state(self, valve: str, frame) -> float:
        if self.cfg.valve_state == "command":
            cmd = frame.commands[valve]
            return 1.0 if cmd >= 0.5 else 0.0
        pos = frame.valves[valve]
        if pos >= LIMIT_OPEN:
            self._held[valve] = 1.0
        elif pos <= LIMIT_CLOSED:
            self._held[valve] = 0.0
        return self._held.setdefault(valve, 1.0 if pos >= 0.5 else 0.0)

    def _write_loop(self) -> None:
        period = 1.0 / self.cfg.rate
        nxt = time.perf_counter()
        last_t = None
        while not self._stop.is_set():
            frame = self.runner.frame
            if frame.t != last_t:  # nothing new while paused
                self.link.write(self.values(frame), self.indexes)
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
        """A valve channel changed: the valve opens above the threshold and closes below it.

        A HITL state channel that reports position moves the sim valve at once; anything
        else is a command, and the sim strokes the valve.
        """
        pos = 1.0 if value > self.cfg.valve_threshold else 0.0
        if self.cfg.mode == "hitl" and self.cfg.valve_state == "position":
            self.runner.set_position(valve, pos)
        else:
            self.runner.command(valve, pos)
        self.commands += 1
