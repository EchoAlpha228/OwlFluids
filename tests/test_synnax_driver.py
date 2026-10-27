"""Synnax driver logic against an in-memory stand-in for the Synnax client."""
import threading
import time
from types import SimpleNamespace

import pytest

from owlfluids.drivers.synnax import SynnaxConfig, SynnaxDriver, SynnaxLink
from owlfluids.examples import lox_feed
from owlfluids.sim.realtime import Runner
from owlfluids.sim.simulation import Simulation


class FakeLink:
    instances = []

    def __init__(self, cfg):
        self.channels = {}
        self.written = []
        self.inbox = []
        self.lock = threading.Lock()
        FakeLink.instances.append(self)

    def ensure_channels(self, index, data, commands, create):
        self.channels = {"index": index, "data": data, "commands": commands}
        return [index]

    def open(self, write, read):
        self.write_channels, self.read_channels = write, read

    def write(self, values, indexes):
        with self.lock:
            self.written.append(dict(values))

    def read(self, timeout):
        with self.lock:
            if self.inbox:
                return self.inbox.pop(0)
        time.sleep(timeout / 10)
        return {}

    def close(self):
        self.closed = True


def test_default_channel_names_and_units():
    sim = Simulation(lox_feed())
    sensors, valves = SynnaxConfig().resolve(sim)
    assert sensors["PT-TK"] == "pt_tk"
    assert valves["V-M"] == ("v_m_cmd", "v_m_state")
    with pytest.raises(KeyError):
        SynnaxConfig(sensors={"NOPE": "x"}).resolve(sim)
    with pytest.raises(ValueError):
        SynnaxConfig.from_dict({"hots": "typo"})


def test_modes_pick_valve_channels():
    sim = Simulation(lox_feed())
    spec = {"V-M": {"cmd": "mpv_cmd", "state": "mpv_state"}, "V-P": {"read": "a", "write": "b"}}
    _, hootl = SynnaxConfig(valves=spec).resolve(sim)
    assert hootl == {"V-M": ("mpv_cmd", "mpv_state"), "V-P": ("a", "b")}  # read/write still override
    _, hitl = SynnaxConfig.from_dict({"mode": "HITL", "valves": spec}).resolve(sim)
    assert hitl == {"V-M": ("mpv_state", None), "V-P": ("a", None)}  # hardware owns the states
    _, hitl = SynnaxConfig(mode="hitl").resolve(sim)
    assert hitl["V-M"] == ("v_m_state", None)
    with pytest.raises(ValueError):
        SynnaxConfig(mode="hils")
    with pytest.raises(ValueError):
        SynnaxConfig(valves={"V-M": {"command": "x"}})
    with pytest.raises(ValueError):  # a state channel the driver would both write and read
        SynnaxDriver(Runner(sim, rate=0), SynnaxConfig(valves={"V-M": {"cmd": "x", "state": "x"}}),
                     link_factory=FakeLink)


def test_hitl_reads_hardware_states_and_writes_none():
    runner = Runner(Simulation(lox_feed()), rate=0)
    drv = SynnaxDriver(runner, SynnaxConfig(mode="hitl", rate=50), link_factory=FakeLink)
    link = FakeLink.instances[-1]
    assert link.read_channels == ["v_p_state", "v_m_state"]
    assert not any(c.endswith(("_state", "_cmd")) for c in link.write_channels)
    assert "pt_tk" in link.write_channels
    with link.lock:
        link.inbox.append({"v_m_state": 1.0})
    drv.start()  # runner not started: commands apply at once
    try:
        deadline = time.perf_counter() + 2.0
        while runner.sim.valve_command("V-M") != 1.0 and time.perf_counter() < deadline:
            time.sleep(0.01)
    finally:
        drv.stop()
    assert runner.sim.valve_command("V-M") == 1.0
    assert not any("v_m_state" in w for w in link.written)


def test_hootl_fakes_states_from_command_or_limit_switches():
    runner = Runner(Simulation(lox_feed()), rate=0)
    frame = runner.frame
    echo = SynnaxDriver(runner, SynnaxConfig(), link_factory=FakeLink)
    frame.valves["V-M"] = 0.0
    frame.commands["V-M"] = 1.0
    assert echo.values(frame)["v_m_state"] == 1.0  # a DO task's state echoes the command at once
    switches = SynnaxDriver(runner, SynnaxConfig(valve_state="position"), link_factory=FakeLink)
    states = []
    for pos in (0.0, 0.5, 0.98, 1.0, 0.5, 0.02, 0.0):  # open stroke, then close stroke
        frame.valves["V-M"] = pos
        states.append(switches.values(frame)["v_m_state"])
    assert states == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]  # held mid-stroke, flips only at the ends
    with pytest.raises(ValueError):
        SynnaxConfig(valve_state="switch")


def test_hitl_position_states_skip_the_stroke():
    runner = Runner(Simulation(lox_feed()), rate=0)
    drv = SynnaxDriver(runner, SynnaxConfig(mode="hitl", valve_state="position"), link_factory=FakeLink)
    drv.apply("V-M", 1.0)  # runner not started: applied at once
    assert runner.sim.valve_position("V-M") == 1.0 and runner.sim.valve_command("V-M") == 1.0
    runner = Runner(Simulation(lox_feed()), rate=0)
    cmd = SynnaxDriver(runner, SynnaxConfig(mode="hitl"), link_factory=FakeLink)
    cmd.apply("V-M", 1.0)  # a command echo: the sim strokes the valve
    assert runner.sim.valve_command("V-M") == 1.0 and runner.sim.valve_position("V-M") == 0.0


class FakeClient:
    """Just enough of synnax.Synnax.channels for SynnaxLink.ensure_channels."""

    def __init__(self, channels):
        self.store = {c.name: c for c in channels}
        self.channels = self
        self.created = []

    def retrieve(self, names):
        return [c for c in self.store.values() if c.name in names or c.key in names]

    def create(self, name, is_index=False, index=0, virtual=False, data_type=None):
        c = SimpleNamespace(name=name, key=100 + len(self.store), is_index=is_index, index=index, virtual=virtual)
        self.store[name] = c
        self.created.append(name)
        return c


def _channel(name, key, index=0, is_index=False, virtual=False):
    return SimpleNamespace(name=name, key=key, index=index, is_index=is_index, virtual=virtual)


def test_link_keeps_existing_indexes_and_creates_missing_on_its_own():
    link = object.__new__(SynnaxLink)
    link.sy = SimpleNamespace(DataType=SimpleNamespace(TIMESTAMP="ts", FLOAT32="f32"))
    link.client = FakeClient([_channel("ai_time", 1, is_index=True), _channel("pt_tk", 2, index=1),
                              _channel("do_time", 3, is_index=True), _channel("mpv_state", 4, index=3),
                              _channel("mpv_cmd", 5, virtual=True)])
    # every channel already on the stand: write its indexes, create nothing
    assert link.ensure_channels("owl_time", ["pt_tk", "mpv_state"], ["mpv_cmd"], False) == ["ai_time", "do_time"]
    assert link.client.created == []
    with pytest.raises(RuntimeError, match="owl_time, pt_inj"):
        link.ensure_channels("owl_time", ["pt_tk", "pt_inj"], [], False)
    indexes = link.ensure_channels("owl_time", ["pt_tk", "pt_inj"], ["press_cmd"], True)
    assert indexes == ["ai_time", "owl_time"]
    assert link.client.created == ["owl_time", "pt_inj", "press_cmd"]
    assert link.client.store["pt_inj"].index == link.client.store["owl_time"].key
    assert link.client.store["press_cmd"].virtual
    with pytest.raises(RuntimeError, match="index channel"):
        link.ensure_channels("owl_time", ["ai_time"], [], True)


def test_link_refuses_an_index_the_hardware_also_writes():
    link = object.__new__(SynnaxLink)
    link.sy = SimpleNamespace(DataType=SimpleNamespace(TIMESTAMP="ts", FLOAT32="f32"))
    # one DAQ task stamps both the sensors and the valve states on daq_time
    link.client = FakeClient([_channel("daq_time", 1, is_index=True), _channel("pt_tk", 2, index=1),
                              _channel("mpv_state", 3, index=1)])
    with pytest.raises(RuntimeError, match="daq_time.*mpv_state"):
        link.ensure_channels("owl_time", ["pt_tk"], ["mpv_state"], True)


@pytest.mark.slow
def test_driver_publishes_sensors_and_applies_valve_commands():
    runner = Runner(Simulation(lox_feed()), rate=0)  # as fast as possible
    cfg = SynnaxConfig(rate=200)
    drv = SynnaxDriver(runner, cfg, link_factory=FakeLink)
    link = FakeLink.instances[-1]
    assert "pt_tk" in link.channels["data"] and "v_m_state" in link.channels["data"]
    assert link.channels["commands"] == ["v_p_cmd", "v_m_cmd"]
    runner.start()
    drv.start()
    try:
        runner.wait(0.2)
        with link.lock:
            link.inbox.append({"v_m_cmd": 1.0})
        runner.wait_until(lambda f: f.valves["V-M"] > 0.99, timeout=5.0)
        runner.wait(0.5)
        time.sleep(0.1)
    finally:
        drv.stop()
        runner.stop()
    assert drv.error is None
    assert drv.commands == 1
    last = link.written[-1]
    assert last["v_m_state"] == pytest.approx(1.0)
    assert 100 < last["pt_tk"] < 400  # psi, the sensor's display unit
    assert last["fm_lox"] > 0.5


class StoredLink(FakeLink):
    """A link whose valve channel already holds a value, and whose writer fails on close."""

    def latest(self, names):
        return {n: 1.0 for n in names if n == "v_m_cmd"}

    def close(self):
        raise RuntimeError("writer lost its connection")


def test_driver_syncs_valves_at_connect_and_stops_cleanly():
    runner = Runner(Simulation(lox_feed()), rate=0)
    vp = runner.sim.valve_command("V-P")
    assert runner.sim.valve_command("V-M") == 0.0
    drv = SynnaxDriver(runner, SynnaxConfig(rate=50), link_factory=StoredLink)
    drv.start()  # runner not started: commands apply at once
    assert runner.sim.valve_command("V-M") == 1.0
    assert runner.sim.valve_command("V-P") == vp  # no stored value: left alone
    drv.stop()  # must not raise
    assert isinstance(drv.error, RuntimeError)


def test_version_mismatch():
    from owlfluids.drivers.synnax import version_mismatch

    assert version_mismatch("0.60.0", "0.60.3") is None
    assert version_mismatch("0.60.0", "v0.60.1") is None
    assert "upgrade the Core" in version_mismatch("0.60.0", "0.59.2")
    assert "upgrade the OwlFluids build" in version_mismatch("0.59.0", "0.60.0")
    assert "did not report" in version_mismatch("0.60.0", "")
    assert version_mismatch("", "0.60.0") is None  # client version unknown: nothing to compare


def test_check_versions_modes():
    from owlfluids.drivers.synnax import check_versions

    def client(core):
        state = SimpleNamespace(client_version="0.60.0", node_version=core)
        return SimpleNamespace(connectivity=SimpleNamespace(check=lambda: state))

    assert check_versions(client("0.60.2"), "warn") is None
    assert "0.58.2" in check_versions(client("0.58.2"), "warn")  # warns, still connects
    with pytest.raises(RuntimeError, match="strict"):
        check_versions(client("0.58.2"), "strict")
    assert check_versions(client("0.58.2"), "off") is None
    assert check_versions(SimpleNamespace(), "strict") is None  # an old client can't report versions

    def broken():
        raise ConnectionError("down")

    assert check_versions(SimpleNamespace(connectivity=SimpleNamespace(check=broken)), "strict") is None
    assert SynnaxConfig.from_dict({"check_version": False}).check_version == "off"
    assert SynnaxConfig.from_dict({"check_version": True}).check_version == "warn"
    with pytest.raises(ValueError, match="check_version"):
        SynnaxConfig.from_dict({"check_version": "maybe"})


def test_link_warnings_reach_the_runner():
    class WarningLink(FakeLink):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.warnings = ["Synnax: versions differ"]

    runner = Runner(Simulation(lox_feed()))
    SynnaxDriver(runner, SynnaxConfig(), link_factory=WarningLink)
    assert "Synnax: versions differ" in runner.warnings
