"""Synnax driver logic against an in-memory stand-in for the Synnax client."""
import threading
import time

import pytest

from owlfluids.drivers.synnax import SynnaxConfig, SynnaxDriver
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

    def open(self, write, read):
        self.write_channels, self.read_channels = write, read

    def write(self, values, index):
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
