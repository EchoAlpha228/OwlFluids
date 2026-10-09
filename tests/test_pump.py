"""Air pumps: on at the start pressure, off at the stop pressure, at their rated flow."""
import pytest

from owlfluids.core.network import BuildError
from owlfluids.io.system import System
from owlfluids.sim.simulation import Simulation

PSIG = 6894.757


def _receiver(vent_cv=0.03):
    s = System("pump")
    s.add("C", "air_pump", flow="5 g/s", start_pressure="100 psig", stop_pressure="120 psig", spin_up="0.2 s")
    s.add("R", "volume", volume="20 L", fluid="air", pressure="1 atm", temperature="293 K")
    s.add("V", "valve", cv=vent_cv, open_time="50 ms")
    s.add("ATM", "boundary", fluid="air")
    s.connect("C.out", "R.a")
    s.connect("R.b", "V.in")
    s.connect("V.out", "ATM.port")
    s.sense("PT", "pressure", "R.a", units="psig")
    s.sense("ON", "valve_position", "C")
    return s


def _gauge(sim):
    return (sim.nout[sim.node("R.a"), 0] - 101325.0) / PSIG


def test_pump_fills_to_its_stop_pressure_at_its_rated_flow():
    sim = Simulation(_receiver())
    m0 = sim.mass("R")
    sim.run(10.0)
    assert sim.pump_running("C")
    assert sim.mass("R") - m0 == pytest.approx(5e-3 * (10.0 - 0.1), rel=0.02)  # less half the spin-up
    sim.run(40.0)
    assert not sim.pump_running("C")
    assert 119.0 < _gauge(sim) < 122.0
    assert sim.read("ON") == 0.0


def test_pump_restarts_at_its_start_pressure():
    sim = Simulation(_receiver())
    sim.run(40.0)
    assert not sim.pump_running("C")
    sim.open("V")  # air is used
    lowest = 1e9
    restarted = False
    for _ in range(600):
        sim.run(0.1)
        lowest = min(lowest, _gauge(sim))
        if sim.pump_running("C"):
            restarted = True
            break
    assert restarted
    assert 99.0 < lowest < 100.5


def test_stop_pressure_must_be_above_start():
    s = _receiver()
    s.component("C").params["stop_pressure"] = "90 psig"
    with pytest.raises((BuildError, ValueError), match="stop_pressure"):
        Simulation(s)
