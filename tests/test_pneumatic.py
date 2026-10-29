"""Pneumatic valves: they draw on their air supply only while held away from their normal state."""
import pytest

from owlfluids.core.network import BuildError
from owlfluids.io.system import System
from owlfluids.sim.simulation import Simulation

PSIG = 6894.757
ATM = 101325.0


def _stand(supply="100 psig", **valve):
    """A GN2 line through a normally closed and a normally open valve, both run on a 2 L air supply."""
    s = System("pneumatic")
    s.add("A", "pneumatic_supply", volume="2 L", fluid="air", pressure=supply, temperature="293 K")
    s.add("V-DUMP", "valve", cv=0.5, actuator="solenoid")  # empties the supply
    s.add("ATM-A", "boundary", fluid="air")
    s.connect("A.a", "V-DUMP.in")
    s.connect("V-DUMP.out", "ATM-A.port")
    s.add("N2", "boundary", fluid="nitrogen", pressure="500 psig")
    s.add("NC", "valve", cv=1, **valve)
    s.add("M", "volume", volume="1 L", fluid="nitrogen", pressure="1 atm")
    s.add("NO", "valve", cv=1, normally="open", **valve)
    s.add("ATM", "boundary", fluid="nitrogen")
    for a, b in [("N2.port", "NC.in"), ("NC.out", "M.a"), ("M.b", "NO.in"), ("NO.out", "ATM.port")]:
        s.connect(a, b)
    return s


def _psig(sim):
    return (sim.nout[sim.node("A"), 0] - ATM) / PSIG


def test_normally_closed_draws_to_open_and_vents_to_close():
    sim = Simulation(_stand())
    m0 = sim.mass("A")
    sim.run(0.5)
    assert sim.mass("A") == pytest.approx(m0, rel=1e-9)  # nothing held: nothing drawn
    sim.open("NC")
    sim.run(0.05)
    assert sim.air_draw("NC") > 0.0
    assert sim.valve_position("NC") < 0.5  # still filling its actuator
    sim.run(1.0)
    assert sim.valve_position("NC") == 1.0
    assert sim.actuator_pressure("NC") == pytest.approx(sim.nout[sim.node("A"), 0], abs=50.0)
    assert sim.air_draw("NC") == 0.0  # held open on a charged actuator
    # the supply gave up about one actuator's worth of air: 100 cc at the supply's pressure
    # (more, as the header cools while it empties)
    m_act = (sim.actuator_pressure("NC") - ATM) * 100e-6 / (287.0 * 293.0)
    assert m_act < m0 - sim.mass("A") < 1.6 * m_act
    m1 = sim.mass("A")
    sim.close("NC")
    sim.run(1.0)
    assert sim.valve_position("NC") == 0.0
    assert sim.actuator_pressure("NC") == pytest.approx(ATM)
    assert sim.mass("A") == pytest.approx(m1, rel=1e-9)  # closing vents to the atmosphere


def test_normally_open_draws_to_close():
    sim = Simulation(_stand())
    m0 = sim.mass("A")
    sim.open("NO")  # already open: nothing to do
    sim.run(0.5)
    assert sim.mass("A") == pytest.approx(m0, rel=1e-9)
    assert sim.valve_position("NO") == 1.0
    sim.close("NO")
    sim.run(1.0)
    assert sim.valve_position("NO") == 0.0
    assert sim.mass("A") < m0
    m1 = sim.mass("A")
    sim.open("NO")
    sim.run(1.0)
    assert sim.valve_position("NO") == 1.0
    assert sim.mass("A") == pytest.approx(m1, rel=1e-9)


def test_low_supply_strokes_partway_or_not_at_all():
    # strokes between 30 and 60 psig of actuator pressure
    sim = Simulation(_stand("45 psig"))
    sim.open("NC")
    sim.run(1.0)
    assert 0.2 < sim.valve_position("NC") < 0.6
    sim = Simulation(_stand("20 psig"))
    sim.open("NC")
    sim.run(1.0)
    assert sim.valve_position("NC") == 0.0


def test_losing_the_supply_springs_valves_back():
    sim = Simulation(_stand())
    sim.open("NC")
    sim.close("NO")
    sim.run(1.0)
    assert sim.valve_position("NC") == 1.0 and sim.valve_position("NO") == 0.0
    sim.open("V-DUMP")  # a solenoid valve: it never draws air itself
    sim.run(3.0)
    assert _psig(sim) < 5.0
    assert sim.valve_position("NC") == 0.0  # fails closed
    assert sim.valve_position("NO") == 1.0  # fails open
    assert sim.actuator_pressure("V-DUMP") is None


def test_stroke_settings():
    fast = Simulation(_stand(pilot_cv=1.0))
    slow = Simulation(_stand())
    for sim in (fast, slow):
        sim.open("NC")
        sim.run(0.1)
    assert fast.valve_position("NC") == 1.0
    assert slow.valve_position("NC") < 0.5


def test_without_a_supply_valves_move_on_command():
    s = _stand()
    s.components = [c for c in s.components if c.id not in ("A", "V-DUMP", "ATM-A")]
    s.connections = [c for c in s.connections if not any(e.startswith(("A.", "V-DUMP.")) for e in c)]
    sim = Simulation(s)
    assert sim.net.actuators == {}
    sim.open("NC")
    sim.run(0.06)
    assert sim.valve_position("NC") == 1.0
    assert sim.actuator_pressure("NC") is None


def test_deleting_the_supply_turns_off_the_air_sim():
    """Valves still naming a supply that was deleted move on command, with a note, not a build error."""
    s = _stand(actuator="pneumatic", air_supply="A")
    s.components = [c for c in s.components if c.id not in ("A", "V-DUMP", "ATM-A")]
    s.connections = [c for c in s.connections if not any(e.startswith(("A.", "V-DUMP.")) for e in c)]
    sim = Simulation(s)
    assert sim.net.actuators == {}
    assert any(w.startswith("NC: air_supply 'A' is not in the system") for w in sim.net.warnings)
    sim.open("NC")
    sim.close("NO")
    sim.run(0.06)
    assert sim.valve_position("NC") == 1.0
    assert sim.valve_position("NO") == 0.0


def test_solenoid_type_and_actuator_setting_draw_nothing():
    s = _stand(actuator="solenoid")
    s.add("SOL", "solenoid", cv=0.1)
    s.add("ATM2", "boundary", fluid="nitrogen")
    s.connect("M.c", "SOL.in")
    s.connect("SOL.out", "ATM2.port")
    sim = Simulation(s)
    assert sim.net.actuators == {}


def test_valves_assumed_pneumatic_are_named():
    sim = Simulation(_stand())
    assert any(w.startswith("A runs NC, NO as pneumatic valves") for w in sim.net.warnings)
    sim = Simulation(_stand(actuator="pneumatic"))
    assert not any("pneumatic valves" in w for w in sim.net.warnings)


def test_choosing_a_supply():
    s = _stand()
    s.add("B", "pneumatic_supply", volume="1 L", fluid="nitrogen", pressure="80 psig")
    with pytest.raises(BuildError, match="set air_supply to one of A, B"):
        Simulation(s)
    s.component("NC").params["air_supply"] = "B"
    s.component("NO").params["air_supply"] = "A"
    sim = Simulation(s)
    m_b = sim.mass("B")
    sim.open("NC")
    sim.run(1.0)
    assert sim.mass("B") < m_b
    s.component("NO").params["air_supply"] = "M"
    with pytest.raises(BuildError, match="'M' is not a pneumatic supply"):
        Simulation(s)


def test_measured_position_keeps_its_actuator():
    sim = Simulation(_stand())
    sim.set_position("NC", 1.0)
    sim.run(0.5)
    assert sim.valve_position("NC") == 1.0
    sim.set_position("NC", 0.0)
    sim.run(0.5)
    assert sim.valve_position("NC") == 0.0


@pytest.mark.parametrize("n, supply, dt, pilot_cv", [(16, "2 L", "5 ms", 0.5), (30, "0.5 L", "10 ms", 0.1)])
def test_many_actuators_on_a_small_supply_settle(n, supply, dt, pilot_cv):
    """Actuators together hold as much air as the supply: filling them must not ring."""
    s = System("bank")
    s.settings["dt"] = dt
    s.add("A", "pneumatic_supply", volume=supply, fluid="air", pressure="100 psig", temperature="293 K")
    s.add("N2", "boundary", fluid="nitrogen", pressure="50 psig")
    s.add("ATM", "boundary", fluid="nitrogen")
    for i in range(n):
        s.add(f"V{i}", "valve", cv=0.05, pilot_cv=pilot_cv)
        s.connect("N2.port", f"V{i}.in")
        s.connect(f"V{i}.out", "ATM.port")
    sim = Simulation(s)
    sim.open(*[f"V{i}" for i in range(n)])
    sim.run(3.0)
    p = []
    for _ in range(40):
        sim.run(sim.dt)
        p.append(sim.nout[sim.node("A"), 0])
    assert max(p) - min(p) < 100.0  # Pa
    assert all(abs(sim.actuator_pressure(f"V{i}") - p[-1]) < 100.0 for i in range(n))
    # the supply ends up shared with the actuators, too low to stroke them fully: all alike, and steady
    x = sim.valve_position("V0")
    assert x < 1.0
    assert all(sim.valve_position(f"V{i}") == x for i in range(n))


def test_an_uncharged_supply_is_flagged():
    sim = Simulation(_stand("1 atm"))
    assert any(w.startswith("A starts at 0 psig, below the 30 psig its valves need") for w in sim.net.warnings)
    sim = Simulation(_stand())
    assert not any("its valves need" in w for w in sim.net.warnings)
