"""Relief valves: they vent to the atmosphere with nothing on their outlet, and say when wired backwards."""
import pytest

from owlfluids.io.system import System
from owlfluids.sim.simulation import Simulation

PSI = 6894.757


def _bottle(*wires):
    s = System("rv")
    s.add("TK", "tank", fluid="nitrogen", volume="10 L", pressure="1950 psi", temperature="293 K")
    s.add("ORV", "relief_valve", cv=0.2, set_pressure="1200 psi")
    s.add("ATM", "boundary", fluid="nitrogen")
    for a, b in wires:
        s.connect(a, b)
    return Simulation(s)


@pytest.mark.parametrize("wires", [[("TK.top", "ORV.in"), ("ORV.out", "ATM.port")], [("TK.top", "ORV.in")]],
                         ids=["piped to a vent", "outlet left open"])
def test_a_relief_valve_relieves_down_toward_its_set_pressure(wires):
    sim = _bottle(*wires)
    sim.run(2.0)
    assert sim.mass_flow("ORV") > 0.05
    sim.run(30.0)
    assert 1200 * PSI < sim.pressure("TK.top") < 1250 * PSI  # it reseats near its set pressure (gauge over 1 atm)


def test_a_backwards_relief_valve_says_so():
    sim = _bottle(("TK.top", "ORV.out"), ("ORV.in", "ATM.port"))
    assert any("ORV: its inlet is on the vent ATM" in w for w in sim.net.warnings)
    sim.run(2.0)
    assert sim.mass_flow("ORV") == 0.0


def test_an_unconnected_valve_port_is_capped_and_says_so():
    s = System("capped")
    s.add("TK", "tank", fluid="nitrogen", volume="10 L", pressure="100 psi", temperature="293 K")
    s.add("V", "valve", cv=0.2)
    s.connect("TK.top", "V.in")
    sim = Simulation(s)
    assert "V.out is not connected to anything, so it is capped" in sim.net.warnings
