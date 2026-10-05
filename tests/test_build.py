import pytest

from owlfluids.core.network import BuildError, build_network
from owlfluids.io.system import System
from owlfluids.sim.simulation import Simulation


def test_yaml_round_trip(tmp_path, gas_blowdown):
    s = gas_blowdown()
    s.sense("PT-1", "pressure", "TK.port", units="psi")
    p = tmp_path / "sys.yaml"
    s.save(p)
    s2 = System.load(p)
    assert s2.to_dict() == s.to_dict()
    sim = Simulation(s2)
    assert sim.read("PT-1") == pytest.approx(200e5, rel=1e-2)  # table interpolation


def test_two_fluids_on_one_line_is_an_error():
    s = System("bad")
    s.add("A", "tank", fluid="nitrogen", volume="1 L", pressure="10 bar", temperature="300 K")
    s.add("B", "tank", fluid="oxygen", volume="1 L", pressure="10 bar", temperature="300 K")
    s.add("V", "valve", cv=0.1)
    s.connect("A.port", "V.in")
    s.connect("V.out", "B.port")
    with pytest.raises(BuildError, match="one fluid"):
        build_network(s)


def test_unknown_port_is_an_error():
    s = System("bad")
    s.add("A", "tank", fluid="nitrogen", volume="1 L", pressure="10 bar", temperature="300 K")
    s.add("V", "valve", cv=0.1)
    s.connect("A.nope", "V.in")
    with pytest.raises(BuildError, match="no port"):
        build_network(s)


def test_lines_prime_through_open_paths_only():
    s = System("prime")
    s.add("K", "tank", fluid="nitrogen", volume="10 L", pressure="100 bar", temperature="300 K")
    s.add("L1", "pipe", length="1 m", diameter="6 mm")
    s.add("V1", "valve", cv=0.1)
    s.add("L2", "pipe", length="1 m", diameter="6 mm")
    s.add("AMB", "boundary", fluid="nitrogen")
    for a, b in [("K.port", "L1.in"), ("L1.out", "V1.in"), ("V1.out", "L2.in"), ("L2.out", "AMB.port")]:
        s.connect(a, b)
    sim = Simulation(s)
    assert sim.pressure("V1.in") == pytest.approx(100e5, rel=5e-3)
    assert sim.pressure("V1.out") == pytest.approx(101325, rel=5e-3)
    s.settings["prime_lines"] = False
    sim = Simulation(s)
    assert sim.pressure("V1.in") == pytest.approx(101325, rel=5e-3)


def test_valve_commands_and_stroke_time(gas_blowdown):
    s = gas_blowdown()
    s.add("V", "valve", cv=0.2, open_time="100 ms")
    s.connect("TK.port", "V.in")
    s.add("AMB2", "boundary", fluid="nitrogen")
    s.connect("V.out", "AMB2.port")
    sim = Simulation(s)
    sim.open("V")
    sim.run(0.05)
    assert sim.valve_position("V") == pytest.approx(0.5, abs=0.02)
    sim.run(0.06)
    assert sim.valve_position("V") == 1.0
    sim.command("V", 0.25)
    sim.run(0.2)
    assert sim.valve_position("V") == pytest.approx(0.25)
    with pytest.raises(KeyError):
        sim.open("NOPE")
