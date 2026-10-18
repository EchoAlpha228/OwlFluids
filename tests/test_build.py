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
    # two propellants may not meet; a purge gas may (see test_purge.py)
    s = System("bad")
    s.add("A", "tank", fluid="methane", volume="1 L", pressure="10 bar", temperature="300 K")
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
    s.add("F1", "orifice", cv=1.0)
    s.add("V1", "valve", cv=0.1)
    s.add("F2", "orifice", cv=1.0)
    s.add("AMB", "boundary", fluid="nitrogen")
    for a, b in [("K.port", "F1.in"), ("F1.out", "V1.in"), ("V1.out", "F2.in"), ("F2.out", "AMB.port")]:
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


def _tank_system(liquid, pressure, **kw):
    s = System("tank")
    s.add("TK", "pressurized_tank", liquid=liquid, pressurant="nitrogen", volume="20 L", fill=0.8,
          pressure=pressure, **kw)
    s.add("F", "fitting", kind="elbow90", diameter="10 mm")
    s.add("V", "valve", cv="5e-2")  # numbers written as text still parse
    s.add("A", "boundary", fluid=liquid)
    s.connect("TK.liquid", "F.in"), s.connect("F.out", "V.in"), s.connect("V.out", "A.port")
    return s


def test_tank_liquid_starts_at_its_set_temperature():
    for liquid, p, t in (("nitrous", "1200 psi", 280.0), ("oxygen", "1000 psi", 80.0)):
        sim = Simulation(_tank_system(liquid, p, liquid_temperature=f"{t} K"))
        assert sim.temperature("TK.liquid") == pytest.approx(t, abs=0.05)
        assert sim.quality("V.in") < 0  # the feed line is primed with liquid


def test_tank_liquid_that_would_boil_is_an_error_or_defaults_cold():
    with pytest.raises(BuildError, match="not a liquid"):
        build_network(_tank_system("oxygen", "50 psi", liquid_temperature="120 K"))
    net = build_network(_tank_system("oxygen", "50 psi"))  # no temperature: just below boiling
    assert any("boiling point" in w for w in net.warnings)
    net = build_network(_tank_system("oxygen", "50 psi", liquid_temperature="103.6 K"))  # within 0.5 K
    assert any("too close to boiling" in w for w in net.warnings)
    from owlfluids.fluids.registry import FluidSet
    fs = FluidSet(["oxygen"], ["table"])
    t_boil = fs.saturation_temperature(0, 50 * 6894.757)
    net = build_network(_tank_system("oxygen", "50 psi", liquid_temperature=f"{t_boil!r} K"))  # exactly boiling
    assert any("too close to boiling" in w for w in net.warnings)


def test_example_files_match_the_built_in_examples():
    """examples/*.yaml are written by `owlfluids example`; regenerate them when an example changes."""
    from pathlib import Path

    from owlfluids.examples import EXAMPLES

    root = Path(__file__).resolve().parents[1] / "examples"
    for name, make in EXAMPLES.items():
        assert System.load(root / f"{name}.yaml").to_dict() == make().to_dict(), f"examples/{name}.yaml is stale"


def test_synnax_example_maps_lox_feed_file():
    from pathlib import Path

    from owlfluids.drivers.synnax import SynnaxConfig

    root = Path(__file__).resolve().parents[1] / "examples"
    SynnaxConfig.load(root / "synnax.yaml").resolve(Simulation(System.load(root / "lox-feed.yaml")))
