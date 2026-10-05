"""Two-phase tanks (LN2 dewar, nitrous) and line chilldown."""
import CoolProp.CoolProp as CP
import pytest

from owlfluids.examples import ln2_wet_dress, n2o_blowdown
from owlfluids.io.system import System
from owlfluids.sim.simulation import Simulation

PSI = 6894.757

# fixed work budget, so forced-step counts don't depend on the CI machine's speed
BUDGET = 200


@pytest.mark.slow
def test_closed_ln2_dewar_self_pressurizes_along_saturation():
    s = System("dewar")
    s.add("D", "tank", fluid="nitrogen", volume="180 L", fill=0.8, pressure="22 psi", heat_transfer=5.0)
    s.add("V", "valve", cv=1.0)
    s.add("A", "boundary", fluid="nitrogen")
    s.connect("D.top", "V.in")
    s.connect("V.out", "A.port")
    sim = Simulation(s, step_budget=BUDGET)
    m0 = sim.mass("D")
    p_prev = sim.pressure("D")
    for _ in range(3):
        sim.run(60.0)
        p, t = sim.pressure("D"), sim.temperature("D")
        assert p > p_prev
        assert 0.0 < sim.quality("D") < 0.05
        assert p == pytest.approx(CP.PropsSI("P", "T", t, "Q", 0, "Nitrogen"), rel=0.02)
        p_prev = p
    assert sim.mass("D") == pytest.approx(m0, rel=1e-9)
    # venting the vapor space drops the pressure and draws vapor, not liquid
    sim.open("V")
    sim.run(10.0)
    assert sim.pressure("D") < p_prev
    assert sim.solver_stats()["forced_steps"] == 0


@pytest.mark.slow
def test_nitrous_blowdown_holds_near_vapor_pressure_and_cools():
    sim = Simulation(n2o_blowdown(), step_budget=BUDGET)
    p0, t0 = sim.pressure("K"), sim.temperature("K")
    assert p0 == pytest.approx(CP.PropsSI("P", "T", 293.15, "Q", 0, "NitrousOxide"), rel=0.02)
    sim.open("V")
    sim.run(4.0)
    assert sim.temperature("K") < t0 - 0.5  # evaporative cooling
    assert sim.pressure("K") == pytest.approx(CP.PropsSI("P", "T", sim.temperature("K"), "Q", 0, "NitrousOxide"),
                                              rel=0.02)
    assert 0.1 < sim.mass_flow("INJ") < 0.3
    assert sim.solver_stats()["forced_steps"] == 0


@pytest.mark.slow
def test_ln2_chills_a_warm_line_until_liquid_reaches_the_end():
    s = System("chill")
    s.add("TK", "pressurized_tank", liquid="nitrogen", pressurant="nitrogen", volume="200 L", fill=0.8,
          pressure="60 psi", liquid_temperature="80 K")
    s.add("V1", "valve", cv=2.0, open_time="100 ms")
    s.add("L1", "pipe", length="5 m", diameter="0.5 in", wall_thickness="0.065 in", segments=5)
    s.add("O1", "orifice", cda="20 mm2")
    s.add("A", "boundary", fluid="nitrogen")
    for a, b in [("TK.liquid", "V1.in"), ("V1.out", "L1.in"), ("L1.out", "O1.in"), ("O1.out", "A.port")]:
        s.connect(a, b)
    sim = Simulation(s, step_budget=BUDGET)
    i_wall = sim.net.node_names.index("L1.s2")
    sx = sim.net.NI[i_wall, 3] + 2
    assert sim.x[sx] == pytest.approx(293.15)
    sim.open("V1")
    walls = []
    for _ in range(8):
        sim.run(5.0)
        walls.append(sim.x[sx])
    assert all(b < a for a, b in zip(walls, walls[1:], strict=False))  # the wall only cools
    assert walls[-1] < 130.0
    assert sim.quality("O1.in") < 0.1  # liquid (or nearly) at the outlet by the end
    assert sim.mass_flow("O1") > 0.2
    assert sim.solver_stats()["forced_steps"] < 60  # violent boiling in the first seconds


@pytest.mark.slow
def test_wet_dress_example_runs_cleanly():
    sim = Simulation(ln2_wet_dress(), step_budget=BUDGET)
    sim.open("V-P")
    sim.run(5.0)
    assert sim.read("PT-TK") > 100 * PSI
    sim.open("V-M")
    sim.run(10.0)
    assert sim.read("TC-INJ") < 90.0
    assert sim.read("TC-LINE") < 250.0
    assert 0.5 < sim.read("FM-LN2") < 1.5
    assert sim.solver_stats()["forced_steps"] == 0
