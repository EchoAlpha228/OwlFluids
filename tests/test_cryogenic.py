"""Cryogenic liquid feeds: primed lines, flow into warm lines, conservation."""
import pytest

from owlfluids.io.system import System
from owlfluids.sim.simulation import Simulation

# fixed work budget, so forced-step counts don't depend on the CI machine's speed
BUDGET = 200


def lox_feed():
    s = System("lox feed")
    s.add("TK", "pressurized_tank", liquid="oxygen", pressurant="nitrogen", volume="60 L", fill=0.85,
          pressure="80 psi", liquid_temperature="90 K")
    s.add("F1", "fitting", kind="elbow90", diameter="19 mm")
    s.add("V1", "valve", cv=6)
    s.add("INJ", "orifice", cda="60 mm2")
    s.add("AMB", "boundary", fluid="oxygen")
    for a, b in [("TK.liquid", "F1.in"), ("F1.out", "V1.in"), ("V1.out", "INJ.in"), ("INJ.out", "AMB.port")]:
        s.connect(a, b)
    return s


@pytest.mark.slow
def test_feed_line_starts_full_of_liquid_at_tank_pressure():
    sim = Simulation(lox_feed(), step_budget=BUDGET)
    assert sim.temperature("V1.in") == pytest.approx(90.0, abs=0.5)
    assert sim.quality("V1.in") < 0  # compressed liquid
    assert sim.pressure("V1.in") == pytest.approx(sim.pressure("TK.liquid"), rel=0.02)
    # downstream of the closed valve the line is warm gas at ambient pressure
    assert sim.temperature("V1.out") == pytest.approx(293.15, abs=0.5)
    sim.run(0.5)
    assert sim.solver_stats()["failed_substeps"] == 0


@pytest.mark.slow
def test_lox_flows_into_warm_line_and_conserves_oxygen():
    sim = Simulation(lox_feed(), step_budget=BUDGET)
    net = sim.net
    ox = [i for i in range(net.NI.shape[0]) if net.NI[i, 0] != 1 and net.fluids.names[net.NI[i, 1]] == "oxygen"]
    sx = net.NI[ox, 3]
    m0 = sim.x[sx].sum()
    out = 0.0
    sim.open("V1")
    for _ in range(1500):
        sim.step(1)
        out += sim.mass_flow("INJ") * sim.dt
    m1 = sim.x[sx].sum()
    assert m1 + out == pytest.approx(m0, rel=5e-3)
    # steady LOX flow: single-phase injector, Bernoulli with the line pressure
    p_inj = sim.pressure("INJ.in")
    assert 1.0 < sim.mass_flow("INJ") < 2.0
    assert 2e5 < p_inj < 6e5
    assert sim.quality("INJ.in") < 0
    assert sim.solver_stats()["forced_steps"] < 30
