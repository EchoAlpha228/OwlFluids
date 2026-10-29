"""Analytic checks of the network solver."""
import math

import pytest

from owlfluids.io.system import System
from owlfluids.sim.simulation import Simulation


def test_ideal_gas_choked_blowdown_matches_isentropic_solution(gas_blowdown):
    sim = Simulation(gas_blowdown(model="ideal"))
    fs = sim.net.fluids
    R, g = fs.par[0, 0], fs.par[0, 2]
    k = 5e-6 / 0.05 * math.sqrt(g * R * 300) * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1)))
    for t in (2.0, 5.0, 10.0):
        sim.run(t - sim.t)
        exact = 200e5 * (1 + (g - 1) / 2 * k * t) ** (-2 * g / (g - 1))
        assert sim.pressure("TK") == pytest.approx(exact, rel=2e-3)


def test_real_gas_blowdown_conserves_mass(gas_blowdown):
    sim = Simulation(gas_blowdown(p="3000 psi"))
    m0 = sim.mass("TK")
    out = 0.0
    for _ in range(200):
        sim.run(0.01)
        out += sim.mass_flow("O1") * 0.01
    # trapezoid-free estimate: the outflow integral agrees to its own discretization
    assert sim.mass("TK") + out == pytest.approx(m0, rel=2e-3)
    assert sim.pressure("TK") < 3000 * 6894.76


def test_closed_system_equalizes_and_conserves_mass_and_energy():
    s = System("equalize")
    s.add("A", "tank", fluid="nitrogen", volume="10 L", pressure="100 bar", temperature="300 K")
    s.add("B", "tank", fluid="nitrogen", volume="20 L", pressure="1 bar", temperature="300 K")
    s.add("V", "valve", cv=0.3, open_time="20 ms")
    s.connect("A.port", "V.in")
    s.connect("V.out", "B.port")
    sim = Simulation(s)
    nodes = [i for i in range(sim.net.NI.shape[0]) if sim.net.NI[i, 0] != 1]
    sx = sim.net.NI[nodes, 3]
    m0, e0 = sim.x[sx].sum(), sim.x[sx + 1].sum()
    sim.open("V")
    sim.run(20.0)
    assert sim.x[sx].sum() == pytest.approx(m0, rel=1e-9)
    assert sim.x[sx + 1].sum() == pytest.approx(e0, rel=1e-6)
    assert sim.pressure("A") == pytest.approx(sim.pressure("B"), rel=1e-3)


def test_torricelli_drain():
    s = System("drain")
    s.add("TK", "tank", fluid="water", pressurant="nitrogen", volume="1 m3", fill=0.9,
          pressure="1 atm", diameter="1 m")
    s.add("VENT", "orifice", cda="0.01 m2")
    s.add("AIR", "boundary", fluid="nitrogen")
    s.add("DR", "orifice", cda="5 cm2")
    s.add("OUT", "boundary", fluid="water")
    s.connect("TK.top", "VENT.in")
    s.connect("VENT.out", "AIR.port")
    s.connect("TK.bottom", "DR.in")
    s.connect("DR.out", "OUT.port")
    sim = Simulation(s)
    area = math.pi / 4
    h0 = 0.9 / area
    sim.run(60.0)
    rho = sim.density("TK.bottom")
    h = sim.mass("TK.bottom") / rho / area
    exact = (math.sqrt(h0) - 5e-4 / area * math.sqrt(9.80665 / 2) * 60.0) ** 2
    assert h == pytest.approx(exact, rel=2e-3)
