"""Analytic checks of the network solver."""
import math

import numpy as np
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
    s.add("L", "pipe", length="1 m", diameter="6 mm")
    s.add("V", "valve", cv=0.3, open_time="20 ms")
    s.connect("A.port", "L.in")
    s.connect("L.out", "V.in")
    s.connect("V.out", "B.port")
    sim = Simulation(s)
    nodes = [i for i in range(sim.net.NI.shape[0]) if sim.net.NI[i, 0] != 1]
    sx = sim.net.NI[nodes, 3]
    m0, e0 = sim.x[sx].sum(), sim.x[sx + 1].sum()
    sim.open("V")
    sim.run(20.0)
    assert sim.x[sx].sum() == pytest.approx(m0, rel=1e-9)
    # energy: pipe kinetic energy is not tracked, everything else is conserved
    assert sim.x[sx + 1].sum() == pytest.approx(e0, rel=1e-6)
    assert sim.pressure("A") == pytest.approx(sim.pressure("B"), rel=1e-3)


def test_torricelli_drain():
    s = System("drain")
    s.add("TK", "pressurized_tank", liquid="water", pressurant="nitrogen", volume="1 m3", fill=0.9,
          pressure="1 atm", diameter="1 m")
    s.add("VENT", "orifice", cda="0.01 m2")
    s.add("AIR", "boundary", fluid="nitrogen")
    s.add("DR", "orifice", cda="5 cm2")
    s.add("OUT", "boundary", fluid="water")
    s.connect("TK.gas", "VENT.in")
    s.connect("VENT.out", "AIR.port")
    s.connect("TK.liquid", "DR.in")
    s.connect("DR.out", "OUT.port")
    sim = Simulation(s)
    area = math.pi / 4
    h0 = 0.9 / area
    sim.run(60.0)
    rho = sim.density("TK.liquid")
    h = sim.mass("TK.liquid") / rho / area
    exact = (math.sqrt(h0) - 5e-4 / area * math.sqrt(9.80665 / 2) * 60.0) ** 2
    assert h == pytest.approx(exact, rel=2e-3)


def _pipe_system(dt="1 ms"):
    s = System("pipe")
    s.settings["dt"] = dt
    s.add("SUP", "boundary", fluid="water", pressure="5 bar")
    s.add("L1", "pipe", length="10 m", diameter="10 mm", roughness="1.5 um", segments=5)
    s.add("V1", "valve", diameter="20 mm", cd=1.0, normally="open", open_time="5 ms")
    s.add("OUT", "boundary", fluid="water", pressure="1 bar")
    s.connect("SUP.port", "L1.in")
    s.connect("L1.out", "V1.in")
    s.connect("V1.out", "OUT.port")
    return s


def test_steady_pipe_flow_matches_colebrook():
    sim = Simulation(_pipe_system())
    sim.run(5.0)
    m = sim.mass_flow("L1")
    rho = 998.2
    area = math.pi * 0.01**2 / 4
    v = m / rho / area
    re = rho * v * 0.01 / sim.net.fluids.par[0, 7]
    f = 0.02
    for _ in range(50):
        f = (-2 * math.log10(1.5e-6 / 0.01 / 3.7 + 2.51 / (re * math.sqrt(f)))) ** -2
    dp = f * 10 / 0.01 * rho * v * v / 2 + (m / (area * 4)) ** 2 / (2 * rho)
    assert dp == pytest.approx(4e5, rel=0.02)


def test_water_hammer_follows_joukowsky():
    s = System("hammer")
    s.settings["dt"] = "0.1 ms"
    s.add("SUP", "boundary", fluid="water", pressure="10 bar")
    s.add("L1", "pipe", length="20 m", diameter="20 mm", segments=40)
    s.add("V1", "valve", cda="3.14 cm2", normally="open", open_time="1 ms")
    s.add("OUT", "boundary", fluid="water", pressure="1 bar")
    s.connect("SUP.port", "L1.in")
    s.connect("L1.out", "V1.in")
    s.connect("V1.out", "OUT.port")
    sim = Simulation(s)
    sim.run(3.0)
    rho = 998.2
    v0 = sim.mass_flow("L1.f40") / rho / (math.pi * 0.02**2 / 4)
    p0 = sim.pressure("V1.in")
    c = math.sqrt(sim.net.fluids.par[0, 1] / rho)
    sim.close("V1")
    ps = []
    for _ in range(300):
        sim.step(1)
        ps.append(sim.pressure("V1.in"))
    ps = np.array(ps)
    n_reflect = int(2 * 20 / c / 1e-4)
    plateau = ps[30:n_reflect - 20].mean() - p0
    assert plateau == pytest.approx(rho * c * v0, rel=0.05)
