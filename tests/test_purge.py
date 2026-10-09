"""Purges: any purge gas pushes a propellant out of a line, in real time."""
import time

import numpy as np
import pytest

from owlfluids.core import layout as L
from owlfluids.core.network import BuildError, build_network
from owlfluids.examples import lox_purge
from owlfluids.fluids.registry import PURGE_GASES
from owlfluids.io.system import System
from owlfluids.sim.simulation import Simulation


def _rig(liquid="water", gas="nitrogen", vent=True):
    """Run tank feeding a line and injector, with a purge bottle teed in through a check valve."""
    s = System("purge")
    kw = {"liquid_temperature": "90 K"} if liquid == "oxygen" else {}
    s.add("TK", "pressurized_tank", liquid=liquid, pressurant="nitrogen", volume="20 L", fill=0.8,
          pressure="200 psi", **kw)
    s.add("MV", "valve", cv=1.0, open_time="100 ms")
    s.add("L1", "pipe", length="2 m", diameter="0.4 in")
    s.add("B", "tank", fluid=gas, volume="10 L", pressure="2000 psi", temperature="293 K")
    s.add("REG", "regulator", setpoint="300 psi", cv=0.5)
    s.add("PV", "valve", cv=0.3, open_time="50 ms")
    s.add("CK", "check_valve", cv=0.3)
    for a, b in [("TK.liquid", "MV.in"), ("MV.out", "L1.in"), ("B.port", "REG.in"), ("REG.out", "PV.in"),
                 ("PV.out", "CK.in"), ("CK.out", "L1.in")]:
        s.connect(a, b)
    if vent:
        s.add("INJ", "orifice", diameter="4 mm", cd=0.7)
        s.add("AMB", "boundary", fluid=liquid)
        s.connect("L1.out", "INJ.in")
        s.connect("INJ.out", "AMB.port")
    s.sense("PF", "purge_fraction", "L1.out", units="")
    s.sense("TC", "temperature", "L1.out", units="K")
    return s


def _flow_then_purge(sim, purge=2.0):
    sim.open("MV")
    sim.run(0.5)
    sim.close("MV")
    sim.run(0.3)
    sim.open("PV")
    sim.run(purge)


@pytest.mark.parametrize("gas", PURGE_GASES)
def test_every_purge_gas_clears_a_water_line(gas):
    sim = Simulation(_rig("water", gas))
    _flow_then_purge(sim)
    assert sim.read("PF") > 0.95
    assert sim.solver_stats()["forced_steps"] < 100


@pytest.mark.parametrize("liquid,gas", [("oxygen", "nitrogen"), ("ethanol", "helium"), ("methane", "argon")])
def test_purge_pushes_propellant_out(liquid, gas):
    sim = Simulation(_rig(liquid, gas))
    _flow_then_purge(sim, purge=3.0)
    assert sim.read("PF") > 0.95
    assert sim.nout[sim.node("L1.out"), L.O_RHO] < 50  # gas only: no liquid left
    if liquid == "oxygen":
        assert sim.read("TC") > 200  # the line warms once the LOX is gone
    assert sim.solver_stats()["forced_steps"] < 100


def test_purge_conserves_mass():
    """Dead-ended line: every gram the bottle loses ends up in the line."""
    sim = Simulation(_rig("water", "helium", vent=False))
    vol = [i for i in range(len(sim.net.NI)) if sim.net.NI[i, L.N_KIND] != L.NODE_BOUNDARY]
    m0 = sim.nout[vol, L.O_M].sum()
    _flow_then_purge(sim, purge=1.0)
    assert sim.nout[vol, L.O_M].sum() == pytest.approx(m0, rel=1e-6)


def test_line_between_purge_valve_and_check_starts_full_of_gas():
    sim = Simulation(_rig("oxygen", "nitrogen"))
    i = sim.node("PV.out")
    assert sim.nout[i, L.O_GAS] > 0.99
    assert sim.nout[sim.node("L1.out"), L.O_GAS] < 0.01


def test_builder_names_foreign_fluid_conversions():
    net = build_network(_rig("oxygen", "helium"))
    assert any("TK holds only oxygen" in w and "helium" in w for w in net.warnings)


def test_a_propellant_cannot_purge():
    s = _rig("methane", "nitrogen")
    s.component("B").params["fluid"] = "oxygen"
    with pytest.raises(BuildError):
        build_network(s)


@pytest.mark.slow
def test_lox_purge_example_runs_in_real_time():
    sim = Simulation(lox_purge())
    sim.run(0.2)
    times = []
    for step, dur in (("V-M", 1.0), (None, 0.0), ("V-PG", 2.0)):
        if step == "V-PG":
            sim.close("V-M")
            sim.run(0.3)
        if step:
            sim.open(step)
        for _ in range(int(round(dur / 0.01))):
            t0 = time.perf_counter()
            sim.run(0.01)
            times.append(time.perf_counter() - t0)
    t = np.array(times) / 0.01
    assert t.mean() < 0.5
    assert t.max() < 1.5
    assert sim.read("PF-INJ") > 0.95
    assert sim.read("TC-INJ") > 200
