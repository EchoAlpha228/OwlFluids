"""Regression checks from the fluids validation batch (expected values from hand
calculations, NIST data or published flow models; see each test)."""
import pytest

from owlfluids.io.system import System
from owlfluids.sim.simulation import Simulation

PSI = 6894.757
ATM = 101325.0


def _dead_headed(fluid, t, vol):
    s = System("deadhead")
    s.add("SUP", "boundary", fluid=fluid, pressure="500 psig", temperature=f"{t} K")
    s.add("TUBE", "fitting", k=10, diameter="10.2 mm")
    s.add("L", "volume", volume=vol, fluid=fluid, pressure="500 psig", temperature=f"{t} K")
    s.add("V", "valve", diameter="5 mm", cd=0.9, actuator="solenoid", open_time="5 ms", close_time="5 ms")
    s.add("A", "boundary", fluid=fluid, temperature=f"{t} K")
    s.connect("SUP.port", "TUBE.in")
    s.connect("TUBE.out", "L.a")
    s.connect("L.b", "V.in")
    s.connect("V.out", "A.port")
    return Simulation(s)


@pytest.mark.parametrize("fluid,t,vol", [("water", 293.15, "0.4 L"), ("water", 293.15, "5 L"),
                                         ("oxygen", 90.0, "0.4 L"), ("ethanol", 293.15, "0.4 L")])
def test_dead_headed_liquid_line_equalizes(fluid, t, vol):
    """A liquid volume behind a shut valve comes up to its supply within milliseconds.

    A stiff liquid needs only micrograms to move tens of psi, so it settles in well
    under a step; it used to sit 70 psi low 50 ms after the close, with a phantom flow.
    """
    sim = _dead_headed(fluid, t, vol)
    sim.open("V")
    sim.run(0.5)
    sim.close("V")
    sim.run(0.05)
    assert (sim.pressure("L") - ATM) / PSI == pytest.approx(500.0, abs=1.0)
    sim.run(0.45)
    assert abs(sim.mass_flow("TUBE")) < 1e-4
    assert (sim.pressure("L") - ATM) / PSI == pytest.approx(500.0, abs=0.05)
