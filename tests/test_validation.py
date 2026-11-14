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


@pytest.mark.parametrize("psia", [2400, 4500])
def test_helium_bottle_holds_its_real_mass(psia):
    """A 49 L helium bottle: helium is 8 % denser than ideal at 2,400 psia and 15 % at 4,500 (NIST)."""
    import CoolProp.CoolProp as CP

    s = System("bottle")
    s.add("K", "tank", fluid="helium", volume="49 L", pressure=f"{psia} psi", temperature="293.15 K")
    s.add("O", "orifice", cda="1 mm2")
    s.add("A", "boundary", fluid="helium")
    s.connect("K.top", "O.in")
    s.connect("O.out", "A.port")
    sim = Simulation(s)
    expected = CP.PropsSI("D", "P", psia * PSI, "T", 293.15, "Helium") * 0.049
    assert sim.mass("K") == pytest.approx(expected, rel=0.005)


def test_helium_warms_through_a_regulator():
    """Helium is above its inversion temperature: 3,000 to 500 psig isenthalpic warms it 10.6 K."""
    import CoolProp.CoolProp as CP

    s = System("reg")
    s.add("SUP", "boundary", fluid="helium", pressure="3000 psi", temperature="293.15 K")
    s.add("R", "regulator", cv=0.2, setpoint="500 psig", droop_on="off")
    s.add("V", "volume", volume="1 L", fluid="helium", pressure="1 atm", temperature="293.15 K")
    s.add("O", "orifice", cda="2 mm2")
    s.add("A", "boundary", fluid="helium")
    s.connect("SUP.port", "R.in")
    s.connect("R.out", "V.a")
    s.connect("V.b", "O.in")
    s.connect("O.out", "A.port")
    sim = Simulation(s)
    sim.run(10.0)
    h = CP.PropsSI("H", "P", 3000 * PSI, "T", 293.15, "Helium")
    t_jt = CP.PropsSI("T", "P", sim.pressure("V"), "H", h, "Helium")
    assert t_jt - 293.15 > 10.0
    assert sim.temperature("V") == pytest.approx(t_jt, abs=1.0)


def _dyer_hem(cpn, t, p_up, p_dn=ATM):
    """Dyer's blend of incompressible flow and the true (isentropic, choked) HEM flux, kg/s/m2."""
    import math

    import CoolProp.CoolProp as CP
    import numpy as np

    psat = CP.PropsSI("P", "T", t, "Q", 0, cpn)
    rho = CP.PropsSI("D", "T", t, "Q", 0, cpn)
    h1, s1 = CP.PropsSI("H", "T", t, "Q", 0, cpn), CP.PropsSI("S", "T", t, "Q", 0, cpn)
    g_hem = 0.0
    for p in np.linspace(p_dn * 1.5, psat * 0.999, 300):
        h, d = CP.PropsSI("H", "P", p, "S", s1, cpn), CP.PropsSI("D", "P", p, "S", s1, cpn)
        g_hem = max(g_hem, d * math.sqrt(max(2 * (h1 - h), 0.0)))
    g_hem = math.sqrt(g_hem ** 2 + 2 * rho * (p_up - psat))
    g_spi = math.sqrt(2 * rho * (p_up - p_dn))
    k = math.sqrt((p_up - p_dn) / (psat - p_dn))
    return k / (1 + k) * g_spi + 1 / (1 + k) * g_hem


@pytest.mark.parametrize("fluid,cpn,t", [("nitrous", "NitrousOxide", 293.15), ("nitrous", "NitrousOxide", 273.15),
                                         ("oxygen", "Oxygen", 110.0), ("nitrogen", "Nitrogen", 90.0)])
def test_saturated_tank_drains_at_dyer_flux(fluid, cpn, t):
    """Saturated liquid flashing through an orifice to the atmosphere matches Dyer's model
    with the real HEM flux (it read 17 to 20 % low for nitrous, whose HEM term was about 0)."""
    import CoolProp.CoolProp as CP

    psat = CP.PropsSI("P", "T", t, "Q", 0, cpn)
    s = System("flash")
    s.add("TK", "tank", fluid=fluid, volume="100 L", fill=0.8, pressure=f"{psat} Pa", temperature=f"{t} K",
          diameter="0.3 m")
    s.add("O", "orifice", cda="1 mm2")
    s.add("VV", "volume", volume="0.1 L", fluid=fluid, pressure="1 atm", temperature="300 K")
    s.add("O2", "orifice", cda="5000 mm2")
    s.add("DN", "boundary", fluid=fluid, pressure="1 atm", temperature="300 K")
    s.connect("TK.bottom", "O.in")
    s.connect("O.out", "VV.a")
    s.connect("VV.b", "O2.in")
    s.connect("O2.out", "DN.port")
    sim = Simulation(s)
    sim.run(0.3)
    expected = _dyer_hem(cpn, t, sim.pressure("TK.bottom"))
    assert sim.mass_flow("O") / 1e-6 == pytest.approx(expected, rel=0.06)
