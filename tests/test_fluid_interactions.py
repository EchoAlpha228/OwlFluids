"""The fluid interactions common on a rocket or test stand, each against a reference.

References are CoolProp energy balances unless noted: a press is the rigid-tank
charging balance on the ullage, a vent or drain integrates d(mu) = h dm with what
leaves (saturated vapor off the top, saturated liquid off the bottom), a blowdown
is isentropic for what stays in the bottle, and a regulator is isenthalpic.
"""
import CoolProp.CoolProp as CP
import pytest

from owlfluids.io.system import System
from owlfluids.sim.simulation import Simulation

PSI = 6894.757
COOLPROP = {"oxygen": "Oxygen", "nitrogen": "Nitrogen", "nitrous": "NitrousOxide", "methane": "Methane",
            "helium": "Helium", "air": "Air"}
LIQUID_T = {"oxygen": "90 K", "methane": "112 K", "nitrogen": "77.5 K", "nitrous": "293 K",
            "water": "293 K", "ethanol": "293 K"}


def _press_and_drain(liquid, gas, **tank):
    supply = 1100 if liquid == "nitrous" else 300
    s = System("press")
    s.add("TK", "tank", fluid=liquid, volume="40 L", temperature=LIQUID_T[liquid], diameter="0.25 m", **tank)
    s.add("V", "valve", cv=1)
    s.add("SUP", "boundary", fluid=gas, pressure=f"{supply} psi", temperature="293.15 K")
    s.add("VM", "valve", cv=1.5)
    s.add("INJ", "orifice", cda="15 mm2")
    s.add("ENG", "boundary", fluid=liquid)
    s.connect("SUP.port", "V.in"), s.connect("V.out", "TK.top")
    s.connect("TK.bottom", "VM.in"), s.connect("VM.out", "INJ.in"), s.connect("INJ.out", "ENG.port")
    sim = Simulation(s)
    level, m_liq = sim._level("TK"), sim.mass("TK.bottom")
    sim.vapor0 = sim.mass("TK.top")  # the tank's own vapor, if it is saturated
    sim.open("V")
    sim.run(25.0)
    pressed = (sim.pressure("TK.top"), sim._level("TK"), sim.mass("TK.bottom"))
    sim.open("VM")
    sim.run(3.0)
    return sim, supply * PSI, level, m_liq, pressed


@pytest.mark.parametrize("gas", ["nitrogen", "helium", "air"])
@pytest.mark.parametrize("liquid", list(LIQUID_T))
@pytest.mark.parametrize("tank", [{"fill": 0.6}, {"pressure": "20 psi"}], ids=["saturated", "liquid-full"])
def test_pressing_any_liquid_with_any_gas_keeps_the_gas_out_of_the_liquid(liquid, gas, tank):
    if liquid == "nitrous" and "pressure" in tank:
        pytest.skip("nitrous at 20 psi and 20 C is a gas")
    if liquid == "nitrogen" and gas == "air" and "pressure" in tank:
        pytest.skip("an air ullage starting at LN2's 77.5 K and 20 psi would be liquid air")
    sim, p_sup, level, m_liq, (p, level_pressed, m_liq_pressed) = _press_and_drain(liquid, gas, **tank)
    assert len(sim.net.comp_nodes["TK"]) == 2  # an ullage of the gas over the liquid
    assert p == pytest.approx(p_sup, rel=0.01)
    if gas == liquid:
        # GN2 into LN2 condenses onto the liquid surface, slowly: the level barely moves
        assert m_liq < m_liq_pressed and level_pressed - level < 0.005
    elif "fill" in tank and liquid not in ("water", "ethanol"):
        # the press squeezes the tank's own vapor, and some of it condenses: no more than there was
        assert m_liq <= m_liq_pressed <= m_liq + sim.vapor0
        assert level_pressed == pytest.approx(level, abs=0.002)
    else:
        # (the liquid line below the tank takes in a little as the press compresses it)
        assert m_liq_pressed == pytest.approx(m_liq, rel=2e-5)
        assert level_pressed == pytest.approx(level, abs=2e-5)
    # the press holds the tank up while the liquid drains
    assert sim._level("TK") < level - 0.03
    assert sim.pressure("TK.top") > 0.85 * p_sup
    assert not any("counted as" in w and w.startswith("TK") for w in sim.net.warnings)


@pytest.mark.parametrize("liquid", ["oxygen", "ethanol"])
def test_a_second_gas_into_an_ullage_mixes_with_its_pressurant(liquid):
    """GHe pressing a tank padded with GN2 presses it as GHe would, not as more GN2."""
    pad = _press_and_drain(liquid, "helium", fill=0.6, pressurant="nitrogen", pressure="20 psi")
    pure = _press_and_drain(liquid, "helium", fill=0.6, pressurant="helium", pressure="20 psi")
    assert any("its ullage holds a mix" in w for w in pad[0].net.warnings)
    assert pad[0].pressure("TK.top") == pytest.approx(pure[0].pressure("TK.top"), rel=0.02)
    assert pad[0]._level("TK") == pytest.approx(pure[0]._level("TK"), abs=0.003)


def test_topping_a_gn2_bottle_with_helium_mixes_them():
    s = System("top")
    s.add("K", "tank", fluid="nitrogen", volume="10 L", pressure="100 psi", temperature="293.15 K")
    s.add("V", "valve", cv=0.3)
    s.add("SUP", "boundary", fluid="helium", pressure="1000 psi", temperature="293.15 K")
    s.connect("SUP.port", "V.in"), s.connect("V.out", "K.top")
    sim = Simulation(s)
    m0 = sim.mass("K")
    sim.open("V")
    sim.run(10.0)
    assert sim.pressure("K.top") == pytest.approx(1000 * PSI, rel=0.01)
    # 0.079 kg of GN2 alone at 1000 psi would need ten times its mass; helium needs far less
    assert m0 < sim.mass("K") < 3 * m0


def _drained(name, vol, m, t0, m_end, leaving, steps=300):
    """(T, p) after taking a tank from mass m at t0 to m_end, what leaves at its saturated enthalpy."""
    u_tot = m * CP.PropsSI("U", "D", m / vol, "T", t0, name)
    dm = (m - m_end) / steps
    for _ in range(steps):
        t = CP.PropsSI("T", "D", m / vol, "U", u_tot / m, name)
        u_tot -= CP.PropsSI("H", "T", t, "Q", leaving, name) * dm
        m -= dm
    return CP.PropsSI("T", "D", m / vol, "U", u_tot / m, name), CP.PropsSI("P", "D", m / vol, "U", u_tot / m, name)


@pytest.mark.parametrize("fluid,t0", [("oxygen", 120.0), ("nitrogen", 90.0), ("nitrous", 293.0), ("methane", 140.0)])
def test_venting_a_saturated_tank_boils_it_down_its_saturation_line(fluid, t0):
    s = System("vent")
    s.add("TK", "tank", fluid=fluid, volume="20 L", fill=0.5, temperature=f"{t0} K")
    s.add("V", "valve", cv=0.05)
    s.add("ATM", "boundary", fluid=fluid)
    s.connect("TK.top", "V.in"), s.connect("V.out", "ATM.port")
    sim = Simulation(s)
    assert sim.net.NI[sim.node("TK.top"), 6] < 0  # air backing in through the vent is no press: no mixing
    m0 = sim.mass("TK")
    sim.open("V")
    sim.run(30.0)
    t, p = _drained(COOLPROP[fluid], 0.020, m0, t0, sim.mass("TK"), leaving=1.0)
    assert sim.mass("TK") < m0
    # it boils fast enough to stay within a fraction of a kelvin of its saturation line
    assert sim.temperature("TK.bottom") == pytest.approx(t, abs=0.15)
    assert sim.pressure("TK.top") == pytest.approx(p, rel=0.005)


def test_a_self_pressurizing_nitrous_tank_cools_as_it_drains_liquid():
    s = System("drain")
    s.add("TK", "tank", fluid="nitrous", volume="10 L", fill=0.8, temperature="293 K")
    s.add("V", "valve", cv=0.1)
    s.add("ATM", "boundary", fluid="nitrous")
    s.connect("TK.bottom", "V.in"), s.connect("V.out", "ATM.port")
    sim = Simulation(s)
    m0 = sim.mass("TK")
    sim.open("V")
    sim.run(5.0)
    t, p = _drained("NitrousOxide", 0.010, m0, 293.0, sim.mass("TK"), leaving=0.0)
    assert sim.mass("TK") < 0.97 * m0
    assert sim.temperature("TK.bottom") == pytest.approx(t, abs=0.1)
    assert sim.pressure("TK.top") == pytest.approx(p, rel=0.01)


@pytest.mark.parametrize("gas", ["nitrogen", "helium", "air"])
def test_a_gas_bottle_blowdown_is_isentropic_for_what_stays(gas):
    s = System("blowdown")
    s.add("K", "tank", fluid=gas, volume="10 L", pressure="200 bar", temperature="300 K")
    s.add("O", "orifice", cda="2 mm2")
    s.add("ATM", "boundary", fluid=gas)
    s.connect("K.top", "O.in"), s.connect("O.out", "ATM.port")
    sim = Simulation(s)
    sim.run(3.0)
    s0 = CP.PropsSI("S", "P", 200e5, "T", 300.0, COOLPROP[gas])
    t = CP.PropsSI("T", "P", sim.pressure("K.top"), "S", s0, COOLPROP[gas])
    assert sim.pressure("K.top") < 180e5
    assert sim.temperature("K.top") == pytest.approx(t, rel=0.005)


def test_a_gn2_regulator_cools_the_gas_as_a_throttle_does():
    s = System("reg")
    s.add("SUP", "boundary", fluid="nitrogen", pressure="4000 psi", temperature="293.15 K")
    s.add("REG", "regulator", cv=0.2, setpoint="300 psi")
    s.add("M", "volume", volume="1 L")
    s.add("O", "orifice", cda="3 mm2")
    s.add("ATM", "boundary", fluid="nitrogen")
    s.connect("SUP.port", "REG.in"), s.connect("REG.out", "M.a"), s.connect("M.b", "O.in")
    s.connect("O.out", "ATM.port")
    sim = Simulation(s)
    sim.run(20.0)
    h = CP.PropsSI("H", "P", 4000 * PSI, "T", 293.15, "Nitrogen")
    assert sim.temperature("M.a") == pytest.approx(CP.PropsSI("T", "P", sim.pressure("M.a"), "H", h, "Nitrogen"),
                                                   abs=1.0)  # about 33 K of Joule-Thomson cooling


def test_ln2_transfers_from_a_dewar_to_a_run_tank():
    s = System("transfer")
    s.add("DW", "tank", fluid="nitrogen", volume="100 L", fill=0.8, pressure="50 psi")
    s.add("RT", "tank", fluid="nitrogen", volume="20 L", fill=0.02, pressure="15 psi")
    s.add("V", "valve", cv=0.5)
    s.connect("DW.bottom", "V.in"), s.connect("V.out", "RT.top")
    sim = Simulation(s)
    assert not any("pushed into" in w for w in sim.net.warnings)  # liquid from a dewar is not a press
    m0 = sim.total_mass()
    sim.open("V")
    sim.run(20.0)
    assert sim.total_mass() == pytest.approx(m0, rel=1e-6)
    assert sim._level("RT") > 0.05


def test_ethanol_transfers_into_a_vented_empty_tank():
    s = System("fill")
    s.add("SUP", "tank", fluid="ethanol", pressurant="nitrogen", volume="60 L", fill=0.8, pressure="100 psi")
    s.add("RT", "tank", fluid="ethanol", pressurant="nitrogen", volume="20 L", fill=0.0, pressure="1 atm")
    s.add("V", "valve", cv=0.5)
    s.add("VV", "valve", cv=1)
    s.add("ATM", "boundary", fluid="nitrogen")
    s.connect("SUP.bottom", "V.in"), s.connect("V.out", "RT.bottom")
    s.connect("RT.top", "VV.in"), s.connect("VV.out", "ATM.port")
    sim = Simulation(s)
    m0 = sim.mass("SUP.bottom") + sim.mass("RT.bottom")
    sim.open("V", "VV")
    sim.run(15.0)
    assert sim.mass("SUP.bottom") + sim.mass("RT.bottom") == pytest.approx(m0, rel=1e-6)
    assert sim._level("RT") > 0.1
    assert sim.pressure("RT.top") == pytest.approx(101325.0, rel=0.05)


@pytest.mark.parametrize("liquid,gas,t", [("oxygen", "nitrogen", "90 K"), ("ethanol", "helium", "293 K"),
                                          ("nitrous", "nitrogen", "293 K"), ("methane", "helium", "112 K")])
def test_a_purge_clears_the_line_and_leaves_the_tank_alone(liquid, gas, t):
    s = System("purge")
    s.add("TK", "tank", fluid=liquid, pressurant="nitrogen", volume="40 L", fill=0.8, temperature=t,
          pressure="900 psi" if liquid == "nitrous" else "300 psi")
    s.add("VM", "valve", cv=1)
    s.add("J", "volume", volume="0.5 L")
    s.add("CV", "check_valve", cv=0.5)
    s.add("VP", "valve", cv=0.3)
    s.add("PG", "boundary", fluid=gas, pressure="1200 psi" if liquid == "nitrous" else "400 psi")
    s.add("INJ", "orifice", cda="10 mm2")
    s.add("ENG", "boundary", fluid=liquid)
    s.connect("TK.bottom", "VM.in"), s.connect("VM.out", "J.a")
    s.connect("PG.port", "VP.in"), s.connect("VP.out", "CV.in"), s.connect("CV.out", "J.b")
    s.connect("J.c", "INJ.in"), s.connect("INJ.out", "ENG.port")
    s.sense("PF", "purge_fraction", "J.a")
    sim = Simulation(s)
    sim.open("VM")
    sim.run(2.0)
    sim.close("VM")
    sim.run(3.0)  # the line to the shut valve settles: a two-phase pocket there bounces on the column
    m_liq = sim.mass("TK.bottom")
    sim.open("VP")
    sim.run(4.0)  # (the purge pushes the line to the injector out too)
    assert sim.read("PF") > 0.9
    assert sim.mass("TK.bottom") == pytest.approx(m_liq, rel=1e-5)


def test_filling_an_unvented_air_tank_compresses_the_air_adiabatically():
    """Liquid filled into a closed tank of air squeezes the air: p V^k and T V^(k-1) hold (k = 1.4)."""
    s = System("closed fill")
    s.settings["fluid_models"] = {"air": "ideal"}  # the textbook's ideal gas
    s.add("SUP", "boundary", fluid="ethanol", pressure="100 psi")
    s.add("RT", "tank", fluid="air", volume="20 L", pressure="1 atm", temperature="293.15 K",
          ullage_heat_coefficient=0)  # the textbook's adiabatic limit
    s.add("V", "valve", cv=0.5)
    s.connect("SUP.port", "V.in"), s.connect("V.out", "RT.bottom")
    sim = Simulation(s)
    assert any("ethanol collecting under the air" in w for w in sim.net.warnings)
    sim.open("V")
    sim.run(90.0)
    ratio = 100 * PSI / 101325.0
    assert sim.pressure("RT.bottom") == pytest.approx(100 * PSI, rel=1e-3)  # the supply holds the bottom
    assert sim.pressure("RT.top") < sim.pressure("RT.bottom")  # less the ethanol's head
    assert sim._level("RT") == pytest.approx(1.0 - ratio ** (-1 / 1.4), abs=0.002)  # 74.6 %, then it stops
    assert sim.temperature("RT.top") == pytest.approx(293.15 * ratio ** (0.4 / 1.4), rel=0.005)


def test_lox_filled_into_a_gn2_tank_collects_as_liquid():
    s = System("lox fill")
    s.add("SUP", "tank", fluid="oxygen", pressurant="nitrogen", volume="100 L", fill=0.8, pressure="100 psi",
          temperature="90 K")
    s.add("RT", "tank", fluid="nitrogen", volume="20 L", pressure="1 atm", temperature="293.15 K")
    s.add("V", "valve", cv=0.5)
    s.connect("SUP.bottom", "V.in"), s.connect("V.out", "RT.bottom")
    sim = Simulation(s)
    m0 = sim.mass("SUP.bottom") + sim.line_mass()
    sim.open("V")
    sim.run(16.0)
    # all of it is LOX (with what the line between holds)
    assert sim.mass("RT.bottom") == pytest.approx(m0 - sim.mass("SUP.bottom") - sim.line_mass(), rel=1e-6)
    assert sim._level("RT") > 0.15
    # with no vent, the GN2 above it is squeezed, though the LOX cools it
    assert sim.pressure("RT.top") > 1.1 * 101325.0
