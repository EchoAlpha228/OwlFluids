"""Pressing a tank with gas, against worked examples.

The worked examples are the textbook unsteady-flow energy balance for a rigid,
insulated tank charged from a supply line (for example Cengel & Boles,
*Thermodynamics: An Engineering Approach*, ch. 5, "charging of a rigid tank"):

    m2 u2 = m1 u1 + (m2 - m1) h_line

For an ideal gas filling an evacuated tank this gives the classic T2 = k T_line
(air from a 300 K line ends at 420 K). Pressing the ullage of a propellant tank is
the same balance with the ullage as the tank: with no heat exchange with the liquid
it is the "ideal adiabatic" pressurization limit, the least pressurant a press needs.
The liquid takes no part, so its mass and level must not change.
"""
import CoolProp.CoolProp as CP
import pytest
from scipy.optimize import brentq

from owlfluids.core.network import build_network
from owlfluids.io.system import System
from owlfluids.sim.simulation import Simulation

PSI = 6894.757


def _charged(p1, t1, vol, p_line, t_line):
    """Mass and temperature after charging a rigid, adiabatic tank of nitrogen to p_line (CoolProp)."""
    u1 = CP.PropsSI("U", "P", p1, "T", t1, "Nitrogen")
    m1 = CP.PropsSI("D", "P", p1, "T", t1, "Nitrogen") * vol
    h = CP.PropsSI("H", "P", p_line, "T", t_line, "Nitrogen")

    def res(m2):
        u2 = (m1 * u1 + (m2 - m1) * h) / m2
        return CP.PropsSI("P", "D", m2 / vol, "U", u2, "Nitrogen") - p_line

    m2 = brentq(res, m1 * 1.0001, CP.PropsSI("D", "P", p_line, "T", 100.0, "Nitrogen") * vol)
    u2 = (m1 * u1 + (m2 - m1) * h) / m2
    return m2, CP.PropsSI("T", "D", m2 / vol, "U", u2, "Nitrogen")


def _ideal_t2(p1, t1, p2, t_line, k=1.4):
    """Ideal gas, constant k: T2 = p2 / (p1 / T1 + (p2 - p1) / (k T_line)); T2 = k T_line from vacuum."""
    return p2 / (p1 / t1 + (p2 - p1) / (k * t_line))


def test_charging_an_evacuated_tank_ends_at_k_times_the_line_temperature():
    s = System("charge")
    s.add("TK", "volume", fluid="nitrogen", volume="10 L", pressure="0.02 bar", temperature="300 K")
    s.add("O", "orifice", cda="10 mm2")
    s.add("LINE", "boundary", fluid="nitrogen", pressure="3 bar", temperature="300 K")
    s.connect("LINE.port", "O.in"), s.connect("O.out", "TK.a")
    sim = Simulation(s)
    sim.run(8.0)
    assert sim.pressure("TK.a") == pytest.approx(3e5, rel=1e-3)
    m2, t2 = _charged(0.02e5, 300.0, 0.010, 3e5, 300.0)
    assert sim.mass("TK") == pytest.approx(m2, rel=0.01)
    assert sim.temperature("TK.a") == pytest.approx(t2, rel=0.01)
    # the textbook ideal-gas answer, near k T_line = 420 K
    assert sim.temperature("TK.a") == pytest.approx(_ideal_t2(0.02e5, 300.0, 3e5, 300.0), rel=0.015)


def _lox_press(**tank):
    s = System("press")
    s.add("TK", "tank", fluid="oxygen", volume="60 L", fill=0.5, temperature="90 K", **tank)
    s.add("V", "valve", cv=1)
    s.add("GN2", "boundary", fluid="nitrogen", pressure="300 psi", temperature="293.15 K")
    s.add("O", "orifice", cda="20 mm2")
    s.add("ENG", "boundary", fluid="oxygen")
    s.add("V-M", "valve", cv=2)
    s.connect("GN2.port", "V.in"), s.connect("V.out", "TK.top")
    s.connect("TK.bottom", "V-M.in"), s.connect("V-M.out", "O.in"), s.connect("O.out", "ENG.port")
    s.sense("LL", "level", "TK")
    return s


@pytest.mark.parametrize("tank", [{}, {"pressurant": "nitrogen", "pressure": "15.5 psi",
                                       "gas_temperature": "90 K"}],
                         ids=["saturated LOX tank", "LOX under a nitrogen pressurant"])
def test_pressing_lox_with_gn2_keeps_the_liquid_and_matches_adiabatic_ullage_compression(tank):
    sim = Simulation(_lox_press(**tank))
    nodes = sim.net.comp_nodes["TK"]
    assert [sim.net.node_names[i] for i in nodes] == ["TK.ullage", "TK.liquid"]
    p1, t1 = sim.pressure("TK.top"), sim.temperature("TK.top")
    m_liq, level = sim.mass("TK.bottom"), sim._level("TK")
    v_ull = 0.060 * 0.5
    m_g1 = sim.mass("TK.top")
    sim.open("V")
    sim.run(30.0)
    # GN2 raised the pressure, not the level: none of it became liquid oxygen
    assert sim.pressure("TK.top") == pytest.approx(300 * PSI, rel=2e-3)
    assert sim.mass("TK.bottom") == pytest.approx(m_liq, rel=1e-6)
    assert sim._level("TK") == pytest.approx(level, abs=1e-6)
    # pressurant taken and ullage temperature: the rigid-tank energy balance on the ullage
    m2, t2 = _charged(p1, t1, v_ull, 300 * PSI, 293.15)
    assert sim.mass("TK.top") - m_g1 == pytest.approx(m2 - CP.PropsSI("D", "P", p1, "T", t1, "Nitrogen") * v_ull,
                                                     rel=0.02)
    assert sim.temperature("TK.top") == pytest.approx(t2, rel=0.01)
    assert sim.temperature("TK.top") == pytest.approx(_ideal_t2(p1, t1, 300 * PSI, 293.15), rel=0.03)
    # then the press holds the tank at 300 psi as it drains, and the level falls
    sim.open("V-M")
    sim.run(3.0)
    assert sim._level("TK") < level - 0.02
    assert sim.pressure("TK.top") > 250 * PSI


def test_a_saturated_tank_gets_an_ullage_only_from_a_gas_that_can_press_it():
    s = _lox_press()
    net = build_network(s)
    assert len(net.comp_nodes["TK"]) == 2
    assert any("TK: nitrogen can be pushed into its oxygen" in w for w in net.warnings)
    # an air vent at 1 atm is not a press: the tank stays saturated, so it still boils off
    v = System("vented")
    v.add("TK", "tank", fluid="oxygen", volume="60 L", fill=0.5, temperature="90 K")
    v.add("V", "valve", cv=0.2, normally="open")
    v.add("ATM", "boundary", fluid="air")
    v.connect("TK.top", "V.in"), v.connect("V.out", "ATM.port")
    assert len(build_network(v).comp_nodes["TK"]) == 1
    # nor is GN2 into an LN2 tank: there the gas is the tank's own fluid, and condenses
    n = System("ln2")
    n.add("TK", "tank", fluid="nitrogen", volume="60 L", fill=0.5, temperature="77 K")
    n.add("V", "valve", cv=0.2)
    n.add("GN2", "boundary", fluid="nitrogen", pressure="100 psi")
    n.connect("GN2.port", "V.in"), n.connect("V.out", "TK.top")
    assert len(build_network(n).comp_nodes["TK"]) == 1
