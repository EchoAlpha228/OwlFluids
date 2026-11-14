"""Sensors are parts of their own: put on any line, they read what the fluid does there."""
import math

import pytest

from owlfluids.core.lines import Lines
from owlfluids.examples import lox_feed
from owlfluids.io.system import System
from owlfluids.sim.simulation import Simulation


def _branches(**sense) -> System:
    """A GN2 bottle through a valve into a tee; one branch through a second tee to a vent orifice,
    the other straight to a second, larger vent orifice:

        K -- V -- TEE1 -- TEE2 -- O1 -- A1
                    \\
                     O2 -- A2
    """
    s = System("branches")
    s.settings["sensor_noise"] = "off"
    s.add("K", "tank", fluid="nitrogen", volume="20 L", pressure="100 bar", temperature="300 K")
    s.add("V", "valve", cv=0.5)
    s.add("TEE1", "tee")
    s.add("TEE2", "tee")
    s.add("O1", "orifice", cda="3 mm2")
    s.add("O2", "orifice", cda="6 mm2")
    s.add("A1", "boundary", fluid="nitrogen")
    s.add("A2", "boundary", fluid="nitrogen")
    for a, b in [("K.top", "V.in"), ("V.out", "TEE1.a"), ("TEE1.b", "TEE2.a"), ("TEE2.b", "O1.in"),
                 ("O1.out", "A1.port"), ("TEE1.c", "O2.in"), ("O2.out", "A2.port")]:
        s.connect(a, b)
    for sid, (stype, line) in sense.items():
        s.sense(sid, stype, line=line)
    return s


def test_flow_meters_read_the_flow_along_their_line_wherever_they_sit():
    s = _branches(F_V=("mass_flow", ("K.top", "V.in")),  # out of the bottle, into the valve
                  F_TT=("mass_flow", ("TEE1.b", "TEE2.a")),  # between two tees: the branch to O1
                  F_O2=("mass_flow", ("TEE1.c", "O2.in")),
                  F_BACK=("mass_flow", ("TEE2.a", "TEE1.b")),  # the same line, read the other way
                  P_TT=("pressure", ("TEE1.b", "TEE2.a")),
                  P_V=("pressure", ("V.out", "TEE1.a")))
    sim = Simulation(s, step_budget=200)
    sim.open("V")
    sim.run(0.5)
    o1, o2, v = sim.mass_flow("O1"), sim.mass_flow("O2"), sim.mass_flow("V")
    assert o1 > 0.01 and o2 > 1.5 * o1
    # every line carries its own flow; with the bottle blowing down slowly, it is what the
    # elements at its ends pass, less what the line's ends take up as the pressure falls
    assert sim.read("F_V") == sim.mass_flow("K.top -- V.in") == pytest.approx(v, rel=0.01)
    assert sim.read("F_TT") == pytest.approx(o1, rel=0.01)  # no flow element at either end, still read
    assert sim.read("F_O2") == pytest.approx(o2, rel=0.01)
    assert sim.read("F_BACK") == -sim.read("F_TT")
    assert sim.read("F_V") == pytest.approx(sim.read("F_TT") + sim.read("F_O2"), rel=0.01)  # mass balance
    # lines have no friction: the pressure at their two ends differs only while they get flowing
    assert sim.read("P_TT") == sim.pressure("TEE1.a")  # reads the end it sits at (the first, with no place)
    assert sim.read("P_V") == sim.pressure("V.out")
    assert sim.read("P_TT") == pytest.approx(sim.read("P_V"), rel=0.01)
    assert not any("reads nothing" in w for w in sim.net.warnings)


def test_a_line_in_a_loop_has_a_flow_of_its_own():
    s = _branches(F_LOOP=("mass_flow", ("TEE1.b", "TEE2.a")), F_OTHER=("mass_flow", ("TEE1.d", "TEE2.c")))
    s.connect("TEE1.d", "TEE2.c")  # a second line between the tees: the two share the flow
    sim = Simulation(s, step_budget=200)
    sim.open("V")
    sim.run(0.5)
    a, b = sim.read("F_LOOP"), sim.read("F_OTHER")
    assert a > 0 and b > 0 and a + b == pytest.approx(sim.mass_flow("O1"), rel=0.01)
    assert a == pytest.approx(b, rel=0.01)  # two lines alike take half each


def test_a_line_of_length_zero_reads_the_element_at_its_end():
    s = _branches(F_O1=("mass_flow", ("O1.out", "A1.port")), F_IN=("mass_flow", ("V.in", "K.top")))
    s.line_specs["O1.out -- A1.port"] = {"length": 0}
    s.line_specs["K.top -- V.in"] = {"length": "0 m"}
    sim = Simulation(s, step_budget=200)
    sim.open("V")
    sim.run(0.2)
    assert "O1.out -- A1.port" not in sim.net.link_names
    assert sim.read("F_O1") == sim.mass_flow("O1") > 0
    assert sim.read("F_IN") == -sim.mass_flow("V")  # read from the valve's inlet toward the bottle


def test_a_sensor_on_no_line_reads_nothing():
    s = _branches()
    s.sense("PT-LOOSE", "pressure")
    s.sensors[-1].pos = (100.0, 200.0)
    sim = Simulation(s, step_budget=200)
    assert math.isnan(sim.read("PT-LOOSE"))
    assert any("PT-LOOSE reads nothing" in w for w in sim.net.warnings)
    again = System.from_dict(s.to_dict())
    assert again.sensors[-1].pos == (100.0, 200.0) and again.sensors[-1].line is None


def test_default_flow_direction():
    s = _branches()
    lines = Lines(s)
    assert lines.flow_order("V.in", "K.top") == ("K.top", "V.in")  # into a valve's inlet
    assert lines.flow_order("TEE1.a", "V.out") == ("V.out", "TEE1.a")  # on from its outlet
    assert lines.flow_order("A1.port", "O1.out") == ("O1.out", "A1.port")
    assert lines.flow_order("TEE2.a", "TEE1.b") == ("TEE2.a", "TEE1.b")  # between tees: as drawn


def test_old_files_read_the_same_once_their_sensors_are_on_lines():
    """Files from before sensors were parts of their own had them at a part's port (or on a valve);
    they load onto the line drawn there and read the same."""
    old = lox_feed().to_dict()
    old["sensors"] = [{"id": "PT-TK", "type": "pressure", "at": "TK.top"},
                      {"id": "PT-INJ", "type": "pressure", "at": "INJ.in"},
                      {"id": "FM-LOX", "type": "mass_flow", "at": "V-M"},
                      {"id": "FM-OUT", "type": "mass_flow", "at": "TK.bottom"},
                      {"id": "FM-IN", "type": "mass_flow", "at": "INJ.in"},
                      {"id": "LL-LOX", "type": "level", "at": "TK"}]
    old["settings"] = {"sensor_noise": "off"}
    s = System.from_dict(old)
    assert [x.line for x in s.sensors] == [("V-P.out", "TK.top"), ("V-M.out", "INJ.in"), ("V-M.out", "INJ.in"),
                                           ("TK.bottom", "F1.in"), ("V-M.out", "INJ.in"), None]
    assert s.sensors[-1].at == "TK" and any("sensors sit on lines now" in n for n in s.notes)
    sim = Simulation(s, step_budget=200)
    sim.open("V-M")
    sim.run(0.5)
    v = sim.mass_flow("V-M")
    assert v > 0.5
    # the meters read the lines, which carry what the valve passes, less what they take up
    assert sim.read("FM-LOX") == pytest.approx(v, rel=0.01) and sim.read("FM-OUT") == pytest.approx(v, rel=0.05)
    assert sim.read("FM-IN") == pytest.approx(v, rel=0.01)
    assert sim.read("PT-INJ") == sim.pressure("INJ.in") and sim.read("PT-TK") == sim.pressure("TK.top")
    again = System.from_dict(s.to_dict())
    assert not again.notes and [x.line for x in again.sensors] == [x.line for x in s.sensors]


def test_renaming_a_part_keeps_its_line_sensors():
    s = System.from_dict(_branches(F=("mass_flow", ("TEE1.c", "O2.in"))).to_dict())
    s._fix_refs(lambda r: r.replace("O2.", "INJ."))
    assert s.sensors[0].line == ("TEE1.c", "INJ.in")
