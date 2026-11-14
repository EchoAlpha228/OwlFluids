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
    assert sim.read("F_V") == pytest.approx(v, rel=1e-9)
    assert sim.read("F_TT") == pytest.approx(o1, rel=1e-9)  # no flow element at either end, still read
    assert sim.read("F_O2") == pytest.approx(o2, rel=1e-9)
    assert sim.read("F_BACK") == pytest.approx(-o1, rel=1e-9)
    assert sim.read("F_V") == pytest.approx(sim.read("F_TT") + sim.read("F_O2"), rel=1e-3)  # mass balance
    # lines have no friction of their own: all the lines joining the tees are one point
    assert sim.read("P_TT") == sim.read("P_V") == sim.pressure("TEE1.a")
    assert not any("reads nothing" in w for w in sim.net.warnings)


def test_a_line_in_a_loop_has_no_flow_of_its_own():
    s = _branches(F_LOOP=("mass_flow", ("TEE1.b", "TEE2.a")), P_LOOP=("pressure", ("TEE1.b", "TEE2.a")))
    s.connect("TEE1.d", "TEE2.c")  # a second line between the tees: the two share the flow somehow
    assert Lines(s).flow_side("TEE1.b", "TEE2.a") is None
    sim = Simulation(s, step_budget=200)
    assert math.isnan(sim.read("F_LOOP")) and math.isfinite(sim.read("P_LOOP"))
    assert any("F_LOOP reads nothing" in w and "loop" in w for w in sim.net.warnings)


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
    assert sim.read("FM-LOX") == pytest.approx(v) and sim.read("FM-OUT") == pytest.approx(v, rel=0.05)
    assert sim.read("FM-IN") == pytest.approx(v)
    assert sim.read("PT-INJ") == sim.pressure("INJ.in") and sim.read("PT-TK") == sim.pressure("TK.top")
    again = System.from_dict(s.to_dict())
    assert not again.notes and [x.line for x in again.sensors] == [x.line for x in s.sensors]


def test_renaming_a_part_keeps_its_line_sensors():
    s = System.from_dict(_branches(F=("mass_flow", ("TEE1.c", "O2.in"))).to_dict())
    s._fix_refs(lambda r: r.replace("O2.", "INJ."))
    assert s.sensors[0].line == ("TEE1.c", "INJ.in")
