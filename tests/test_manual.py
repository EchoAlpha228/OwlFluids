"""Manual valves and regulator knobs: worked by the operator, never by Synnax."""
import os

import pytest

from owlfluids import units
from owlfluids.components import get_type
from owlfluids.drivers.synnax import SynnaxConfig
from owlfluids.io.system import System
from owlfluids.sim.realtime import Runner
from owlfluids.sim.simulation import Simulation


def hand_vent(**valve):
    s = System("hand vent")
    s.add("TK", "tank", fluid="nitrogen", volume="10 L", pressure="1000 psi", temperature="300 K")
    s.add("V1", "valve", cv=0.2)
    s.add("MV1", "manual_valve", cv=0.2, **valve)
    s.add("AMB", "boundary", fluid="nitrogen", pressure="1 bar")
    s.connect("TK.port", "V1.in")
    s.connect("V1.out", "MV1.in")
    s.connect("MV1.out", "AMB.port")
    return s


def regulated():
    s = System("regulated")
    s.add("TK", "tank", fluid="nitrogen", volume="50 L", pressure="3000 psi", temperature="300 K")
    s.add("REG", "regulator", cv=0.5, setpoint="300 psig", droop="20 psi")
    s.add("VOL", "volume", volume="1 L")
    s.add("O1", "orifice", cda="0.5 mm2")
    s.add("AMB", "boundary", fluid="nitrogen", pressure="1 bar")
    s.connect("TK.port", "REG.in")
    s.connect("REG.out", "VOL.a")
    s.connect("VOL.b", "O1.in")
    s.connect("O1.out", "AMB.port")
    s.sense("PT-VOL", "pressure", "VOL.a", noise=0.0, lag=0.0)
    return s


def test_manual_valve_throttles_to_any_position():
    sim = Simulation(hand_vent(open_time="0.5 s"))
    assert sim.manual_valves() == ["MV1"]
    assert sim.automated_valves() == ["V1"]
    sim.open("V1")
    sim.command("MV1", 0.4)
    sim.run(1.0)
    assert sim.valve_position("MV1") == pytest.approx(0.4)
    assert sim.valve_position("V1") == pytest.approx(1.0)


def test_manual_valve_strokes_at_hand_speed():
    sim = Simulation(hand_vent())  # 1 s to turn fully open by default
    sim.command("MV1", 1.0)
    sim.run(0.5)
    assert 0.3 < sim.valve_position("MV1") < 0.7


def test_old_throttle_valves_load_as_manual_valves():
    s = System.from_dict({"name": "old", "components": [
        {"id": "TV1", "type": "throttle_valve", "cv": 0.5, "position": 0.3},
        {"id": "NV1", "type": "needle_valve", "cv": 0.1},
    ]})
    tv = s.component("TV1")
    assert tv.type == "manual_valve"
    assert tv.params["open_time"] == "2 s"  # throttle valves stroked in 2 s by default
    assert tv.params["position"] == 0.3
    assert any("TV1" in n and "manual" in n for n in s.notes)
    assert s.to_dict()["components"][0]["type"] == "manual_valve"
    nv = s.component("NV1")
    assert nv.type == "manual_valve" and nv.params["open_time"] == "2 s"  # an alias of throttle_valve before
    with pytest.raises(ValueError):
        get_type("throttle_valve")  # only files are migrated; the type itself is gone


def test_synnax_never_sees_manual_valves():
    sim = Simulation(hand_vent())
    for mode in ("hootl", "hitl"):
        _, valves = SynnaxConfig(mode=mode).resolve(sim)
        assert list(valves) == ["V1"]
    with pytest.raises(ValueError, match="manual valve"):
        SynnaxConfig(valves={"MV1": {}}).resolve(sim)


def test_regulator_setpoint_turns_during_a_run_and_resets():
    sim = Simulation(regulated())
    assert sim.regulators() == ["REG"]
    sim.run(3.0)
    low = sim.read("PT-VOL")
    assert units.convert(low, "psig") == pytest.approx(300, abs=25)
    sim.set_setpoint("REG", "500 psig")
    assert sim.setpoint("REG") == pytest.approx(units.parse("500 psig"))
    sim.run(3.0)
    assert units.convert(sim.read("PT-VOL"), "psig") == pytest.approx(500, abs=25)
    sim.reset()
    assert sim.setpoint("REG") == pytest.approx(units.parse("300 psig"))
    with pytest.raises(KeyError):
        sim.set_setpoint("TK", "100 psi")


def test_runner_publishes_and_applies_setpoints():
    runner = Runner(Simulation(regulated()), rate=0)
    assert runner.frame.setpoints["REG"] == pytest.approx(units.parse("300 psig"))
    with pytest.raises(KeyError):
        runner.set_setpoint("O1", "1 psi")
    runner.set_setpoint("REG", "150 psig")  # stopped: applied at once
    assert runner.frame.setpoints["REG"] == pytest.approx(units.parse("150 psig"))


@pytest.mark.gui
def test_valve_panel_has_sliders_for_manual_valves_and_regulator_boxes():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from owlfluids.app.panels import ValvePanel

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = ValvePanel()
    panel.set_valves(["V1", "MV1"], {"MV1": True}, ["REG"])
    assert panel.rows["V1"].slider is None
    assert panel.rows["MV1"].slider is not None
    got = []
    panel.commanded.connect(lambda n, v: got.append((n, v)))
    panel.setpoint_changed.connect(lambda n, v: got.append((n, v)))
    panel.rows["MV1"].slider.setValue(35)
    panel.regulators["REG"].box.setValue(250.0)
    assert got[0] == ("MV1", 0.35)
    assert got[1][0] == "REG" and got[1][1] == pytest.approx(units.parse("250 psig"))
    sim = Simulation(regulated())
    runner = Runner(sim, rate=0)
    panel.show_frame(runner.frame)
    assert panel.regulators["REG"].box.value() == pytest.approx(300.0)
    box = panel.regulators["REG"]
    box.box.lineEdit().textEdited.emit("31")  # typing: the frame doesn't overwrite it
    runner.set_setpoint("REG", "100 psig")
    panel.show_frame(runner.frame)
    assert box.box.value() == pytest.approx(300.0)
    box.box.editingFinished.emit()  # applied (still focused): the box follows the sim again, e.g. after a reset
    panel.show_frame(runner.frame)
    assert box.box.value() == pytest.approx(100.0)
    app.processEvents()
