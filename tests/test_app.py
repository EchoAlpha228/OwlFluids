"""Desktop app smoke tests (offscreen)."""
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("pyqtgraph")

from owlfluids.app.window import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def pump(app, seconds):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


@pytest.mark.gui
def test_edit_save_and_run(app, tmp_path):
    w = MainWindow()
    w.show()
    sc = w.scene
    from PySide6.QtCore import QPointF

    sc.add_component("tank", QPointF(0, 0))
    sc.add_component("valve", QPointF(150, 0))
    sc.add_component("boundary", QPointF(300, 0))
    sc.connect_ports("K1.port", "V1.in")
    sc.connect_ports("V1.out", "ATM1.port")
    sc.system.sense("PT-K1", "pressure", "K1.port", units="psi")
    assert w.dirty
    w.path = tmp_path / "s.yaml"
    assert w.save()
    w.open_file(tmp_path / "s.yaml")
    assert [c.id for c in w.scene.system.components] == ["K1", "V1", "ATM1"]

    w.run()
    deadline = time.time() + 60
    while w.runner is None and time.time() < deadline:
        pump(app, 0.05)
    assert w.runner is not None, w.status_time.text()
    p0 = w.runner.frame.sensors["PT-K1"]
    w._toggle_valve("V1")
    pump(app, 1.0)
    assert w.runner.frame.commands["V1"] == 1.0
    assert w.runner.frame.sensors["PT-K1"] < p0
    w.stop()
    assert w.runner is None
    w.dirty = False
    w.close()


@pytest.mark.gui
def test_examples_load_and_lay_out(app):
    w = MainWindow()
    for name in ("gn2-blowdown", "lox-feed", "test-stand", "nitrous-fill"):
        w.load_example(name)
        assert len(w.scene.items_by_id) == len(w.scene.system.components)
        assert all(c.position is not None for c in w.scene.system.components)
    w.dirty = False
    w.close()


def test_near_zero_flows_show_as_zero():
    from owlfluids.app.window import _flow_text

    assert _flow_text(3e-7) == "0 kg/s" and _flow_text(-5e-5) == "0 kg/s"
    assert _flow_text(0.0123) == "0.0123 kg/s" and _flow_text(-2.5) == "-2.5 kg/s"


@pytest.mark.gui
def test_update_check_reports_through_the_gui_thread(app, monkeypatch, tmp_path):
    from owlfluids import update
    from owlfluids.app import updates

    monkeypatch.setattr(update, "CONFIG", tmp_path / "update.json")
    shown = []
    monkeypatch.setattr(updates.QMessageBox, "information", lambda *a: shown.append(a[2]))
    monkeypatch.setattr(update, "latest_release", lambda: None)
    w = MainWindow()
    assert [a.text() for a in w.menuBar().actions()][-1] == "&Help"
    w.updater.check_on_start()  # not frozen: no check
    w.updater.check()
    pump(app, 0.3)
    assert shown and "up to date" in shown[0]
    d = updates.UpdateSettings(w)
    d.token.setText(" tok ")
    d._ok()
    assert update.load_config() == {"token": "tok", "check_on_start": True}
    w.dirty = False
    w.close()


@pytest.mark.gui
def test_quiet_update_notice_does_not_interrupt_a_run(app, monkeypatch):
    from owlfluids import update
    from owlfluids.app import updates

    monkeypatch.setattr(update, "frozen", lambda: True)
    monkeypatch.setattr(updates.QMessageBox, "exec", lambda *a: pytest.fail("no dialog during a run"))
    rel = update.Release(version="99.0.0", tag="v99.0.0", url="u", asset_name="a.exe", asset_api_url="x", size=1)
    w = MainWindow()
    w.runner = object()  # a run in progress
    w.updater._on_checked(rel, True)
    assert "99.0.0" in w.statusBar().currentMessage()
    w.runner = None
    w.dirty = False
    w.close()


def test_recent_files_and_last_folder(tmp_path):
    from owlfluids.app import prefs

    files = []
    for i in range(prefs.RECENT_MAX + 2):
        f = tmp_path / f"s{i}.yaml"
        f.write_text("name: s\n")
        files.append(f)
        prefs.add_recent(f)
    prefs.add_recent(files[3])  # opening one again moves it to the top, once
    recent = prefs.recent()
    assert recent[0] == str(files[3].resolve())
    assert len(recent) == prefs.RECENT_MAX and len(set(recent)) == len(recent)
    files[3].unlink()  # a deleted file drops out of the list
    assert str(files[3].resolve()) not in prefs.recent()
    assert prefs.last_dir() == str(tmp_path.resolve())


@pytest.mark.gui
@pytest.mark.parametrize("example, a, b", [
    ("test-stand", "V-OXM.in", "TEE-OXD.a"),
    ("test-stand", "V-FUM.in", "TEE-FUD.a"),
    ("lox-purge", "V-PG.out", "CK-PG.in"),
])
def test_wires_route_around_parts(app, example, a, b):
    """Lines whose simple shapes all run through a part (here the wire's own valve) take a clear detour."""
    from owlfluids.app import routing

    w = MainWindow()
    w.load_example(example)
    sc = w.scene
    wire = next(x for x in sc.wires if {x.a, x.b} == {a, b})
    obstacles = [it.obstacle() for it in sc.items_by_id.values()]
    assert routing.cost(wire.points, obstacles) < routing.HIT_COST
    w.close()


@pytest.mark.gui
def test_sensors_on_lines(app):
    w = MainWindow()
    w.show()
    w.load_example("lox-feed")
    sc = w.scene
    wire = next(x for x in sc.wires if {x.a, x.b} == {"V-M.out", "INJ.in"})
    p = sc.add_line_sensor(wire, "pressure")
    q = sc.add_line_sensor(wire, "mass_flow")
    assert (p.id, q.id) == ("PT-1", "FM-1")  # named like the properties panel names them
    assert q.at in ("V-M.out", "INJ.in")  # a flow element's end
    assert set(sc.sensor_items) >= {"PT-1", "FM-1", "PT-TK"}  # sensors at a port already show on its line
    a, b = sc.sensor_items["PT-1"].pos(), sc.sensor_items["FM-1"].pos()
    assert a != b and a.y() == b.y()  # spread along the line, above it
    tee_line = sc.line_sensor_end
    assert tee_line(wire, "temperature") is not None
    w.run()
    deadline = time.time() + 60
    while w.runner is None and time.time() < deadline:
        pump(app, 0.05)
    assert w.runner is not None
    w._toggle_valve("V-M")
    pump(app, 1.0)
    assert sc.sensor_labels["FM-1"][1].text().endswith("kg/s")
    assert float(sc.sensor_labels["FM-1"][1].text().split()[0]) > 0.5
    assert sc.sensor_labels["PT-1"][1].text().endswith("psi")
    w.stop()
    sc.clearSelection()
    sc.sensor_items["PT-1"].setSelected(True)
    assert w.properties.findChild(QtWidgets.QLineEdit).text() == "PT-1"  # the sensor's own panel
    sc.delete_selected()
    assert "PT-1" not in [s.id for s in sc.system.sensors] and "PT-1" not in sc.sensor_items
    sc.clearSelection()
    wire.setSelected(True)
    sc.delete_selected()  # the line's sensors go with it
    assert "FM-1" not in [s.id for s in sc.system.sensors]
    w.dirty = False
    w.close()


@pytest.mark.gui
def test_middle_button_drags_the_schematic(app):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    w = MainWindow()
    w.resize(1200, 800)
    w.show()
    w.load_example("test-stand")
    v = w.view
    pump(app, 0.1)
    k = v.transform().m11()
    vp = v.viewport()
    start = vp.rect().center()
    before = v.mapToScene(start)
    QTest.mousePress(vp, Qt.MiddleButton, Qt.NoModifier, start)
    QTest.mouseMove(vp, start + QPoint(80, 40))
    QTest.mouseRelease(vp, Qt.MiddleButton, Qt.NoModifier, start + QPoint(80, 40))
    moved = v.mapToScene(start) - before  # the drawing follows the mouse, even past its edge
    assert moved.x() == pytest.approx(-80 / k, abs=3 / k) and moved.y() == pytest.approx(-40 / k, abs=3 / k)
    w.dirty = False
    w.close()


def test_tank_drawing_follows_its_fill():
    from owlfluids.app import symbols
    from owlfluids.app.schematic import _tank_state

    w, h = 60, 110
    assert symbols.level_height(0.5, w, h) == pytest.approx(h / 2, abs=0.1)  # symmetric
    assert symbols.level_height(0.0, w, h) == 0 and symbols.level_height(1.0, w, h) == pytest.approx(h)
    # the rounded bottom holds less per unit height, so a low fill stands higher than its share
    assert symbols.level_height(0.05, w, h) > 0.05 * h
    st = _tank_state({"fluid": "oxygen", "pressurant": "helium", "fill": 0.3})
    assert st["fill"] == 0.3 and st["gas_colour"] == symbols.fluid_colour("helium", gas=True)
    assert _tank_state({"fluid": "nitrous", "fill": "0.6"})["gas_colour"] == symbols.fluid_colour("nitrous", gas=True)
    assert "fill" not in _tank_state({"fluid": "nitrogen", "pressure": "1000 psi"})  # a gas bottle
    assert _tank_state({"fluid": "water"})["fill"] == 1.0
