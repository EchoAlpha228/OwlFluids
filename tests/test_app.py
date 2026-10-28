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
