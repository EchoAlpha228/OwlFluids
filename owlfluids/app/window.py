"""The main window: edit a system, run it in real time, command valves, watch plots."""

from __future__ import annotations

import threading
import traceback
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QToolBar,
)

from .. import units
from ..components import get_type
from ..core.network import BuildError
from ..examples import EXAMPLES
from ..io.logger import open_logger, read_log
from ..io.system import System
from ..sim.realtime import Runner
from ..sim.simulation import Simulation
from . import symbols
from .panels import Palette, Properties, ValvePanel
from .plots import PlotPanel
from .schematic import Scene, View

RATES = [("0.25x", 0.25), ("0.5x", 0.5), ("1x real time", 1.0), ("2x", 2.0), ("5x", 5.0), ("As fast as possible", 0.0)]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.path: Path | None = None
        self.dirty = False
        self.runner: Runner | None = None
        self.log = None
        self._building = False
        self._build_result = None

        self.scene = Scene()
        self.view = View(self.scene)
        self.setCentralWidget(self.view)

        self.palette = Palette()
        self.properties = Properties()
        self.valves = ValvePanel()
        self.plots = PlotPanel()
        self._dock("Parts", self.palette, Qt.LeftDockWidgetArea)
        self.side_tabs = QTabWidget()
        self.side_tabs.addTab(self.properties, "Properties")
        self.side_tabs.addTab(self.valves, "Valves")
        self._dock("Inspector", self.side_tabs, Qt.RightDockWidgetArea)
        self.plot_dock = self._dock("Plots", self.plots, Qt.BottomDockWidgetArea)

        self._make_actions()
        self._make_status()

        self.palette.add_requested.connect(self._add_at_centre)
        self.scene.component_selected.connect(self._selected)
        self.scene.changed_system.connect(self._mark_dirty)
        self.scene.valve_clicked.connect(self._toggle_valve)
        self.properties.edited.connect(self._param_edited)
        self.properties.rename_requested.connect(self._rename)
        self.valves.commanded.connect(self._command)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(50)
        self.resize(1400, 900)
        self.new_system()

    def _dock(self, title, widget, area) -> QDockWidget:
        d = QDockWidget(title, self)
        d.setObjectName(title)
        d.setWidget(widget)
        self.addDockWidget(area, d)
        return d

    # ------------------------------------------------------------------ actions
    def _make_actions(self) -> None:
        tb = QToolBar("Main")
        tb.setObjectName("main toolbar")
        tb.setMovable(False)
        self.addToolBar(tb)
        m_file = self.menuBar().addMenu("&File")
        m_edit = self.menuBar().addMenu("&Edit")
        m_run = self.menuBar().addMenu("&Run")
        m_view = self.menuBar().addMenu("&View")

        def act(text, slot, shortcut=None, menu=None, toolbar=False, tip=""):
            a = QAction(text, self)
            a.triggered.connect(slot)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            if tip:
                a.setToolTip(tip)
            if menu is not None:
                menu.addAction(a)
            if toolbar:
                tb.addAction(a)
            return a

        act("New", self.new_system, QKeySequence.New, m_file, True)
        act("Open…", self.open_dialog, QKeySequence.Open, m_file, True)
        act("Save", self.save, QKeySequence.Save, m_file, True)
        act("Save as…", self.save_as, QKeySequence.SaveAs, m_file)
        m_ex = m_file.addMenu("Examples")
        for name in EXAMPLES:
            act(name, lambda _=False, n=name: self.load_example(n), menu=m_ex)
        m_file.addSeparator()
        act("Overlay a recorded log (CSV/HDF5)…", self.overlay_log, menu=m_file)
        m_file.addSeparator()
        act("Quit", self.close, QKeySequence.Quit, m_file)

        self.a_rotate = act("Rotate", self.scene.rotate_selected, "R", m_edit)
        self.a_delete = act("Delete", self.scene.delete_selected, QKeySequence.Delete, m_edit)
        act("Settings…", self.edit_settings, menu=m_edit)

        tb.addSeparator()
        self.a_run = act("▶ Run", self.run, "F5", m_run, True, "Build the system and run it in real time")
        self.a_pause = act("⏸ Pause", self.pause, "F6", m_run, True)
        self.a_stop = act("■ Stop", self.stop, "Shift+F5", m_run, True)
        self.a_reset = act("⟲ Reset", self.reset, "Ctrl+R", m_run, True, "Back to initial conditions")
        tb.addSeparator()
        tb.addWidget(QLabel(" Speed "))
        self.rate_box = QComboBox()
        for label, r in RATES:
            self.rate_box.addItem(label, r)
        self.rate_box.setCurrentIndex(2)
        self.rate_box.currentIndexChanged.connect(self._rate_changed)
        tb.addWidget(self.rate_box)
        tb.addSeparator()
        self.a_log = act("● Record", self.toggle_log, None, m_run, True, "Record sensors and valves to a file")
        self.a_log.setCheckable(True)
        act("Fit schematic", self.view.fit, "F", m_view, True)
        m_view.addAction(self.plot_dock.toggleViewAction())
        self._set_running_ui(False)

    def _make_status(self) -> None:
        self.status_time = QLabel("stopped")
        self.status_perf = QLabel("")
        self.statusBar().addWidget(self.status_time, 1)
        self.statusBar().addPermanentWidget(self.status_perf)

    def _set_running_ui(self, running: bool) -> None:
        self.a_run.setEnabled(not running)
        self.a_pause.setEnabled(running)
        self.a_stop.setEnabled(running)
        self.a_reset.setEnabled(running)
        self.a_log.setEnabled(running)
        self.a_rotate.setEnabled(not running)
        self.a_delete.setEnabled(not running)
        self.palette.setEnabled(not running)

    # ------------------------------------------------------------------ files
    def new_system(self) -> None:
        if not self._confirm_discard():
            return
        self._set_system(System("untitled"), None)

    def load_example(self, name: str) -> None:
        if not self._confirm_discard():
            return
        self._set_system(EXAMPLES[name](), None)

    def open_dialog(self) -> None:
        if not self._confirm_discard():
            return
        fn, _ = QFileDialog.getOpenFileName(self, "Open system", "", "OwlFluids systems (*.yaml *.yml)")
        if fn:
            self.open_file(fn)

    def open_file(self, fn: str | Path) -> None:
        try:
            sysm = System.load(fn)
        except Exception as e:  # noqa: BLE001 - any bad file is reported, not fatal
            QMessageBox.critical(self, "Could not open", str(e))
            return
        self._set_system(sysm, Path(fn))

    def _set_system(self, sysm: System, path: Path | None) -> None:
        self.stop()
        self.path = path
        self.scene.set_system(sysm)
        self.properties.show_component(None, sysm)
        self.dirty = False
        self._update_title()
        QTimer.singleShot(0, self.view.fit)

    def save(self) -> bool:
        if self.path is None:
            return self.save_as()
        self.scene.system.save(self.path)
        self.dirty = False
        self._update_title()
        return True

    def save_as(self) -> bool:
        fn, _ = QFileDialog.getSaveFileName(self, "Save system", self.scene.system.name + ".yaml",
                                            "OwlFluids systems (*.yaml)")
        if not fn:
            return False
        self.path = Path(fn)
        if self.scene.system.name == "untitled":
            self.scene.system.name = self.path.stem
        return self.save()

    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        r = QMessageBox.question(self, "Unsaved changes", "Save changes to this system first?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Save:
            return self.save()
        return r == QMessageBox.Discard

    def closeEvent(self, e):
        if self._confirm_discard():
            self.stop()
            e.accept()
        else:
            e.ignore()

    def _mark_dirty(self) -> None:
        self.dirty = True
        self._update_title()

    def _update_title(self) -> None:
        name = self.path.name if self.path else self.scene.system.name
        self.setWindowTitle(f"OwlFluids — {name}{' *' if self.dirty else ''}")

    def edit_settings(self) -> None:
        from .settings import SettingsDialog

        if SettingsDialog(self.scene.system, self).exec():
            self._mark_dirty()

    def overlay_log(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(self, "Overlay a recorded log", "", "Logs (*.csv *.h5 *.hdf5)")
        if not fn:
            return
        try:
            names, data = read_log(fn)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Could not read log", str(e))
            return
        if not self.plots.curves:
            self.plots.configure(self.scene.system, [c.id for c in self.scene.system.components
                                                     if get_type(c.type).is_valve])
        n = self.plots.load_overlay(names, data)
        self.statusBar().showMessage(f"overlaid {n} columns from {Path(fn).name}", 5000)

    # ------------------------------------------------------------------ editing
    def _add_at_centre(self, type_name: str) -> None:
        if not self.scene.running:
            self.scene.add_component(type_name, self.view.mapToScene(self.view.viewport().rect().center()))

    def _selected(self, comp) -> None:
        self.properties.show_component(comp, self.scene.system, read_only=self.scene.running)
        if comp is not None:
            self.side_tabs.setCurrentWidget(self.properties)

    def _param_edited(self) -> None:
        item = self.scene.items_by_id.get(self.properties.comp.id) if self.properties.comp else None
        if item is not None:
            item.update()
        self._mark_dirty()

    def _rename(self, old: str, new: str) -> None:
        try:
            self.scene.rename(old, new)
        except ValueError as e:
            QMessageBox.warning(self, "Rename", str(e))
        self.properties.show_component(self.scene.system.component(new) if any(
            c.id == new for c in self.scene.system.components) else None, self.scene.system)

    # ------------------------------------------------------------------ running
    def run(self) -> None:
        if self.runner is not None or self._building:
            return
        if not self.scene.system.components:
            QMessageBox.information(self, "Run", "Add some components first (or open an example).")
            return
        self._building = True
        self._build_result = None
        self.status_time.setText("building network and loading property tables…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        system = System.from_dict(self.scene.system.to_dict())

        def build():
            try:
                self._build_result = Simulation(system)
            except BuildError as e:
                self._build_result = e
            except Exception as e:  # noqa: BLE001
                self._build_result = RuntimeError(traceback.format_exc(limit=3) + str(e))

        threading.Thread(target=build, daemon=True).start()

    def _finish_build(self) -> None:
        self._building = False
        QApplication.restoreOverrideCursor()
        res = self._build_result
        if isinstance(res, Exception):
            self.status_time.setText("stopped")
            QMessageBox.critical(self, "Cannot run this system", str(res))
            return
        sim: Simulation = res
        self.runner = Runner(sim, rate=self.rate_box.currentData())
        throttles = {v: get_type(sim.system.component(v).type).type_name == "throttle_valve" for v in sim.valves()}
        self.valves.set_valves(sim.valves(), throttles)
        self.plots.configure(sim.system, sim.valves())
        self.scene.set_running(True)
        self.properties.show_component(None, self.scene.system)
        self._set_running_ui(True)
        self.side_tabs.setCurrentWidget(self.valves)
        for w in sim.net.warnings:
            self.statusBar().showMessage(w, 8000)
        self.runner.start()

    def pause(self) -> None:
        if self.runner is None:
            return
        if self.runner.paused:
            self.runner.resume()
            self.a_pause.setText("⏸ Pause")
        else:
            self.runner.pause()
            self.a_pause.setText("▶ Resume")

    def stop(self) -> None:
        if self.log is not None:
            self.toggle_log(False)
        if self.runner is not None:
            self.runner.stop()
            self.runner = None
        self.a_pause.setText("⏸ Pause")
        self.scene.set_running(False)
        self.valves.set_valves([], {})
        self._set_running_ui(False)
        self.status_time.setText("stopped")

    def reset(self) -> None:
        if self.runner is not None:
            self.runner.reset()

    def _rate_changed(self) -> None:
        if self.runner is not None:
            self.runner.set_rate(self.rate_box.currentData())

    def _toggle_valve(self, name: str) -> None:
        if self.runner is not None and name in self.runner.sim.net.valves:
            cmd = self.runner.frame.commands.get(name, 0.0)
            self.runner.command(name, 0.0 if cmd > 0.5 else 1.0)

    def _command(self, name: str, position: float) -> None:
        if self.runner is not None:
            self.runner.command(name, position)

    def toggle_log(self, on: bool) -> None:
        if on and self.runner is not None and self.log is None:
            fn, _ = QFileDialog.getSaveFileName(self, "Record to", "run.csv", "CSV (*.csv);;HDF5 (*.h5)")
            if not fn:
                self.a_log.setChecked(False)
                return
            self.log = open_logger(fn, self.runner)
            self.statusBar().showMessage(f"recording to {fn}", 4000)
        elif not on and self.log is not None:
            log, self.log = self.log, None
            if self.runner is not None:
                self.runner.submit(lambda sim: log.close())
            else:
                log.close()
            self.a_log.setChecked(False)

    def _tick(self) -> None:
        if self._building and self._build_result is not None:
            self._finish_build()
        r = self.runner
        if r is None:
            return
        if r.error is not None:
            err = r.error
            self.stop()
            QMessageBox.critical(self, "Simulation stopped", f"{type(err).__name__}: {err}")
            return
        f = r.frame
        self.valves.show_frame(f)
        self.scene.show_values(self._component_values(r.sim, f))
        if self.plot_dock.isVisible():
            self.plots.update_from(r.history)
        st = f.stats
        rate = self.rate_box.currentData()
        self.status_time.setText(f"t = {f.t:8.2f} s" + ("   (paused)" if r.paused else ""))
        self.status_perf.setText(
            f"CPU {f.load * 100:4.0f}% of {'real time' if rate else 'max speed'}   lag {f.lag * 1000:4.1f} ms   "
            f"slips {r.slips}   forced steps {st.get('forced_steps', 0)}")

    def _component_values(self, sim: Simulation, f) -> dict:
        net = sim.net
        out = {}
        for c in sim.system.components:
            ct = get_type(c.type)
            nodes = net.comp_nodes.get(c.id, [])
            links = net.comp_links.get(c.id, [])
            state: dict = {}
            text = ""
            if ct.is_valve or ct.type_name in ("check_valve", "relief_valve", "regulator"):
                if links:
                    k = links[0]
                    state["position"] = f.valves.get(c.id, sim.lout[k, 1])
                    text = f"{f.link_mdot[k]:.3g} kg/s"
            elif ct.type_name in ("orifice", "fitting"):
                if links:
                    text = f"{f.link_mdot[links[0]]:.3g} kg/s"
            elif ct.type_name == "boundary":
                text = ""
            elif ct.type_name == "pipe" and not nodes and links:
                text = f"{f.link_mdot[links[0]]:.3g} kg/s"
            elif nodes:
                i = nodes[0]
                p = units.to_psi(f.node_p[i])
                text = f"{p:.0f} psi  {f.node_t[i]:.0f} K"
                if ct.type_name == "pressurized_tank" and len(nodes) > 1:
                    liq = nodes[1]
                    vol = sim.NF[nodes[0], 0]
                    state["fill"] = min(max(sim.nout[liq, 5] / max(sim.nout[liq, 3], 1e-9) / vol, 0.0), 1.0)
                    state["liquid_colour"] = symbols.CRYO if f.node_t[liq] < 150 else symbols.LIQUID
                    text = f"{p:.0f} psi  {f.node_t[liq]:.0f} K liq"
                elif ct.type_name == "pipe" and links:
                    mid = links[len(links) // 2]
                    text = f"{f.link_mdot[mid]:.3g} kg/s  {p:.0f} psi"
                    if f.node_x[i] < 0.0 or (0.0 <= f.node_x[i] <= 0.5):
                        state["pipe_colour"] = symbols.CRYO if f.node_t[i] < 150 else symbols.LIQUID
                elif ct.type_name == "tank" and 0.0 <= f.node_x[i] <= 1.0:
                    state["fill"] = 1.0 - f.node_x[i]
            out[c.id] = (text, state)
        return out

