"""The main window: edit a system, run it in real time, command valves, watch plots."""

from __future__ import annotations

import threading
import time
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
from ..core import layout as L
from ..core.network import BuildError
from ..examples import EXAMPLES
from ..io.logger import open_logger, read_log
from ..io.system import Sensor, System
from ..sim.realtime import Runner
from ..sim.simulation import Simulation
from . import prefs, symbols
from . import theme as T
from .panels import Palette, Properties, ValvePanel
from .plots import FLOW_ZERO, PlotPanel
from .schematic import Scene, View

RATES = [("0.25x", 0.25), ("0.5x", 0.5), ("1x real time", 1.0), ("2x", 2.0), ("5x", 5.0), ("As fast as possible", 0.0)]


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name) or "run"


def _flow_text(mdot: float) -> str:
    """Mass flow label for the drawing; display only, the solver's values are untouched."""
    return "0 kg/s" if abs(mdot) < FLOW_ZERO else f"{mdot:.3g} kg/s"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        T.apply(QApplication.instance())
        self.path: Path | None = None
        self.dirty = False
        self.runner: Runner | None = None
        self.log = None
        self.driver = None
        self._building = False
        self._build_result = None
        self._build_token = None
        self._warned = 0

        self.scene = Scene()
        self.view = View(self.scene)
        self.setCentralWidget(self.view)

        self.palette = Palette()
        self.properties = Properties()
        self.valves = ValvePanel()
        self.plots = PlotPanel()
        parts = self._dock("PARTS", self.palette, Qt.LeftDockWidgetArea)
        self.side_tabs = QTabWidget()
        self.side_tabs.addTab(self.properties, "Properties")
        self.side_tabs.addTab(self.valves, "Valves")
        inspector = self._dock("INSPECTOR", self.side_tabs, Qt.RightDockWidgetArea)
        self.plot_dock = self._dock("PLOTS", self.plots, Qt.BottomDockWidgetArea)
        self.resizeDocks([parts, inspector], [190, 340], Qt.Horizontal)  # the schematic gets the rest
        # the schematic refits when the plots come and go, so the system stays in view
        self.plot_dock.visibilityChanged.connect(lambda _on: QTimer.singleShot(0, self.view.fit))

        self._make_actions()
        self._make_status()

        self.palette.add_requested.connect(self._add_at_centre)
        self.scene.component_selected.connect(self._selected)
        self.scene.changed_system.connect(self._mark_dirty)
        self.scene.valve_clicked.connect(self._toggle_valve)
        self.properties.edited.connect(self._param_edited)
        self.properties.rename_requested.connect(self._rename)
        self.scene.rename_wanted.connect(self.properties.focus_id)
        self.scene.renamed.connect(self._renamed_by_fluid)
        self.valves.commanded.connect(self._command)
        self.valves.setpoint_changed.connect(self._set_setpoint)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(50)
        screen = QApplication.primaryScreen()
        g = screen.availableGeometry() if screen else None
        self.resize(min(1400, int(g.width() * 0.9)) if g else 1400, min(900, int(g.height() * 0.9)) if g else 900)
        st = prefs.settings()  # the size and dock layout from last time, if any
        if st.value("geometry") is not None:
            self.restoreGeometry(st.value("geometry"))
        if st.value("window_state") is not None:
            self.restoreState(st.value("window_state"))
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
        m_help = self.menuBar().addMenu("&Help")

        def act(text, slot, shortcut=None, menu=None, toolbar=False, tip=""):
            a = QAction(text, self)
            a.triggered.connect(slot)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            if tip or shortcut:
                keys = QKeySequence(shortcut).toString(QKeySequence.NativeText) if shortcut else ""
                a.setToolTip(f"{tip or text.lstrip('▶⏸■⟲● ')}" + (f" ({keys})" if keys else ""))
            if menu is not None:
                menu.addAction(a)
            if toolbar:
                tb.addAction(a)
            return a

        act("New", self.new_system, QKeySequence.New, m_file, True)
        act("Open…", self.open_dialog, QKeySequence.Open, m_file, True)
        act("Save", self.save, QKeySequence.Save, m_file, True)
        act("Save as…", self.save_as, QKeySequence.SaveAs, m_file)
        self.m_recent = m_file.addMenu("Recent")
        self.m_recent.aboutToShow.connect(self._fill_recent)
        m_ex = m_file.addMenu("Examples")
        for name in EXAMPLES:
            act(name, lambda _=False, n=name: self.load_example(n), menu=m_ex)
        m_file.addSeparator()
        act("Overlay a recorded log (CSV/HDF5)…", self.overlay_log, menu=m_file)
        m_file.addSeparator()
        act("Quit", self.close, QKeySequence.Quit, m_file)

        tb.addSeparator()
        self.a_rotate = act("↻ Rotate", lambda: self.scene.rotate_selected(90), "R", m_edit, True,
                            "Rotate the selected parts clockwise")
        self.a_rotate_back = act("↺ Rotate back", lambda: self.scene.rotate_selected(-90), "Shift+R", m_edit, True,
                                 "Rotate the selected parts counter-clockwise")
        self.a_delete = act("Delete", self.scene.delete_selected, QKeySequence.Delete, m_edit)
        self.a_rename = act("Rename", self.properties.focus_id, "F2", m_edit,
                            tip="Rename the selected part or sensor (or double-click it)")
        act("Settings…", self.edit_settings, "Ctrl+,", m_edit, tip="Time step, ambient and solver settings")

        tb.addSeparator()
        self.a_run = act("▶ Run", self.run, "F5", m_run, True, "Build the system and run it in real time")
        self.a_pause = act("⏸ Pause", self.pause, "F6", m_run, True, "Pause or resume the clock")
        self.a_stop = act("■ Stop", self.stop, "Shift+F5", m_run, True, "Stop and return to editing")
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
        self.a_log = act("● Record", self.toggle_log, "Ctrl+L", m_run, True, "Record sensors and valves to a file")
        self.a_log.setCheckable(True)
        m_run.addSeparator()
        self.a_synnax = act("Connect to Synnax…", self.toggle_synnax, None, m_run, False,
                            "HOOTL or HITL: send sensors to Synnax, take valve channels from it")
        self.a_synnax.setCheckable(True)
        act("Fit schematic", self.view.fit, "F", m_view, True)
        m_view.addAction(self.plot_dock.toggleViewAction())
        from .updates import Updater

        self.updater = Updater(self)
        act("Check for updates…", lambda: self.updater.check(), menu=m_help)
        act("Update settings…", self.updater.settings, menu=m_help)
        m_help.addSeparator()
        act("About OwlFluids", self.about, menu=m_help)
        self._set_running_ui(False)

    def about(self) -> None:
        from .. import __version__

        QMessageBox.about(self, "About OwlFluids", f"OwlFluids {__version__}\n\n"
                          "Real-time 1-D fluid network simulator for test automation.")

    def _make_status(self) -> None:
        self.status_badge = QLabel()
        self.status_badge.setAlignment(Qt.AlignCenter)
        self.status_badge.setMinimumWidth(84)
        self._badge = None
        self.status_time = QLabel("")
        T.monospace(self.status_time)
        self.status_perf = QLabel("")
        T.monospace(self.status_perf)
        self.statusBar().addWidget(self.status_badge)
        self.statusBar().addWidget(self.status_time, 1)
        self.statusBar().addPermanentWidget(self.status_perf)
        self._set_badge("stopped")

    def _set_badge(self, state: str) -> None:
        if state == self._badge:  # a style sheet change re-polishes the label; only on a change
            return
        self._badge = state
        bg, fg = T.BADGES[state]
        self.status_badge.setText(state.upper())
        self.status_badge.setStyleSheet(f"background: {bg}; color: {fg}; font-weight: bold; padding: 2px 8px;"
                                        " border-radius: 2px; letter-spacing: 1px;")

    def _set_running_ui(self, running: bool) -> None:
        self.a_run.setEnabled(not running)
        self.a_pause.setEnabled(running)
        self.a_stop.setEnabled(running)
        self.a_reset.setEnabled(running)
        self.a_log.setEnabled(running)
        self.a_synnax.setEnabled(running)
        self.a_rotate.setEnabled(not running)
        self.a_rotate_back.setEnabled(not running)
        self.a_delete.setEnabled(not running)
        self.a_rename.setEnabled(not running)
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
        fn, _ = QFileDialog.getOpenFileName(self, "Open system", self._start_dir(), "OwlFluids systems (*.yaml *.yml)")
        if fn:
            self.open_file(fn)

    def open_file(self, fn: str | Path) -> None:
        try:
            sysm = System.load(fn)
        except Exception as e:  # noqa: BLE001 - any bad file is reported, not fatal
            QMessageBox.critical(self, "Could not open", str(e))
            return
        self._set_system(sysm, Path(fn))
        prefs.add_recent(fn)

    def _fill_recent(self) -> None:
        self.m_recent.clear()
        files = prefs.recent()
        for f in files:
            a = self.m_recent.addAction(Path(f).name)
            a.setToolTip(f)
            a.triggered.connect(lambda _c=False, f=f: self._confirm_discard() and self.open_file(f))
        if not files:
            self.m_recent.addAction("No recent systems").setEnabled(False)

    def _set_system(self, sysm: System, path: Path | None) -> None:
        self.stop()
        self.path = path
        self.scene.set_system(sysm)
        self.properties.show_component(None, sysm)
        self.plot_dock.hide()  # nothing to plot while editing; the schematic gets the room (Run brings it back)
        self.dirty = False
        self._update_title()
        QTimer.singleShot(0, self.view.fit)

    def save(self) -> bool:
        if self.path is None:
            return self.save_as()
        try:
            self.scene.system.save(self.path)
        except OSError as e:
            QMessageBox.critical(self, "Could not save", f"{e}\n\nTry File > Save as… to pick another folder.")
            return False
        self.dirty = False
        self._update_title()
        return True

    def save_as(self) -> bool:
        fn, _ = QFileDialog.getSaveFileName(self, "Save system",
                                            str(Path(self._start_dir()) / (self.scene.system.name + ".yaml")),
                                            "OwlFluids systems (*.yaml)")
        if not fn:
            return False
        old_path, old_name = self.path, self.scene.system.name
        self.path = Path(fn)
        if self.scene.system.name == "untitled":
            self.scene.system.name = self.path.stem
        if self.save():
            prefs.add_recent(self.path)
            return True
        self.path, self.scene.system.name = old_path, old_name  # Ctrl+S must not retry the bad path
        return False

    def _start_dir(self) -> str:
        """Where file dialogs open: next to the current system file, else the last folder used."""
        return str(self.path.parent) if self.path else prefs.last_dir()

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
            st = prefs.settings()
            st.setValue("geometry", self.saveGeometry())
            st.setValue("window_state", self.saveState())
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
        fn, _ = QFileDialog.getOpenFileName(self, "Overlay a recorded log", self._start_dir(),
                                            "Logs (*.csv *.h5 *.hdf5)")
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
        if isinstance(comp, Sensor):
            self.properties.show_sensor(comp, self.scene.system, read_only=self.scene.running)
            self.side_tabs.setCurrentWidget(self.properties)
            return
        self.properties.show_component(comp, self.scene.system, read_only=self.scene.running)
        if comp is not None:
            self.side_tabs.setCurrentWidget(self.properties)

    def _param_edited(self) -> None:
        item = self.scene.items_by_id.get(self.properties.comp.id) if self.properties.comp else None
        if item is not None:
            item.update()
        self.scene.sync_sensor_labels()
        self.scene.refresh_media()
        self._mark_dirty()

    def _rename(self, old: str, new: str) -> None:
        try:
            self.scene.rename(old, new)
        except ValueError as e:
            QMessageBox.warning(self, "Rename", str(e))
        self.properties.show_component(self.scene.system.component(new) if any(
            c.id == new for c in self.scene.system.components) else None, self.scene.system)

    def _renamed_by_fluid(self) -> None:
        """A default name followed its commodity: show it in the panel once the edit that caused it is done."""
        QTimer.singleShot(0, self.scene._on_selection)

    # ------------------------------------------------------------------ running
    def run(self) -> None:
        if self.runner is not None or self._building:
            return
        if not self.scene.system.components:
            QMessageBox.information(self, "Run", "Add some components first (or open an example).")
            return
        self._building = True
        self._build_result = None
        token = self._build_token = object()
        self._set_badge("building")
        self.status_time.setText("building network and loading property tables…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        system = System.from_dict(self.scene.system.to_dict())

        def build():
            try:
                res = Simulation(system)
            except BuildError as e:
                res = e
            except Exception as e:  # noqa: BLE001
                res = RuntimeError(traceback.format_exc(limit=3) + str(e))
            if self._build_token is token:  # a build cancelled by Stop or Open is dropped
                self._build_result = (token, res)

        threading.Thread(target=build, daemon=True).start()

    def _finish_build(self) -> None:
        self._building = False
        QApplication.restoreOverrideCursor()
        _token, res = self._build_result
        if isinstance(res, Exception):
            self._set_badge("stopped")
            self.status_time.setText("")
            QMessageBox.critical(self, "Cannot run this system", str(res))
            return
        sim: Simulation = res
        self.runner = Runner(sim, rate=self.rate_box.currentData())
        self._warned = 0
        manual = {v: True for v in sim.manual_valves()}
        media = self.scene.media  # the fluid each line carries, as traced for the line colours
        fluids = {v: (media.get(f"{v}.in") or media.get(f"{v}.out") or (None,))[0] for v in sim.valves()}
        self.valves.set_valves(sim.valves(), manual, sim.regulators(), fluids)
        self.plots.configure(sim.system, sim.valves())
        if not self.plot_dock.isVisible():
            self.plot_dock.show()
            self.resizeDocks([self.plot_dock], [int(self.height() * 0.42)], Qt.Vertical)
        names = sim.net.fluids.names
        self._node_fluids = [names[k] if k >= 0 else None for k in sim.net.NI[:, L.N_FLUID]]
        self._fluid_colours = {}
        self.scene.set_running(True)
        self._colour_wires(sim)
        self._flow_wires = self._wire_links(sim)
        self._last_tick = time.perf_counter()
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
        if self._building:  # forget a build in progress; its result is dropped when it lands
            self._building = False
            self._build_token = None
            QApplication.restoreOverrideCursor()
        if self.driver is not None:
            self.toggle_synnax(False)
        if self.log is not None:
            self.toggle_log(False)
        if self.runner is not None:
            self.runner.stop()
            self.runner = None
        self.a_pause.setText("⏸ Pause")
        self.scene.set_running(False)
        self._colour_wires(None)
        self.valves.set_valves([], {})
        self._set_running_ui(False)
        self._set_badge("stopped")
        self.status_time.setText("")

    def reset(self) -> None:
        if self.runner is not None:
            self.runner.reset()
            self.statusBar().showMessage("reset to initial conditions", 3000)

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

    def _set_setpoint(self, name: str, pressure: float) -> None:
        if self.runner is not None:
            self.runner.set_setpoint(name, pressure)

    def toggle_log(self, on: bool) -> None:
        if on and self.runner is not None and self.log is None:
            stem = self.path.stem if self.path else _safe_name(self.scene.system.name)
            default = Path(self._start_dir()) / f"{stem}-{time.strftime('%Y%m%d-%H%M%S')}.csv"
            fn, _ = QFileDialog.getSaveFileName(self, "Record to", str(default), "CSV (*.csv);;HDF5 (*.h5)")
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

    def toggle_synnax(self, on: bool) -> None:
        if on and self.runner is not None and self.driver is None:
            fn, _ = QFileDialog.getOpenFileName(self, "Synnax settings (a YAML file with a synnax: section)",
                                                self._start_dir(),
                                                "YAML (*.yaml *.yml)")
            if not fn:
                self.a_synnax.setChecked(False)
                return
            try:
                from ..drivers.synnax import SynnaxConfig, SynnaxDriver

                self.driver = SynnaxDriver(self.runner, SynnaxConfig.load(fn))
                self.driver.start()
            except Exception as e:  # noqa: BLE001 - connection problems are reported, not fatal
                self.driver = None
                self.a_synnax.setChecked(False)
                QMessageBox.critical(self, "Synnax", f"Could not connect: {e}")
                return
            self.statusBar().showMessage(f"Synnax {self.driver.cfg.mode.upper()} connected: "
                                         f"{len(self.driver.sensors)} sensors out, "
                                         f"{len(self.driver.valves)} valves in", 6000)
        elif not on and self.driver is not None:
            drv, self.driver = self.driver, None
            drv.stop()
            self.a_synnax.setChecked(False)

    def _tick(self) -> None:
        if self._building and self._build_result is not None and self._build_result[0] is self._build_token:
            self._finish_build()
        r = self.runner
        if r is None:
            return
        if r.error is not None:
            err = r.error
            self.stop()
            QMessageBox.critical(self, "Simulation stopped", f"{type(err).__name__}: {err}")
            return
        if self.driver is not None and self.driver.error is not None:
            err = self.driver.error
            self.toggle_synnax(False)
            QMessageBox.warning(self, "Synnax", f"Synnax link stopped: {err}")
        for w in r.warnings[self._warned:]:
            self.statusBar().showMessage(w, 8000)
        self._warned = len(r.warnings)
        f = r.frame
        self.valves.show_frame(f)
        self.scene.show_values(self._component_values(r.sim, f))
        if self.scene.sensor_items:
            self.scene.show_sensor_values(self._sensor_values(f))
        self._show_wire_colours(f)
        now = time.perf_counter()
        dt, self._last_tick = (0.0 if r.paused else now - self._last_tick), now
        mdot = f.link_mdot
        flows = {}
        for k, terms in self._flow_wires:
            q = sum(sgn * mdot[li] for li, sgn in terms)
            if abs(q) >= FLOW_ZERO:
                flows[k] = 1 if q > 0 else -1
        self.scene.show_flows(flows, dt)
        if self.plot_dock.isVisible():
            self.plots.update_from(r.history)
        st = f.stats
        rate = self.rate_box.currentData()
        self._set_badge("paused" if r.paused else "running")
        self.status_time.setText(f"T+ {f.t:9.2f} s")
        warn = f.load > 0.8 or f.lag > 0.05  # near the real-time limit: caution colour
        if warn != self.status_perf.property("warn"):
            self.status_perf.setProperty("warn", warn)
            self.status_perf.setStyleSheet(f"color: {T.AMBER};" if warn else "")
        self.status_perf.setText(
            f"CPU {f.load * 100:4.0f}% of {'real time' if rate else 'max speed'}   lag {f.lag * 1000:4.1f} ms   "
            f"slips {r.slips}   forced steps {st.get('forced_steps', 0)}"
            + (f"   Synnax ↑{self.driver.writes} ↓{self.driver.commands}" if self.driver else "")
            + ("   ● REC" if self.log else ""))

    def _fluid_colour(self, node: int, gas: bool):
        """Material colour of a node's fluid; cached, since it is asked for every component every tick."""
        key = (node, gas)
        c = self._fluid_colours.get(key)
        if c is None:
            c = self._fluid_colours[key] = symbols.fluid_colour(self._node_fluids[node], gas)
        return c

    def _colour_wires(self, sim: Simulation | None) -> None:
        """While running, each wire takes the colour of what is in it; gas is see-through."""
        if sim is None:
            self._wire_nodes = []
            self.scene.refresh_media()
            return
        net = sim.net
        self._wire_nodes = [(w, net.port_nodes.get(w.a, net.port_nodes.get(w.b))) for w in self.scene.wires]
        self._wire_nodes = [(w, i) for w, i in self._wire_nodes if i is not None]

    def _show_wire_colours(self, f) -> None:
        for w, i in self._wire_nodes:
            w.set_colour(self._fluid_colour(i, f.node_x[i] > 0.5))

    def _wire_links(self, sim: Simulation) -> list[tuple[int, list[tuple[int, float]]]]:
        """For each wire that carries flow: (wire index, [(link, sign), ...]); flow a->b is the sum of sign * mdot.

        A wire joins two ports that share a node. Its flow is the flow of the flow element
        (valve, orifice, regulator...) at one of its ends; between a volume (tank, manifold)
        and a tee it is everything leaving the volume's node.
        """
        net = sim.net
        n_links = net.LI.shape[0]

        def link_at(port: str):
            """The link of the component that owns ``port`` with an end on that port's node, and that end."""
            comp, node = port.split(".")[0], net.port_nodes.get(port)
            for k in net.comp_links.get(comp, []):
                if net.LI[k, L.L_A] == node:
                    return [(k, -1.0)]  # positive flow leaves this node through the component
                if net.LI[k, L.L_B] == node:
                    return [(k, 1.0)]  # positive flow arrives here out of the component
            return None

        def out_of_volume(port: str):
            """Everything leaving a volume's node, for a wire from the volume to a tee."""
            comp, node = port.split(".")[0], net.port_nodes.get(port)
            if node is None or not net.comp_nodes.get(comp) or get_type(sim.system.component(comp).type).merges_ports:
                return None
            terms = [(k, 1.0 if net.LI[k, L.L_A] == node else -1.0) for k in range(n_links)
                     if node in (net.LI[k, L.L_A], net.LI[k, L.L_B])]
            return terms or None

        def flip(terms):
            return None if terms is None else [(k, -sgn) for k, sgn in terms]

        out = []
        for k, w in enumerate(self.scene.wires):
            terms = (link_at(w.a) or flip(link_at(w.b)) or out_of_volume(w.a) or flip(out_of_volume(w.b)))
            if terms:
                out.append((k, terms))
        return out

    def _sensor_values(self, f) -> dict[str, str]:
        """Readings beside the sensor bubbles on the lines, in each sensor's display units."""
        out = {}
        for sid, item in self.scene.sensor_items.items():
            v = f.sensors.get(sid)
            if v is None:
                continue
            s = item.sensor
            unit = s.units or units.DEFAULT_UNITS.get(s.type, "")
            if s.type == "mass_flow" and abs(v) < FLOW_ZERO:
                v = 0.0
            out[sid] = f"{units.convert(v, unit):.4g} {unit}".strip()
        return out

    def _component_values(self, sim: Simulation, f) -> dict:
        net = sim.net
        out = {}
        for c in sim.system.components:
            ct = get_type(c.type)
            nodes = net.comp_nodes.get(c.id, [])
            links = net.comp_links.get(c.id, [])
            state: dict = {}
            text = ""
            if ct.is_valve or ct.type_name in ("check_valve", "relief_valve", "regulator", "air_pump"):
                if links:
                    k = links[0]
                    state["position"] = f.valves.get(c.id, f.link_open[k])
                    text = _flow_text(f.link_mdot[k])
            elif ct.type_name in ("orifice", "fitting"):
                if links:
                    text = _flow_text(f.link_mdot[links[0]])
                i = net.port_nodes.get(f"{c.id}.in")
                if ct.type_name == "fitting" and i is not None:
                    state["line_colour"] = self._fluid_colour(i, f.node_x[i] > 0.5)
            elif ct.type_name == "boundary":
                text = ""
            elif nodes:
                i = nodes[0]
                p = units.to_psi(f.node_p[i])
                text = f"{p:.0f} psi  {f.node_t[i]:.0f} K"
                if ct.type_name == "volume":
                    state["fill_colour"] = self._fluid_colour(i, f.node_x[i] > 0.5)
                if ct.type_name == "tank" and len(nodes) > 1:  # liquid under a pressurant
                    liq = nodes[1]
                    vol = net.NF[nodes[0], L.NF_VOL]
                    state["fill"] = min(max(f.node_m[liq] / max(f.node_rho[liq], 1e-9) / vol, 0.0), 1.0)
                    state["liquid_colour"] = self._fluid_colour(liq, False)
                    state["gas_colour"] = self._fluid_colour(i, True)
                    text = f"{p:.0f} psi  {f.node_t[liq]:.0f} K liq"
                elif ct.type_name == "tank":
                    state["liquid_colour"] = self._fluid_colour(i, False)
                    state["gas_colour"] = self._fluid_colour(i, True)
                    state["fill"] = sim.liquid_fraction(i, float(f.node_t[i]), float(f.node_x[i]),
                                                        float(f.node_rho[i]))
            out[c.id] = (text, state)
        return out

