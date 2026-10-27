"""Live strip charts of sensors and valve positions, with optional overlay of a recorded log."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .. import units
from . import theme as T

FLOW_ZERO = 1e-4  # kg/s; smaller flows are solver residue (or a closed path) and are shown as 0
GROUPS = [("pressure", "Pressure"), ("temperature", "Temperature"), ("mass_flow", "Mass flow"),
          ("other", "Other"), ("valve", "Valve position")]
COLOURS = T.TRACES


def _swatch(colour: str) -> QIcon:
    pm = QPixmap(10, 10)
    pm.fill(QColor(colour))
    return QIcon(pm)


class PlotPanel(QWidget):
    def __init__(self):
        super().__init__()
        # no antialiasing: live curves repaint 20 times a second and it roughly halves their cost
        pg.setConfigOptions(antialias=False, background=T.GRAY_950, foreground=T.GRAY_300)
        outer = QHBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        outer.addWidget(split)

        side = QWidget()
        sv = QVBoxLayout(side)
        sv.setContentsMargins(0, 0, 0, 0)
        self.series_list = QListWidget()
        self.series_list.itemChanged.connect(lambda _: self._rebuild_visibility())
        show = QLabel("SIGNALS")
        show.setObjectName("heading")
        sv.addWidget(show)
        sv.addWidget(self.series_list)
        row = QHBoxLayout()
        row.addWidget(QLabel("Window"))
        self.window_box = QComboBox()
        for label, sec in (("10 s", 10.0), ("30 s", 30.0), ("60 s", 60.0), ("All", 0.0)):
            self.window_box.addItem(label, sec)
        self.window_box.setCurrentIndex(1)
        row.addWidget(self.window_box)
        sv.addLayout(row)
        self.clear_overlay_btn = QPushButton("Clear overlay")
        self.clear_overlay_btn.clicked.connect(self.clear_overlay)
        self.clear_overlay_btn.setEnabled(False)
        sv.addWidget(self.clear_overlay_btn)
        split.addWidget(side)

        self.graphics = pg.GraphicsLayoutWidget()
        split.addWidget(self.graphics)
        split.setSizes([160, 900])
        self.plots: dict[str, pg.PlotItem] = {}
        self.curves: dict[str, pg.PlotDataItem] = {}
        self.overlay: dict[str, pg.PlotDataItem] = {}
        self.names: list[str] = []
        self.groups: dict[str, str] = {}
        self.unit_of: dict[str, str] = {}

    def configure(self, system, valves: list[str]) -> None:
        """Set up one curve per sensor (in its display units) and per valve."""
        self.graphics.clear()
        self.plots.clear()
        self.curves.clear()
        self.overlay.clear()
        self.series_list.blockSignals(True)
        self.series_list.clear()
        self.names = [s.id for s in system.sensors] + [f"valve:{v}" for v in valves]
        for s in system.sensors:
            self.groups[s.id] = s.type if s.type in ("pressure", "temperature", "mass_flow") else "other"
            self.unit_of[s.id] = s.units or units.DEFAULT_UNITS.get(s.type, "")
        for v in valves:
            self.groups[f"valve:{v}"] = "valve"
            self.unit_of[f"valve:{v}"] = ""
        used = [g for g, _ in GROUPS if any(self.groups[n] == g for n in self.names)]
        first = None
        for i, (g, title) in enumerate((g, t) for g, t in GROUPS if g in used):
            p = self.graphics.addPlot(row=i, col=0)
            unit = next((self.unit_of[n] for n in self.names if self.groups[n] == g and self.unit_of[n]), "")
            if g == "valve":
                unit = "%"
                p.setYRange(-5, 105, padding=0)
                p.enableAutoRange(axis="y", enable=False)
            p.setLabel("left", title, units=None if not unit else unit)
            p.showGrid(x=True, y=True, alpha=0.15)
            for side in ("left", "bottom"):
                ax = p.getAxis(side)
                ax.enableAutoSIPrefix(False)  # psi stays psi, not kpsi
                ax.setTickFont(T.mono_font(8))
                ax.setPen(T.GRAY_600)
                ax.setTextPen(T.GRAY_300)
            p.getAxis("left").setWidth(64)
            p.getAxis("bottom").setStyle(showValues=False)
            if first is None:
                first = p
            else:
                p.setXLink(first)
            self.plots[g] = p
        if self.plots:
            last = list(self.plots.values())[-1]
            last.getAxis("bottom").setStyle(showValues=True)
            last.setLabel("bottom", "time", units="s")
        for i, n in enumerate(self.names):
            g = self.groups[n]
            colour = COLOURS[i % len(COLOURS)]
            pen = pg.mkPen(colour, width=1.5)
            self.curves[n] = self.plots[g].plot([], [], pen=pen, name=n.replace("valve:", ""))
            it = QListWidgetItem(_swatch(colour), n.replace("valve:", "valve "))
            it.setFont(T.mono_font())
            it.setData(Qt.UserRole, n)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked if g == "valve" and len(valves) > 6 else Qt.Checked)
            self.series_list.addItem(it)
        self.series_list.blockSignals(False)
        self._rebuild_visibility()

    def _rebuild_visibility(self) -> None:
        for i in range(self.series_list.count()):
            it = self.series_list.item(i)
            n = it.data(Qt.UserRole)
            vis = it.checkState() == Qt.Checked
            if n in self.curves:
                self.curves[n].setVisible(vis)
            if n in self.overlay:
                self.overlay[n].setVisible(vis)

    def update_from(self, history) -> None:
        t, v = history.arrays()
        if len(t) == 0:
            return
        win = self.window_box.currentData()
        if win:
            keep = t >= t[-1] - win
            t, v = t[keep], v[keep]
        for j, n in enumerate(history.names):
            c = self.curves.get(n)
            if c is None or not c.isVisible():
                continue
            y = v[:, j]
            if self.groups.get(n) == "mass_flow":  # display only: residue would fill an autoscaled axis
                y = np.where(np.abs(y) < FLOW_ZERO, 0.0, y)
            elif self.groups.get(n) == "valve":
                y = y * 100.0
            c.setData(t, units.convert(y, self.unit_of.get(n, "")))

    def load_overlay(self, names: list[str], data: np.ndarray) -> int:
        """Overlay columns of a recorded log whose names match sensors (already in display units)."""
        n_match = 0
        for j, name in enumerate(names):
            key = name if name in self.curves else (f"valve:{name[:-4]}" if name.endswith(".pos") else None)
            if key not in self.curves or j == 0:
                continue
            g = self.groups[key]
            pen = pg.mkPen(self.curves[key].opts["pen"].color(), width=1.2, style=Qt.DashLine)
            y = data[:, j] * 100.0 if g == "valve" else data[:, j]  # valve axis is in %
            self.overlay[key] = self.plots[g].plot(data[:, 0], y, pen=pen, name=f"{name} (log)")
            n_match += 1
        self.clear_overlay_btn.setEnabled(bool(self.overlay))
        self._rebuild_visibility()
        return n_match

    def clear_overlay(self) -> None:
        for n, c in self.overlay.items():
            self.plots[self.groups[n]].removeItem(c)
        self.overlay.clear()
        self.clear_overlay_btn.setEnabled(False)
