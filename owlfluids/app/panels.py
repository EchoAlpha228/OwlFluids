"""Side panels: component palette, properties editor, sensors, valve controls."""

from __future__ import annotations

from PySide6.QtCore import QMimeData, Qt, Signal
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QToolTip,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import units
from ..components import TYPES, get_type
from ..fluids.registry import FLUIDS
from ..io.system import Component, Sensor

UNIT_EXAMPLES = {"pressure": "300 psi", "temperature": "-196 C", "length": "0.5 in", "area": "0.1 in2",
                 "volume": "2 L", "time": "50 ms", "mass_flow": "1.2 kg/s"}


def _label(name: str) -> str:
    return name.replace("_", " ")


def _show_error(widget, message: str) -> None:
    """Pink field plus the reason, shown right away rather than only in a tooltip."""
    widget.setStyleSheet("background:#fde2e2")
    widget.setToolTip(message)
    QToolTip.showText(widget.mapToGlobal(widget.rect().bottomLeft()), message, widget)


FLUID_PARAMS = ("fluid", "liquid", "pressurant")
SENSOR_TYPES = ("pressure", "temperature", "mass_flow", "valve_position", "quality", "density", "mass",
                "wall_temperature", "level", "purge_fraction")


class Palette(QTreeWidget):
    """Component types grouped by category; drag one onto the schematic or double-click."""

    add_requested = Signal(str)

    def __init__(self):
        super().__init__()
        self.setHeaderHidden(True)
        self.setDragEnabled(True)
        cats: dict[str, QTreeWidgetItem] = {}
        for name, ct in TYPES.items():
            if name != ct.type_name:
                continue
            cat = cats.get(ct.category)
            if cat is None:
                cat = cats[ct.category] = QTreeWidgetItem(self, [ct.category])
                cat.setFlags(Qt.ItemIsEnabled)
            it = QTreeWidgetItem(cat, [ct.label or name])
            it.setData(0, Qt.UserRole, name)
            it.setToolTip(0, ct.help or (ct.__doc__ or "").strip().split("\n")[0])
        self.expandAll()
        self.itemDoubleClicked.connect(self._double)

    def _double(self, item, col):
        name = item.data(0, Qt.UserRole)
        if name:
            self.add_requested.emit(name)

    def startDrag(self, actions):
        item = self.currentItem()
        name = item.data(0, Qt.UserRole) if item else None
        if not name:
            return
        mime = QMimeData()
        mime.setText(name)
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.CopyAction)


class Properties(QScrollArea):
    """Edits the selected component's id and parameters, and the sensors on it."""

    edited = Signal()
    rename_requested = Signal(str, str)

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.comp: Component | None = None
        self.system = None
        self._body = QWidget()
        self.setWidget(self._body)
        self._layout = QVBoxLayout(self._body)
        self._layout.setAlignment(Qt.AlignTop)
        self.show_component(None, None)

    def _clear(self) -> None:
        while self._layout.count():
            w = self._layout.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def show_component(self, comp: Component | None, system, read_only: bool = False) -> None:
        self._clear()
        self.comp, self.system = comp, system
        if comp is None:
            hint = QLabel("Select a component to edit it.\n\nDrag parts from the palette, then drag from one "
                          "port (circle) to another to connect them. R rotates, Delete removes.")
            hint.setWordWrap(True)
            self._layout.addWidget(hint)
            return
        ct = get_type(comp.type)
        title = QLabel(f"<b>{ct.label or ct.type_name}</b>")
        self._layout.addWidget(title)
        if ct.help:
            about = QLabel(ct.help)
            about.setWordWrap(True)
            about.setStyleSheet("color: grey")
            self._layout.addWidget(about)
        form_w = QWidget()
        form = QFormLayout(form_w)
        ident = QLineEdit(comp.id)
        ident.setReadOnly(read_only)
        ident.editingFinished.connect(lambda: self._rename(ident))
        form.addRow("id", ident)
        for p in ct.params:
            form.addRow(_label(p.name), self._editor(p, comp, read_only))
        self._layout.addWidget(form_w)
        self._layout.addWidget(QLabel("<b>Sensors</b>"))
        for s in [s for s in system.sensors if s.at.split(".")[0] == comp.id]:
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(QLabel(f"{s.id}: {s.type} at {s.at}" + (f" [{s.units}]" if s.units else "")))
            if not read_only:
                rm = QPushButton("remove")
                rm.clicked.connect(lambda _=False, s=s: self._remove_sensor(s))
                h.addWidget(rm)
            self._layout.addWidget(row)
        if not read_only:
            self._layout.addWidget(self._sensor_adder(comp, ct))

    def _editor(self, p, comp, read_only):
        cur = comp.params.get(p.name)
        if isinstance(cur, bool) and p.choices == ("on", "off"):
            cur = "on" if cur else "off"  # YAML reads a bare on/off as a bool
        if p.name in FLUID_PARAMS or p.choices:
            box = QComboBox()
            box.setEditable(p.name in FLUID_PARAMS)
            opts = list(FLUIDS) if p.name in FLUID_PARAMS else list(p.choices)
            box.addItem("" if p.default is None else f"(default: {p.default})", None)
            for o in opts:
                box.addItem(o, o)
            if cur is not None:
                i = box.findData(cur)
                if i < 0:
                    box.addItem(str(cur), cur)
                    i = box.count() - 1
                box.setCurrentIndex(i)
            box.setEnabled(not read_only)
            box.setToolTip(p.help)
            box.currentIndexChanged.connect(lambda _i, b=box, n=p.name: self._set(n, b.currentData()))
            return box
        ed = QLineEdit("" if cur is None else str(cur))
        ed.setPlaceholderText("" if p.default is None else f"default: {p.default}")
        hint = ""
        if p.dimension in UNIT_EXAMPLES:
            hint = f"  ({p.dimension}; units accepted, e.g. {UNIT_EXAMPLES[p.dimension]})"
        ed.setToolTip(p.help + hint)
        ed.setProperty("hint", p.help + hint)
        ed.setReadOnly(read_only)
        ed.editingFinished.connect(lambda e=ed, prm=p: self._set_text(prm, e))
        return ed

    def _set_text(self, p, ed: QLineEdit) -> None:
        text = ed.text().strip()
        if text and p.dimension and p.dimension not in ("text", "choice", "bool"):
            try:
                units.parse(text, p.dimension)
            except (ValueError, TypeError) as e:
                _show_error(ed, str(e))
                return
        ed.setStyleSheet("")
        ed.setToolTip(ed.property("hint") or "")
        value: object = text
        try:
            value = float(text) if text and p.dimension is None else text
            if isinstance(value, float) and value.is_integer() and "." not in text:
                value = int(value)
        except ValueError:
            value = text
        self._set(p.name, value if text else None)

    def _set(self, name, value) -> None:
        if self.comp is None:
            return
        if value is None or value == "":
            self.comp.params.pop(name, None)
        else:
            self.comp.params[name] = value
        self.edited.emit()

    def _rename(self, ed: QLineEdit) -> None:
        new = ed.text().strip()
        if self.comp is not None and new and new != self.comp.id:
            self.rename_requested.emit(self.comp.id, new)

    def _sensor_adder(self, comp, ct) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        kind = QComboBox()
        for t in SENSOR_TYPES:
            kind.addItem(t)
        where = QComboBox()
        if ct.ports:
            for port in ct.ports:
                where.addItem(f"{comp.id}.{port}")
        where.addItem(comp.id)
        add = QPushButton("add sensor")

        def do():
            t = kind.currentText()
            at = comp.id if t in ("mass_flow", "valve_position", "mass", "level") else where.currentText()
            prefix = {"pressure": "PT", "temperature": "TC", "mass_flow": "FM", "valve_position": "ZS",
                      "mass": "M", "level": "LL", "purge_fraction": "PF"}.get(t, t[:2].upper())
            n = 1
            while any(s.id == f"{prefix}-{comp.id}" + (f"-{n}" if n > 1 else "") for s in self.system.sensors):
                n += 1
            sid = f"{prefix}-{comp.id}" + (f"-{n}" if n > 1 else "")
            self.system.sensors.append(Sensor(sid, t, at, units=units.DEFAULT_UNITS.get(t, "")))
            self.edited.emit()
            self.show_component(comp, self.system)

        add.clicked.connect(do)
        h.addWidget(kind)
        h.addWidget(where)
        h.addWidget(add)
        return w

    def _remove_sensor(self, s: Sensor) -> None:
        self.system.sensors.remove(s)
        self.edited.emit()
        self.show_component(self.comp, self.system)


class ValveRow(QWidget):
    def __init__(self, name: str, manual: bool, panel: ValvePanel):
        super().__init__()
        self.name = name
        h = QHBoxLayout(self)
        h.setContentsMargins(2, 1, 2, 1)
        self.button = QPushButton(name)
        self.button.setCheckable(True)
        self.button.setMinimumWidth(110)
        self.button.toggled.connect(lambda on: panel.commanded.emit(name, 1.0 if on else 0.0))
        h.addWidget(self.button)
        self.slider = None
        if manual:  # a hand valve can be cracked to any opening
            self.button.setToolTip("manual valve: worked from here only, never by Synnax")
            self.slider = QSlider(Qt.Horizontal)
            self.slider.setRange(0, 100)
            self.slider.setMaximumWidth(90)
            # every user change (drag, groove click, keys, wheel); set_values blocks its own updates
            self.slider.valueChanged.connect(lambda v: panel.commanded.emit(name, v / 100.0))
            h.addWidget(self.slider)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(True)
        self.bar.setFormat("%p%")
        self.bar.setMaximumHeight(16)
        h.addWidget(self.bar)

    def set_values(self, cmd: float, pos: float) -> None:
        self.button.blockSignals(True)
        self.button.setChecked(cmd > 0.5)
        self.button.blockSignals(False)
        colour = "#3fae5a" if cmd > 0.5 else "#d64545"
        self.button.setStyleSheet(f"QPushButton {{ background: {colour}; color: white; font-weight: bold; }}")
        if self.slider is not None and not self.slider.isSliderDown():
            self.slider.blockSignals(True)
            self.slider.setValue(int(round(cmd * 100)))
            self.slider.blockSignals(False)
        self.bar.setValue(int(round(pos * 100)))


class RegulatorRow(QWidget):
    """A regulator's knob: its outlet setpoint in psig, applied when the operator finishes typing or steps it."""

    def __init__(self, name: str, panel: ValvePanel):
        super().__init__()
        self.name = name
        h = QHBoxLayout(self)
        h.setContentsMargins(2, 1, 2, 1)
        label = QLabel(name)
        label.setMinimumWidth(110)
        label.setToolTip("regulator: its setpoint is set from here only, never by Synnax")
        h.addWidget(label)
        self.box = QDoubleSpinBox()
        self.box.setRange(0.0, 20000.0)
        self.box.setDecimals(1)
        self.box.setSingleStep(5.0)
        self.box.setSuffix(" psig")
        self.box.setKeyboardTracking(False)  # one setpoint per edit, not one per keystroke
        self.box.valueChanged.connect(lambda v: panel.setpoint_changed.emit(name, units.parse(f"{v} psig")))
        self._typing = False  # mid-edit: leave the box alone; once applied, show the sim's setpoint again
        self.box.lineEdit().textEdited.connect(lambda _t: setattr(self, "_typing", True))
        self.box.editingFinished.connect(lambda: setattr(self, "_typing", False))
        h.addWidget(self.box)
        h.addStretch(1)

    def set_value(self, setpoint: float) -> None:
        if self._typing:
            return  # don't fight the operator's typing
        self.box.blockSignals(True)
        self.box.setValue(units.convert(setpoint, "psig"))
        self.box.blockSignals(False)


class ValvePanel(QScrollArea):
    """One button per valve: click to toggle; manual valves also get a slider, regulators a setpoint box."""

    commanded = Signal(str, float)
    setpoint_changed = Signal(str, float)  # regulator, Pa absolute

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.rows: dict[str, ValveRow] = {}
        self.regulators: dict[str, RegulatorRow] = {}
        self.set_valves([], {})

    def set_valves(self, names: list[str], manual: dict[str, bool], regulators: list[str] = ()) -> None:
        body = QWidget()
        v = QVBoxLayout(body)
        v.setAlignment(Qt.AlignTop)
        self.rows = {}
        self.regulators = {}
        if not names and not regulators:
            v.addWidget(QLabel("Valves appear here while the simulation runs."))
        for n in names:
            row = ValveRow(n, manual.get(n, False), self)
            self.rows[n] = row
            v.addWidget(row)
        if regulators:
            v.addWidget(QLabel("Regulators"))
        for n in regulators:
            row = RegulatorRow(n, self)
            self.regulators[n] = row
            v.addWidget(row)
        self.setWidget(body)

    def show_frame(self, frame) -> None:
        for n, row in self.rows.items():
            row.set_values(frame.commands.get(n, 0.0), frame.valves.get(n, 0.0))
        for n, row in self.regulators.items():
            if n in frame.setpoints:
                row.set_value(frame.setpoints[n])
