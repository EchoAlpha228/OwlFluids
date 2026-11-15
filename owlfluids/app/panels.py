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
from ..core.lines import LINE_TYPES, Lines, part_types
from ..fluids.registry import FLUIDS
from ..io.system import Component, Sensor
from . import theme as T
from .schematic import SENSOR_MIME

UNIT_EXAMPLES = {"pressure": "300 psi", "temperature": "-196 C", "length": "0.5 in", "area": "0.1 in2",
                 "volume": "2 L", "time": "50 ms", "mass_flow": "1.2 kg/s"}


def _label(name: str) -> str:
    return name.replace("_", " ")


def _show_error(widget, message: str) -> None:
    """Red field plus the reason, shown right away rather than only in a tooltip."""
    _set_error(widget, True)
    widget.setToolTip(message)
    QToolTip.showText(widget.mapToGlobal(widget.rect().bottomLeft()), message, widget)


def _set_error(widget, on: bool) -> None:
    widget.setProperty("error", on)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def _heading(text: str) -> QLabel:
    h = QLabel(text.upper())
    h.setObjectName("heading")
    return h


def _fluid_heading(fluid: str | None) -> QLabel:
    """A section heading for one fluid's valves, led by its material colour swatch."""
    h = _heading(fluid or "Other")
    if fluid is not None:
        h.setText(f'<span style="color: {T.FLUID_COLOURS.get(fluid, T.GRAY_300)}">&#9632;</span>&nbsp;{fluid.upper()}')
    h.setContentsMargins(0, 6, 0, 0)
    return h


FLUID_PARAMS = ("fluid", "pressurant")
# sensors in the palette, put anywhere on a line: (name, what it reads)
INSTRUMENTS = {
    "pressure": ("Pressure transducer", "Reads the pressure in the line where it sits."),
    "temperature": ("Thermocouple", "Reads the fluid temperature where it sits, with a probe's lag."),
    "mass_flow": ("Mass flow meter", "Reads the mass flow along the line where it sits."),
    "density": ("Density meter", "Reads the fluid density where it sits."),
    "quality": ("Vapor quality", "Reads the vapor mass fraction where it sits (0 liquid, 1 vapor)."),
    "purge_fraction": ("Purge gas fraction", "Reads how much of the fluid where it sits is purge gas."),
}
# a part's own sensors, added from its properties
PART_SENSOR_LABELS = {"level": "level", "mass": "mass (load cell)", "valve_position": "valve position",
                      "wall_temperature": "wall temperature"}
PART_SENSOR_PREFIX = {"level": "LL", "mass": "M", "valve_position": "ZS", "wall_temperature": "TW"}


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
        # sensors are parts of their own: drop one anywhere on a line and it reads what is there
        cat = QTreeWidgetItem(self, ["Sensors"])
        cat.setFlags(Qt.ItemIsEnabled)
        for stype, (label, tip) in INSTRUMENTS.items():
            it = QTreeWidgetItem(cat, [f"{label} ({LINE_TYPES[stype]})"])
            it.setData(0, Qt.UserRole, SENSOR_MIME + stype)
            it.setToolTip(0, tip + " Drop it on a line (or a tank) to read there; drag it to move it.")
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

    def focus_id(self) -> None:
        """Put the cursor in the shown part's or sensor's id, all selected, ready to type a new name."""
        ed = getattr(self, "_ident", None)
        if ed is not None and not ed.isReadOnly():
            ed.setFocus(Qt.ShortcutFocusReason)
            ed.selectAll()

    def _clear(self) -> None:
        self._ident = None
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
            hint.setObjectName("hint")
            self._layout.addWidget(hint)
            return
        ct = get_type(comp.type)
        title = QLabel(ct.label or ct.type_name)
        title.setFont(T.ui_font(12, bold=True))
        self._layout.addWidget(title)
        if ct.help:
            about = QLabel(ct.help)
            about.setWordWrap(True)
            about.setObjectName("hint")
            self._layout.addWidget(about)
        form_w = QWidget()
        form = QFormLayout(form_w)
        ident = QLineEdit(comp.id)
        ident.setReadOnly(read_only)
        ident.setToolTip("Rename: type a new id and press Enter (F2 or double-click the part to jump here)")
        self._ident = ident
        ident.editingFinished.connect(lambda: self._rename(ident))
        form.addRow("id", ident)
        for p in ct.params:
            form.addRow(_label(p.name), self._editor(p, comp, read_only))
        self._layout.addWidget(form_w)
        self._layout.addWidget(_heading("Sensors"))
        for s in [s for s in system.sensors if s.at.split(".")[0] == comp.id]:  # its own, and any on a free port
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            sid = QLabel(s.id)
            T.monospace(sid)
            h.addWidget(sid)
            what = QLabel(f"{_label(s.type)} at {s.at}" + (f" [{s.units}]" if s.units else ""))
            what.setObjectName("hint")
            h.addWidget(what, 1)
            if not read_only:
                rm = QPushButton("remove")
                rm.clicked.connect(lambda _=False, s=s: self._remove_sensor(s))
                h.addWidget(rm)
            self._layout.addWidget(row)
        kinds = part_types(comp.type)
        if not read_only and kinds:
            self._layout.addWidget(self._sensor_adder(comp, kinds))
        hint = QLabel("Pressure, temperature and flow sensors are parts of their own: drag one from the "
                      "palette onto any line.")
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        self._layout.addWidget(hint)

    def show_sensor(self, s: Sensor, system, read_only: bool = False) -> None:
        """A sensor: its id, display units and where it reads."""
        self._clear()
        self.comp, self.system = None, system
        label = INSTRUMENTS.get(s.type, (f"{_label(s.type).capitalize()} sensor",))[0]
        title = QLabel(label)
        title.setFont(T.ui_font(12, bold=True))
        self._layout.addWidget(title)
        what = _label(s.type)
        line = system.line_of(s)
        if line is not None:
            a, b = s.line
            if s.type == "mass_flow":
                where = (f"Reads the flow along the line between {a} and {b} where it sits, positive from "
                         f"{a.split('.')[0]} toward {b.split('.')[0]}.")
                if not Lines(system).flow_readable(*line):
                    where += " This line has length 0 and no valve, orifice or fitting at its ends: it reads nothing."
            else:
                where = f"Reads the {what} on the line between {a} and {b} where it sits."
            where += " Drag it along the line or onto another."
        elif s.at:
            where = f"Reads the {what} at {s.at}. Drag it onto a line to read there."
        else:
            where = "On no line, so it reads nothing. Drag it onto a line (or a tank) to read there."
        about = QLabel(where)
        about.setWordWrap(True)
        about.setObjectName("hint")
        self._layout.addWidget(about)
        form_w = QWidget()
        form = QFormLayout(form_w)
        ident = QLineEdit(s.id)
        ident.setReadOnly(read_only)
        ident.setToolTip("Rename: type a new id and press Enter (F2 or double-click the sensor to jump here)")
        self._ident = ident
        T.monospace(ident)

        def rename():
            new = ident.text().strip()
            if not new or new == s.id:
                return
            if any(o.id == new for o in system.sensors):
                _show_error(ident, f"{new} is already used")
                return
            _set_error(ident, False)
            system.renamed(s.id, new)
            s.id = new
            self.edited.emit()

        ident.editingFinished.connect(rename)
        form.addRow("id", ident)
        unit = QLineEdit(s.units)
        unit.setPlaceholderText(f"default: {units.DEFAULT_UNITS.get(s.type, '')}")
        unit.setReadOnly(read_only)

        def set_unit():
            text = unit.text().strip()
            try:
                units.convert(1.0, text)
            except (ValueError, KeyError) as e:
                _show_error(unit, str(e))
                return
            _set_error(unit, False)
            s.units = text
            self.edited.emit()

        unit.editingFinished.connect(set_unit)
        form.addRow("units", unit)
        self._layout.addWidget(form_w)
        if s.type == "mass_flow" and line is not None and not read_only:
            flip = QPushButton("reverse direction")
            flip.setToolTip("Read positive the other way along the line")

            def reverse():
                s.line = (s.line[1], s.line[0])
                if s.along is not None:
                    s.along = round(1.0 - s.along, 3)
                self.edited.emit()
                self.show_sensor(s, system)

            flip.clicked.connect(reverse)
            self._layout.addWidget(flip)
        if not read_only:
            rm = QPushButton("remove sensor")

            def remove():
                system.sensors.remove(s)
                self.edited.emit()
                self.show_component(None, system)

            rm.clicked.connect(remove)
            self._layout.addWidget(rm)

    def show_line(self, a: str, b: str, system, read_only: bool = False) -> None:
        """A line: the tubing between two ports, its length and inner diameter."""
        self._clear()
        self.comp, self.system = None, system
        title = QLabel("Line")
        title.setFont(T.ui_font(12, bold=True))
        self._layout.addWidget(title)
        about = QLabel(f"Tubing from {a} to {b}. Its fluid has inertia: a long line takes time to get "
                       "flowing, and a fast valve close on it shows water hammer. It has no friction (the "
                       "valves, orifices and fittings set the flow). Length 0 joins parts bolted straight "
                       "together into one point.")
        about.setWordWrap(True)
        about.setObjectName("hint")
        self._layout.addWidget(about)
        form_w = QWidget()
        form = QFormLayout(form_w)
        spec = system.line_spec(a, b)
        for key, setting, tip in (
                ("length", "line_length", "Tubing length; 0 for parts bolted together"),
                ("diameter", "line_diameter", "Inner diameter, or auto: the largest bore of the parts at "
                                              "its ends (at least 1/4 in tubing's 4.6 mm)")):
            cur = spec.get(key)
            ed = QLineEdit("" if cur is None else str(cur))
            ed.setPlaceholderText(f"default: {system.setting(setting)}")
            hint = f"{tip}  (units accepted, e.g. {UNIT_EXAMPLES['length']}; blank uses the {_label(setting)} setting)"
            ed.setToolTip(hint)
            ed.setProperty("hint", hint)
            ed.setReadOnly(read_only)
            ed.editingFinished.connect(lambda e=ed, k=key: self._set_line(a, b, k, e))
            form.addRow(key, ed)
        self._layout.addWidget(form_w)

    def _set_line(self, a: str, b: str, key: str, ed: QLineEdit) -> None:
        text = ed.text().strip()
        if text and not (key == "diameter" and text.lower() == "auto"):
            try:
                v = units.parse(text, "length")
            except (ValueError, TypeError) as e:
                _show_error(ed, str(e))
                return
            if v < 0 or (key == "diameter" and v == 0):
                _show_error(ed, "a length can't be negative" if v < 0 else "a diameter must be above 0")
                return
        _set_error(ed, False)
        ed.setToolTip(ed.property("hint") or "")
        value = None if not text else 0 if text == "0" else text  # a bare 0 saves as length: 0
        if self.system.line_spec(a, b).get(key) == value:
            return
        self.system.set_line_spec(a, b, key, value)
        self.edited.emit()

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
        _set_error(ed, False)
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

    def _sensor_adder(self, comp, kinds) -> QWidget:
        """Add one of the part's own sensors: its level, mass, valve position or wall temperature."""
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        kind = QComboBox()
        for t in kinds:
            kind.addItem(PART_SENSOR_LABELS[t], t)
        add = QPushButton("add sensor")

        def do():
            t = kind.currentData()
            prefix = PART_SENSOR_PREFIX[t]
            n = 1
            while any(s.id == f"{prefix}-{comp.id}" + (f"-{n}" if n > 1 else "") for s in self.system.sensors):
                n += 1
            sid = f"{prefix}-{comp.id}" + (f"-{n}" if n > 1 else "")
            self.system.sensors.append(Sensor(sid, t, comp.id, units=units.DEFAULT_UNITS.get(t, "")))
            self.edited.emit()
            self.show_component(comp, self.system)

        add.clicked.connect(do)
        h.addWidget(kind, 1)
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
        self.manual = manual
        h = QHBoxLayout(self)
        h.setContentsMargins(2, 1, 2, 1)
        self.button = QPushButton(name)
        T.monospace(self.button)
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
        self._state = None
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(True)
        self.bar.setFormat("%p%")
        self.bar.setMaximumHeight(16)
        h.addWidget(self.bar)

    def set_values(self, cmd: float, pos: float) -> None:
        # open or closed, nothing between: a hand valve cracked off its seat is open
        is_open = cmd > 0.001 if self.manual else cmd > 0.5
        self.button.blockSignals(True)
        self.button.setChecked(is_open)  # the button agrees with the colour
        self.button.blockSignals(False)
        state = "open" if is_open else "closed"
        if state != self._state:  # restyling is costly; only when the command flips
            self._state = state
            colour = T.VALVE_OPEN if is_open else T.VALVE_CLOSED
            self.button.setStyleSheet(f"QPushButton {{ background: {colour}; color: {T.WHITE}; font-weight: bold;"
                                      f" border: 1px solid {colour}; }}")
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
        label = T.monospace(QLabel(name))
        label.setMinimumWidth(110)
        label.setToolTip("regulator: its setpoint is set from here only, never by Synnax")
        h.addWidget(label)
        self.box = T.monospace(QDoubleSpinBox())
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

    def set_valves(self, names: list[str], manual: dict[str, bool], regulators: list[str] = (),
                   fluids: dict[str, str | None] | None = None) -> None:
        """One row per valve; with ``fluids`` (valve -> the fluid in its line) grouped under a heading per
        fluid, once there is more than one."""
        body = QWidget()
        v = QVBoxLayout(body)
        v.setAlignment(Qt.AlignTop)
        self.rows = {}
        self.regulators = {}
        if not names and not regulators:
            hint = QLabel("Valves appear here while the simulation runs.")
            hint.setObjectName("hint")
            v.addWidget(hint)
        fluids = fluids or {}
        groups: dict[str | None, list[str]] = {}
        for n in names:
            groups.setdefault(fluids.get(n), []).append(n)
        order = sorted(f for f in groups if f is not None) + ([None] if None in groups else [])
        for f in order:
            if len(groups) > 1:
                v.addWidget(_fluid_heading(f))
            for n in groups[f]:
                row = ValveRow(n, manual.get(n, False), self)
                self.rows[n] = row
                v.addWidget(row)
        if regulators:
            v.addWidget(_heading("Regulators"))
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
