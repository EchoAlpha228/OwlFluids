"""System settings dialog (time step, initial conditions, property backend, fluid models)."""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLineEdit

from .. import units
from ..fluids.registry import FLUIDS
from ..io.system import DEFAULT_SETTINGS
from .panels import UNIT_EXAMPLES, _show_error

LABELS = {"dt": "time step (dt)"}


class SettingsDialog(QDialog):
    def __init__(self, system, parent=None):
        super().__init__(parent)
        self.setWindowTitle("System settings")
        self.system = system
        form = QFormLayout(self)
        self.name = QLineEdit(system.name)
        form.addRow("name", self.name)
        self.fields = {}
        for key, dim in (("dt", "time"), ("initial_pressure", "pressure"), ("initial_temperature", "temperature"),
                         ("junction_volume", "volume")):
            ed = QLineEdit(str(system.setting(key)))
            ed.setToolTip(f"default: {DEFAULT_SETTINGS[key]}; units accepted, e.g. {UNIT_EXAMPLES[dim]}")
            self.fields[key] = (ed, dim)
            form.addRow(LABELS.get(key, key.replace("_", " ")), ed)
        self.backend = QComboBox()
        self.backend.addItems(["HEOS", "REFPROP"])
        self.backend.setCurrentText(str(system.setting("backend")))
        self.backend.setToolTip("REFPROP needs a local REFPROP install that CoolProp can find")
        form.addRow("property backend", self.backend)
        self.prime = QCheckBox("start lines filled from the tank they are open to")
        self.prime.setChecked(bool(system.setting("prime_lines")))
        form.addRow("prime lines", self.prime)
        self.noise = QCheckBox("realistic noise and lag by sensor type (off: exact values)")
        self.noise.setChecked(str(system.setting("sensor_noise")).lower() != "off")
        self.noise.setToolTip("Applies only to what sensors read; the simulation itself is unchanged")
        form.addRow("sensor noise", self.noise)
        self.models = {}
        cfg = dict(system.setting("fluid_models") or {})
        for name, (_cp, default, allowed) in FLUIDS.items():
            if len(allowed) < 2:
                continue
            box = QComboBox()
            box.addItems(list(allowed))
            box.setCurrentText(cfg.get(name, default))
            self.models[name] = (box, default)
            form.addRow(f"{name} model", box)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def _ok(self) -> None:
        for ed, dim in self.fields.values():
            try:
                units.parse(ed.text(), dim)
            except (ValueError, TypeError) as e:
                _show_error(ed, str(e))
                ed.setFocus()
                return
            ed.setStyleSheet("")
        s = self.system.settings
        self.system.name = self.name.text().strip() or self.system.name
        for key, (ed, _dim) in self.fields.items():
            s[key] = ed.text().strip()
        s["backend"] = self.backend.currentText()
        s["prime_lines"] = self.prime.isChecked()
        s["sensor_noise"] = "realistic" if self.noise.isChecked() else "off"
        s["fluid_models"] = {n: b.currentText() for n, (b, d) in self.models.items() if b.currentText() != d}
        self.accept()
