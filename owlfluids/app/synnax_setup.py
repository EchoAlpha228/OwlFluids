"""Synnax setup: the server and login, the test mode, and which channel each sensor and valve uses.

The host and login are the user's (kept in the app's own settings, so they never
land in a system file kept in git); everything else is saved with the system, in
its ``synnax:`` section, where ``owlfluids run --synnax`` finds it too.
"""

from __future__ import annotations

import threading

import yaml
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import units
from ..components import get_type
from ..drivers import synnax as sx
from . import prefs
from . import theme as T
from .panels import _heading, _set_error, _show_error

MODE_TEXT = {"hootl": "HOOTL: no hardware, the sim fakes valve states",
             "hitl": "HITL: real valves, read their states from the hardware"}
STATE_TEXT = {"command": "echoes the command (digital output task)",
              "position": "reports position (limit switches)"}
CHECK_TEXT = {"warn": "warn", "strict": "refuse to connect", "off": "don't check"}


def load_config(system) -> tuple[sx.SynnaxConfig, str]:
    """This system's Synnax settings with this user's connection, and a problem reading them ('' if none)."""
    d = {**dict(system.synnax or {}), **prefs.synnax_connection()}
    try:
        return sx.SynnaxConfig.from_dict(d), ""
    except (TypeError, ValueError) as e:
        return sx.SynnaxConfig.from_dict(prefs.synnax_connection()), f"The saved Synnax settings were not read: {e}"


def system_parts(system) -> tuple[list, list[str], list[str]]:
    """(sensors, valves automation may command, parts only the operator works: manual valves and regulators)."""
    auto, hand = [], []
    for c in system.components:
        ct = get_type(c.type)
        if ct.is_valve:
            (auto if ct.automated else hand).append(c.id)
        elif ct.type_name == "regulator":
            hand.append(c.id)
    return list(system.sensors), auto, hand


def _mono_item(text: str, editable: bool = True) -> QTableWidgetItem:
    it = QTableWidgetItem(text)
    it.setFont(T.mono_font(9))
    if not editable:
        it.setFlags(it.flags() & ~Qt.ItemIsEditable)
    return it


def _tick_item(on: bool) -> QTableWidgetItem:
    it = QTableWidgetItem()
    it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
    it.setCheckState(Qt.Checked if on else Qt.Unchecked)
    it.setToolTip("ticked: this one gets Synnax channels")
    return it


def _table(headers: list[str]) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().hide()
    t.setSelectionMode(QAbstractItemView.NoSelection)
    t.setEditTriggers(QAbstractItemView.AllEditTriggers)
    t.setShowGrid(False)
    t.setFocusPolicy(Qt.StrongFocus)
    h = t.horizontalHeader()
    h.setSectionResizeMode(QHeaderView.Interactive)
    h.setSectionResizeMode(0, QHeaderView.Fixed)
    t.setColumnWidth(0, 28)
    h.setStretchLastSection(True)
    h.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    return t


class SynnaxDialog(QDialog):
    """Run > Synnax setup (Save keeps it) and Run > Connect to Synnax (Connect keeps it and connects)."""

    link_factory = None  # tests swap in a stand-in for the Synnax client; None means the real one

    def __init__(self, system, parent=None, connect: bool = False):
        super().__init__(parent)
        self.setWindowTitle("Connect to Synnax" if connect else "Synnax setup")
        self.system = system
        self.connecting = connect
        cfg, problem = load_config(system)
        self._spec = {k: dict(v or {}) for k, v in cfg.valves.items()}  # keeps hand-written read/write overrides
        self._probe = None
        self._probe_token = None
        self.result_config: sx.SynnaxConfig | None = None
        self.changed = False  # Save or Connect changed what the system file keeps

        root = QVBoxLayout(self)
        cols = QHBoxLayout()
        root.addLayout(cols, 1)
        left = QVBoxLayout()
        cols.addLayout(left)
        cols.addSpacing(18)
        right = QVBoxLayout()
        cols.addLayout(right, 1)

        # ---- connection
        left.addWidget(_heading("Connection"))
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        self.host = QLineEdit(cfg.host)
        self.host.setPlaceholderText("localhost")
        self.port = QLineEdit(str(cfg.port))
        self.port.setFixedWidth(64)
        T.monospace(self.port)
        server = QWidget()
        sl = QHBoxLayout(server)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(self.host, 1)
        sl.addWidget(QLabel(":"))
        sl.addWidget(self.port)
        form.addRow("server", server)
        self.username = QLineEdit(cfg.username)
        form.addRow("username", self.username)
        self.password = QLineEdit(cfg.password)
        self.password.setEchoMode(QLineEdit.Password)
        form.addRow("password", self.password)
        self.secure = QCheckBox("TLS (the Core runs with certificates)")
        self.secure.setChecked(cfg.secure)
        form.addRow("secure", self.secure)
        left.addLayout(form)
        note = QLabel("Kept in your own settings on this computer, not in the system file.")
        note.setObjectName("hint")
        note.setWordWrap(True)
        left.addWidget(note)
        trow = QHBoxLayout()
        self.test_button = QPushButton("Test connection")
        self.test_button.clicked.connect(self.test_connection)
        trow.addWidget(self.test_button)
        trow.addStretch(1)
        left.addLayout(trow)
        self.test_status = QLabel("")
        self.test_status.setWordWrap(True)
        self.test_status.setTextFormat(Qt.RichText)
        self.test_status.setMinimumHeight(34)
        self.test_status.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        left.addWidget(self.test_status)

        # ---- the test
        left.addSpacing(8)
        left.addWidget(_heading("Test"))
        form = QFormLayout()
        self.mode = QComboBox()
        for k, text in MODE_TEXT.items():
            self.mode.addItem(text, k)
        self.mode.setCurrentIndex(sx.MODES.index(cfg.mode))
        form.addRow("mode", self.mode)
        self.valve_state = QComboBox()
        for k, text in STATE_TEXT.items():
            self.valve_state.addItem(text, k)
        self.valve_state.setCurrentIndex(sx.VALVE_STATES.index(cfg.valve_state))
        self.valve_state.setToolTip("What a valve state channel means on the stand. Keep 'echoes the command' "
                                    "unless HITL has real position feedback.")
        form.addRow("valve state", self.valve_state)
        self.rate = QLineEdit(f"{cfg.rate:g}")
        T.monospace(self.rate)
        self.rate.setToolTip("How often the sensors are written to Synnax, Hz")
        form.addRow("rate (Hz)", self.rate)
        self.index = QLineEdit(cfg.index)
        T.monospace(self.index)
        self.index.setToolTip("Index channel for the channels OwlFluids creates; existing channels keep theirs")
        form.addRow("index channel", self.index)
        self.create = QCheckBox("create channels Synnax doesn't have yet")
        self.create.setChecked(cfg.create_channels)
        form.addRow("channels", self.create)
        self.check = QComboBox()
        for k, text in CHECK_TEXT.items():
            self.check.addItem(text, k)
        self.check.setCurrentIndex(sx.VERSION_CHECKS.index(cfg.check_version))
        self.check.setToolTip("When the Synnax Core and this app's Synnax client differ in major.minor version")
        form.addRow("version mismatch", self.check)
        left.addLayout(form)
        left.addStretch(1)

        # ---- channels
        sensors, valves, hand = system_parts(system)
        skip = set(cfg.exclude)
        right.addWidget(_heading("Sensors"))
        self.sensor_table = _table(["", "sensor", "units", "channel"])
        self.sensor_table.setColumnWidth(1, 110)
        self.sensor_table.setColumnWidth(2, 60)
        for s in sensors:
            r = self.sensor_table.rowCount()
            self.sensor_table.insertRow(r)
            listed = not cfg.sensors or s.id in cfg.sensors
            self.sensor_table.setItem(r, 0, _tick_item(listed and s.id not in skip))
            self.sensor_table.setItem(r, 1, _mono_item(s.id, False))
            unit = QTableWidgetItem(s.units or units.DEFAULT_UNITS.get(s.type, ""))
            unit.setFlags(unit.flags() & ~Qt.ItemIsEditable)
            unit.setForeground(QColor(T.TEXT_DIM))
            unit.setToolTip(f"{s.type.replace('_', ' ')} at {s.at}; written to Synnax in these units")
            self.sensor_table.setItem(r, 2, unit)
            self.sensor_table.setItem(r, 3, _mono_item(cfg.sensors.get(s.id, sx.channel_name(s.id))))
        right.addWidget(self.sensor_table, 3)

        right.addSpacing(6)
        right.addWidget(_heading("Valves"))
        self.valve_table = _table(["", "valve", "command channel", "state channel"])
        self.valve_table.setColumnWidth(1, 110)
        self.valve_table.setColumnWidth(2, 170)
        for v in valves:
            r = self.valve_table.rowCount()
            self.valve_table.insertRow(r)
            spec = self._spec.get(v, {})
            listed = not cfg.valves or v in cfg.valves
            self.valve_table.setItem(r, 0, _tick_item(listed and v not in skip))
            self.valve_table.setItem(r, 1, _mono_item(v, False))
            self.valve_table.setItem(r, 2, _mono_item(spec.get("cmd", f"{sx.channel_name(v)}_cmd")))
            self.valve_table.setItem(r, 3, _mono_item(spec.get("state", f"{sx.channel_name(v)}_state")))
        right.addWidget(self.valve_table, 2)
        for t in (self.sensor_table, self.valve_table):
            t.setMinimumHeight(min(t.rowCount(), 4) * 26 + 30)
        if hand:
            h = QLabel(f"No channels for {', '.join(hand)}: manual valves and regulators are worked "
                       "from the OwlFluids window only.")
            h.setObjectName("hint")
            h.setWordWrap(True)
            right.addWidget(h)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        right.addWidget(self.summary)

        # ---- buttons
        self.error = QLabel("")
        self.error.setStyleSheet(f"color: {T.WHITE}; background: {T.DARK_RED}; padding: 4px 6px;")
        self.error.setWordWrap(True)
        self.error.hide()
        root.addWidget(self.error)
        bottom = QHBoxLayout()
        b = QPushButton("Load file…")
        b.setToolTip("Fill this in from a YAML file with a synnax: section, such as examples/synnax.yaml")
        b.clicked.connect(self.load_file)
        bottom.addWidget(b)
        b = QPushButton("Save file…")
        b.setToolTip("Write these settings, login included, to a YAML file for owlfluids run --synnax")
        b.clicked.connect(self.save_file)
        bottom.addWidget(b)
        bottom.addStretch(1)
        b = QPushButton("Cancel")
        b.clicked.connect(self.reject)
        bottom.addWidget(b)
        self.ok_button = QPushButton("Connect" if connect else "Save")
        self.ok_button.setDefault(True)
        self.ok_button.clicked.connect(self._ok)
        bottom.addWidget(self.ok_button)
        root.addLayout(bottom)

        self.mode.currentIndexChanged.connect(self._refresh)
        self.rate.textChanged.connect(self._refresh)
        for t in (self.sensor_table, self.valve_table):
            t.itemChanged.connect(self._refresh)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_probe)
        self._refresh()
        self.resize(940, 620)
        if problem:
            self._show_problem(problem)

    # ------------------------------------------------------------------ reading the form
    def _rows(self, table: QTableWidget):
        for r in range(table.rowCount()):
            yield (table.item(r, 0).checkState() == Qt.Checked, table.item(r, 1).text(),
                   *(table.item(r, c).text().strip() for c in range(2, table.columnCount())))

    def config(self) -> sx.SynnaxConfig:
        """The settings as filled in. Raises ValueError for a value that can't be used."""
        try:
            port = int(self.port.text().strip())
            if not 0 < port < 65536:
                raise ValueError
        except ValueError:
            raise ValueError("port: a whole number, 1 to 65535") from None
        try:
            rate = float(self.rate.text().strip())
            if rate <= 0:
                raise ValueError
        except ValueError:
            raise ValueError("rate: a number of Hz above 0") from None
        sensors, valves, exclude = {}, {}, []
        all_default = True
        for on, sid, _unit, ch in self._rows(self.sensor_table):
            if not on:
                exclude.append(sid)
                continue
            sensors[sid] = ch
            all_default &= ch == sx.channel_name(sid)
        for on, v, cmd, state in self._rows(self.valve_table):
            if not on:
                exclude.append(v)
                continue
            valves[v] = {**self._spec.get(v, {}), "cmd": cmd, "state": state}
            all_default &= (cmd, state) == (f"{sx.channel_name(v)}_cmd", f"{sx.channel_name(v)}_state")
            all_default &= set(valves[v]) == {"cmd", "state"}
        if all_default:  # the defaults: leave the lists out so new sensors and valves are included
            sensors, valves = {}, {}
        return sx.SynnaxConfig(
            host=self.host.text().strip() or "localhost", port=port, username=self.username.text(),
            password=self.password.text(), secure=self.secure.isChecked(), mode=self.mode.currentData(),
            valve_state=self.valve_state.currentData(), rate=rate, index=self.index.text().strip(),
            create_channels=self.create.isChecked(), check_version=self.check.currentData(), sensors=sensors,
            valves=valves, exclude=exclude)

    def channels(self) -> tuple[list[str], list[str]]:
        """(channels written to Synnax, channels read from it) for the ticked rows and the mode."""
        hitl = self.mode.currentData() == "hitl"
        out = [ch for on, _sid, _u, ch in self._rows(self.sensor_table) if on]
        read = []
        for on, _v, cmd, state in self._rows(self.valve_table):
            if not on:
                continue
            if hitl:
                read.append(state)
            else:
                read.append(cmd)
                out.append(state)
        return out, read

    def _problems(self) -> tuple[str, object] | None:
        """The first thing wrong with the form, and the widget or table cell it is in."""
        if not self.index.text().strip():
            return "index channel: a channel name", self.index
        seen: dict[str, str] = {}
        for table, cols in ((self.sensor_table, (3,)), (self.valve_table, (2, 3))):
            for r in range(table.rowCount()):
                if table.item(r, 0).checkState() != Qt.Checked:
                    continue
                for c in cols:
                    if c == 2 and table is self.valve_table and self.mode.currentData() == "hitl":
                        continue  # HITL doesn't use the command channel
                    item = table.item(r, c)
                    name = item.text().strip()
                    who = table.item(r, 1).text()
                    if not name:
                        return f"{who}: give it a channel name, or untick it", item
                    if name in seen and seen[name] != who:
                        return f"{name} is used by both {seen[name]} and {who}", item
                    seen.setdefault(name, who)
        return None

    # ------------------------------------------------------------------ live hints
    def _refresh(self, *_args) -> None:
        hitl = self.mode.currentData() == "hitl"
        for table in (self.sensor_table, self.valve_table):
            table.blockSignals(True)
            for r in range(table.rowCount()):
                on = table.item(r, 0).checkState() == Qt.Checked
                for c in range(1, table.columnCount()):
                    item = table.item(r, c)
                    unused = not on or (table is self.valve_table and c == 2 and hitl)
                    item.setForeground(QColor(T.TEXT_OFF if unused else (T.TEXT_DIM if c == 2 and table is
                                                                          self.sensor_table else T.TEXT)))
                    item.setBackground(QColor(0, 0, 0, 0))
            table.blockSignals(False)
        self.valve_table.horizontalHeaderItem(2).setText("command (unused in HITL)" if hitl else "command channel")
        n_s = sum(1 for on, *_ in self._rows(self.sensor_table) if on)
        n_v = sum(1 for on, *_ in self._rows(self.valve_table) if on)
        rate = self.rate.text().strip() or "?"

        def n(k, what):
            return f"{k} {what}" + ("" if k == 1 else "s")

        if hitl:
            text = (f"Writes {n(n_s, 'sensor')} to Synnax at {rate} Hz; reads {n(n_v, 'valve state')} "
                    "from the hardware and never writes them.")
        else:
            text = (f"Writes {n(n_s, 'sensor')} and {n(n_v, 'faked valve state')} to Synnax at {rate} Hz; "
                    f"reads {n(n_v, 'valve command')}.")
        self.summary.setText(text)
        self.error.hide()

    def _show_problem(self, message: str, where=None) -> None:
        self.error.setText(message)
        self.error.show()
        if isinstance(where, QLineEdit):
            _show_error(where, message)
            where.setFocus()
        elif isinstance(where, QTableWidgetItem):
            table = where.tableWidget()
            table.blockSignals(True)
            where.setBackground(QColor(T.DARK_RED))
            where.setForeground(QColor(T.WHITE))
            where.setToolTip(message)
            table.blockSignals(False)
            table.scrollToItem(where)

    def _check(self) -> sx.SynnaxConfig | None:
        """The settings, or None after showing what is wrong with them."""
        self._refresh()
        for ed in (self.port, self.rate, self.index):
            _set_error(ed, False)
        bad = self._problems()
        if bad:
            self._show_problem(*bad)
            return None
        try:
            return self.config()
        except ValueError as e:
            msg = str(e)
            self._show_problem(msg, self.port if msg.startswith("port") else self.rate)
            return None

    # ------------------------------------------------------------------ test connection
    def test_connection(self) -> None:
        cfg = self._check()
        if cfg is None:
            return
        out, read = self.channels()
        token = self._probe_token = object()
        self._probe = None
        factory = self.link_factory or sx.SynnaxLink

        def run():
            try:
                res = sx.probe(cfg, [*out, *read], factory)
            except Exception as e:  # noqa: BLE001 - any failure to connect is shown in the dialog
                res = e
            if self._probe_token is token:
                self._probe = (cfg, res)

        self.test_button.setEnabled(False)
        self.test_button.setText("Testing…")
        self._set_status(T.GRAY_300, f"Connecting to {cfg.host}:{cfg.port}…")
        threading.Thread(target=run, daemon=True, name="synnax-test").start()
        self._timer.start(100)

    def _poll_probe(self) -> None:
        if self._probe is None:
            return
        self._timer.stop()
        cfg, res = self._probe
        self._probe = None
        self.test_button.setEnabled(True)
        self.test_button.setText("Test connection")
        if isinstance(res, Exception):
            self._set_status(T.RED, f"Could not connect to {cfg.host}:{cfg.port}: {res}")
            return
        warn = list(res.warnings)
        if res.missing and not cfg.create_channels:
            warn.append("tick 'create channels' or add them in Synnax before connecting")
        text = res.summary(cfg.create_channels)
        if warn:
            text += ". " + " ".join(w.removeprefix("Synnax: ") for w in warn)
        self._set_status(T.AMBER if warn else T.GREEN, text)

    def _set_status(self, colour: str, text: str) -> None:
        esc = text.replace("&", "&amp;").replace("<", "&lt;")
        self.test_status.setText(f'<span style="color: {colour}">&#9632;</span>&nbsp;{esc}')

    # ------------------------------------------------------------------ files
    def load_file(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(self, "Synnax settings (a YAML file with a synnax: section)",
                                            prefs.last_dir(), "YAML (*.yaml *.yml)")
        if not fn:
            return
        try:
            cfg = sx.SynnaxConfig.load(fn)
        except Exception as e:  # noqa: BLE001 - a bad file is reported
            QMessageBox.warning(self, "Synnax", f"Could not read {fn}: {e}")
            return
        self.fill(cfg)

    def fill(self, cfg: sx.SynnaxConfig) -> None:
        """Show these settings; sensors and valves the system doesn't have are left out."""
        self.host.setText(cfg.host)
        self.port.setText(str(cfg.port))
        self.username.setText(cfg.username)
        self.password.setText(cfg.password)
        self.secure.setChecked(cfg.secure)
        self.mode.setCurrentIndex(sx.MODES.index(cfg.mode))
        self.valve_state.setCurrentIndex(sx.VALVE_STATES.index(cfg.valve_state))
        self.rate.setText(f"{cfg.rate:g}")
        self.index.setText(cfg.index)
        self.create.setChecked(cfg.create_channels)
        self.check.setCurrentIndex(sx.VERSION_CHECKS.index(cfg.check_version))
        self._spec = {k: dict(v or {}) for k, v in cfg.valves.items()}
        skip = set(cfg.exclude)
        for t in (self.sensor_table, self.valve_table):
            t.blockSignals(True)
        for r in range(self.sensor_table.rowCount()):
            sid = self.sensor_table.item(r, 1).text()
            on = (not cfg.sensors or sid in cfg.sensors) and sid not in skip
            self.sensor_table.item(r, 0).setCheckState(Qt.Checked if on else Qt.Unchecked)
            self.sensor_table.item(r, 3).setText(cfg.sensors.get(sid, sx.channel_name(sid)))
        for r in range(self.valve_table.rowCount()):
            v = self.valve_table.item(r, 1).text()
            spec = self._spec.get(v, {})
            on = (not cfg.valves or v in cfg.valves) and v not in skip
            self.valve_table.item(r, 0).setCheckState(Qt.Checked if on else Qt.Unchecked)
            self.valve_table.item(r, 2).setText(spec.get("cmd", f"{sx.channel_name(v)}_cmd"))
            self.valve_table.item(r, 3).setText(spec.get("state", f"{sx.channel_name(v)}_state"))
        for t in (self.sensor_table, self.valve_table):
            t.blockSignals(False)
        self._refresh()

    def save_file(self) -> None:
        cfg = self._check()
        if cfg is None:
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Save Synnax settings", prefs.last_dir() or "synnax.yaml",
                                            "YAML (*.yaml *.yml)")
        if not fn:
            return
        conn, rest = cfg.split()
        with open(fn, "w", encoding="utf-8") as fh:
            yaml.safe_dump({"synnax": {**conn, **rest}}, fh, sort_keys=False)

    # ------------------------------------------------------------------ keeping it
    def _ok(self) -> None:
        cfg = self._check()
        if cfg is None:
            return
        self.result_config = cfg
        conn, rest = cfg.split()
        prefs.set_synnax_connection(conn)
        self.changed = rest != dict(self.system.synnax or {})
        self.system.synnax = rest
        self.accept()

    def done(self, r) -> None:
        self._probe_token = None  # a test still running is forgotten
        self._timer.stop()
        super().done(r)
