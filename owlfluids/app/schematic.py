"""The schematic editor: components you can place, move, rotate and wire together.

In run mode the same scene shows live values, and clicking a valve toggles it.
"""

from __future__ import annotations

from collections import defaultdict, deque

from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsObject,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
)

from ..components import get_type
from ..fluids.registry import canonical
from ..io.system import Component, System
from . import symbols
from . import theme as T

GRID = 10.0
WIRE = T.qc(T.GRAY_300)
WIRE_SELECTED = T.qc(T.ACCENT)
PORT = T.qc(T.GRAY_900)
PORT_HOVER = T.qc(T.ACCENT)
FLOW_MARK = T.qc(T.WHITE)
FLOW_SIZE = 2.5  # px, the side of each square
FLOW_DASHES = [1.0, 5.0]  # square, gap; in multiples of FLOW_SIZE
FLOW_SPEED = 2.0  # dash units per second the squares travel (2 x 2.5 = 5 px/s per unit)


def _fluid(name) -> str | None:
    try:
        return canonical(name) if name else None
    except (KeyError, ValueError):
        return None


def snap(p: QPointF) -> QPointF:
    return QPointF(round(p.x() / GRID) * GRID, round(p.y() / GRID) * GRID)


class PortItem(QGraphicsEllipseItem):
    R = 4.5

    def __init__(self, parent: ComponentItem, name: str, pos: tuple[float, float]):
        super().__init__(-self.R, -self.R, 2 * self.R, 2 * self.R, parent)
        self.name = name
        self.setPos(QPointF(*pos))
        self.setBrush(QBrush(PORT))
        self.setPen(QPen(WIRE, 1.2))
        self.setAcceptHoverEvents(True)
        self.setZValue(2)
        self.setToolTip(f"{parent.comp.id}.{name}")

    @property
    def ref(self) -> str:
        return f"{self.parentItem().comp.id}.{self.name}"

    def hoverEnterEvent(self, e):
        self.setBrush(QBrush(PORT_HOVER))

    def hoverLeaveEvent(self, e):
        self.setBrush(QBrush(PORT))


class ComponentItem(QGraphicsObject):
    def __init__(self, comp: Component):
        super().__init__()
        self.comp = comp
        self.ctype = get_type(comp.type)
        self.size, port_pos = symbols.shape(self.ctype.type_name, self.ctype.ports)
        self.ports = {name: PortItem(self, name, port_pos[name]) for name in self.ctype.ports if name in port_pos}
        self.state: dict = {}
        self.value_text = ""
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable
                      | QGraphicsItem.ItemSendsGeometryChanges)
        self.setRotation(comp.rotation)
        self.setZValue(1)
        self.setToolTip(f"{comp.id} ({self.ctype.label or self.ctype.type_name})")

    def boundingRect(self) -> QRectF:
        w, h = self.size
        return QRectF(-w / 2 - 6, -h / 2 - 6, w + 12, h + 12)

    def paint(self, p: QPainter, option, widget=None) -> None:
        symbols.paint(p, self.ctype.type_name, self.size, self.state or self._static_state())
        if self.isSelected():
            p.setPen(QPen(WIRE_SELECTED, 1, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(self.boundingRect().adjusted(2, 2, -2, -2))

    def _static_state(self) -> dict:
        """While editing, tanks show the colours of the fluids they are set to hold."""
        prm = self.comp.params
        if self.ctype.type_name == "pressurized_tank":
            return {"liquid_colour": symbols.fluid_colour(_fluid(prm.get("liquid"))),
                    "gas_colour": symbols.fluid_colour(_fluid(prm.get("pressurant")), gas=True)}
        return {}

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            return snap(value)
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.comp.position = (self.pos().x(), self.pos().y())
            sc = self.scene()
            if sc is not None:
                sc.update_wires(self.comp.id)
                sc.place_labels(self)
        return super().itemChange(change, value)

    def port_scene_pos(self, port: str) -> QPointF:
        item = self.ports.get(port)
        if item is None:
            return self.scenePos()
        return item.scenePos()


class WireItem(QGraphicsPathItem):
    def __init__(self, a: str, b: str):
        super().__init__()
        self.a, self.b = a, b
        self.colour = WIRE
        self.flow = 0  # +1 fluid moving from a to b, -1 from b to a, 0 still
        self.setPen(QPen(WIRE, 2))
        self.setFlags(QGraphicsItem.ItemIsSelectable)
        self.setZValue(0)

    def set_colour(self, colour) -> None:
        """The line's fluid colour while running; None for the plain editing colour."""
        self.colour = colour or WIRE
        self.update()

    def route(self, pa: QPointF, pb: QPointF) -> None:
        path = QPainterPath(pa)
        if abs(pa.y() - pb.y()) < 0.5 or abs(pa.x() - pb.x()) < 0.5:
            path.lineTo(pb)
        else:
            mid = QPointF((pa.x() + pb.x()) / 2, pa.y())
            path.lineTo(mid)
            path.lineTo(QPointF(mid.x(), pb.y()))
            path.lineTo(pb)
        self.setPath(path)

    def paint(self, p, option, widget=None):
        self.setPen(QPen(WIRE_SELECTED if self.isSelected() else self.colour, 3 if self.isSelected() else 2))
        super().paint(p, option, widget)
        if self.flow:  # white squares marching along the line in the direction of flow
            pen = QPen(FLOW_MARK, FLOW_SIZE, Qt.CustomDashLine, Qt.FlatCap)
            pen.setDashPattern(FLOW_DASHES)
            sc = self.scene()
            pen.setDashOffset(-self.flow * (sc.flow_phase if sc is not None else 0.0))
            p.setPen(pen)
            p.drawPath(self.path())


class Scene(QGraphicsScene):
    """Holds the System being edited and mirrors it as graphics items."""

    changed_system = Signal()  # any edit
    valve_clicked = Signal(str)  # run mode
    component_selected = Signal(object)  # Component or None

    def __init__(self):
        super().__init__()
        self.system = System()
        self.items_by_id: dict[str, ComponentItem] = {}
        self.wires: list[WireItem] = []
        self.labels: dict[str, tuple] = {}
        self.running = False
        self.flow_phase = 0.0
        self._drag_from: PortItem | None = None
        self._drag_line: QGraphicsLineItem | None = None
        self.setSceneRect(QRectF(-2000, -2000, 6000, 4000))
        self.selectionChanged.connect(self._on_selection)

    # ------------------------------------------------------------------ model <-> items
    def set_system(self, system: System) -> None:
        self.clear()
        self.items_by_id.clear()
        self.wires.clear()
        self.labels.clear()
        self.system = system
        if any(c.position is None for c in system.components):
            auto_layout(system)
        for c in system.components:
            self._add_item(c)
        for a, b in system.connections:
            self._add_wire(a, b)
        self.update_wires()

    def _add_item(self, comp: Component) -> ComponentItem:
        item = ComponentItem(comp)
        self.addItem(item)
        item.setPos(QPointF(*(comp.position or (0.0, 0.0))))
        name = self.addSimpleText(comp.id)
        name.setFont(T.mono_font(7.5))
        name.setBrush(QBrush(T.qc(T.TEXT_DIM)))
        name.setZValue(3)
        value = self.addSimpleText("")
        value.setFont(T.mono_font(8))
        value.setBrush(QBrush(T.qc(T.WHITE)))
        value.setZValue(3)
        self.labels[comp.id] = (name, value)
        self.items_by_id[comp.id] = item
        self.place_labels(item)
        return item

    def place_labels(self, item: ComponentItem) -> None:
        name, value = self.labels.get(item.comp.id, (None, None))
        if name is None:
            return
        r = item.sceneBoundingRect()
        name.setPos(r.center().x() - name.boundingRect().width() / 2, r.bottom() - 2)
        value.setPos(r.center().x() - value.boundingRect().width() / 2, r.bottom() + 10)

    def _add_wire(self, a: str, b: str) -> WireItem:
        w = WireItem(a, b)
        self.addItem(w)
        self.wires.append(w)
        return w

    def update_wires(self, comp_id: str | None = None) -> None:
        for w in self.wires:
            if comp_id is None or w.a.split(".")[0] == comp_id or w.b.split(".")[0] == comp_id:
                pa, pb = self._port_pos(w.a), self._port_pos(w.b)
                if pa is not None and pb is not None:
                    w.route(pa, pb)

    def _port_pos(self, ref: str) -> QPointF | None:
        cid, _, port = ref.partition(".")
        item = self.items_by_id.get(cid)
        return item.port_scene_pos(port) if item else None

    # ------------------------------------------------------------------ edits
    def add_component(self, type_name: str, pos: QPointF) -> Component:
        ct = get_type(type_name)
        prefix = {"pipe": "L", "valve": "V", "tank": "K", "pressurized_tank": "TK", "boundary": "ATM",
                  "orifice": "O", "fitting": "F", "tee": "T", "check_valve": "CV", "relief_valve": "RV",
                  "regulator": "REG", "volume": "VOL", "throttle_valve": "TV"}.get(type_name, type_name.upper())
        n = 1
        while any(c.id == f"{prefix}{n}" for c in self.system.components):
            n += 1
        params = {p.name: p.default for p in ct.params if p.default is None and p.name in REQUIRED.get(type_name, ())}
        params.update(NEW_DEFAULTS.get(type_name, {}))
        params = {k: v for k, v in params.items() if v is not None}
        p = snap(pos)
        comp = self.system.add(f"{prefix}{n}", type_name, at=[p.x(), p.y()], **params)
        item = self._add_item(comp)
        self.clearSelection()
        item.setSelected(True)
        self.changed_system.emit()
        return comp

    def delete_selected(self) -> None:
        if self.running:
            return
        for it in self.selectedItems():
            if isinstance(it, WireItem):
                self._remove_wire(it)
        for it in self.selectedItems():
            if isinstance(it, ComponentItem):
                cid = it.comp.id
                for w in [w for w in self.wires if w.a.split(".")[0] == cid or w.b.split(".")[0] == cid]:
                    self._remove_wire(w)
                self.system.components.remove(it.comp)
                self.system.sensors[:] = [s for s in self.system.sensors if s.at.split(".")[0] != cid]
                for lab in self.labels.pop(cid, ()):
                    self.removeItem(lab)
                self.removeItem(it)
                del self.items_by_id[cid]
        self.changed_system.emit()

    def _remove_wire(self, w: WireItem) -> None:
        if (w.a, w.b) in self.system.connections:
            self.system.connections.remove((w.a, w.b))
        elif (w.b, w.a) in self.system.connections:
            self.system.connections.remove((w.b, w.a))
        self.wires.remove(w)
        self.removeItem(w)

    def rotate_selected(self) -> None:
        if self.running:
            return
        for it in self.selectedItems():
            if isinstance(it, ComponentItem):
                it.comp.rotation = (it.comp.rotation + 90) % 360
                it.setRotation(it.comp.rotation)
                self.update_wires(it.comp.id)
                self.place_labels(it)
        self.changed_system.emit()

    def rename(self, old: str, new: str) -> None:
        sysm = self.system
        if any(c.id == new for c in sysm.components):
            raise ValueError(f"{new} is already used")
        comp = sysm.component(old)
        comp.id = new

        def fix(ref: str) -> str:
            cid, dot, port = ref.partition(".")
            return new + dot + port if cid == old else ref

        sysm.connections[:] = [(fix(a), fix(b)) for a, b in sysm.connections]
        for s in sysm.sensors:
            s.at = fix(s.at)
        for w in self.wires:
            w.a, w.b = fix(w.a), fix(w.b)
        item = self.items_by_id.pop(old)
        self.items_by_id[new] = item
        self.labels[new] = self.labels.pop(old)
        self.labels[new][0].setText(new)
        self.place_labels(item)
        self.changed_system.emit()

    def _on_selection(self) -> None:
        sel = [i for i in self.selectedItems() if isinstance(i, ComponentItem)]
        self.component_selected.emit(sel[0].comp if len(sel) == 1 else None)

    # ------------------------------------------------------------------ wiring by mouse
    def mousePressEvent(self, e):
        item = self.itemAt(e.scenePos(), self.views()[0].transform() if self.views() else None)
        if self.running:
            if isinstance(item, PortItem):
                item = item.parentItem()
            if isinstance(item, ComponentItem) and item.ctype.is_valve:
                self.valve_clicked.emit(item.comp.id)
                return
            super().mousePressEvent(e)
            return
        if isinstance(item, PortItem) and e.button() == Qt.LeftButton:
            self._drag_from = item
            p = item.scenePos()
            self._drag_line = self.addLine(QLineF(p, p), QPen(WIRE_SELECTED, 2, Qt.DashLine))
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._drag_line is not None:
            line = self._drag_line.line()
            line.setP2(e.scenePos())
            self._drag_line.setLine(line)
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._drag_line is not None:
            self.removeItem(self._drag_line)
            self._drag_line = None
            target = None
            for it in self.items(e.scenePos()):
                if isinstance(it, PortItem) and it is not self._drag_from:
                    target = it
                    break
            src, self._drag_from = self._drag_from, None
            if target is not None:
                self.connect_ports(src.ref, target.ref)
            return
        super().mouseReleaseEvent(e)

    def connect_ports(self, a: str, b: str) -> None:
        if a == b or (a, b) in self.system.connections or (b, a) in self.system.connections:
            return
        self.system.connect(a, b)
        w = self._add_wire(a, b)
        w.route(self._port_pos(a), self._port_pos(b))
        self.changed_system.emit()

    # ------------------------------------------------------------------ run mode
    def set_running(self, running: bool) -> None:
        self.running = running
        for it in self.items_by_id.values():
            it.setFlag(QGraphicsItem.ItemIsMovable, not running)
            if not running:
                it.state = {}
                it.update()
        if not running:
            for _name, value in self.labels.values():
                value.setText("")
            self.show_flows({})

    def show_flows(self, flows: dict[int, int], dt: float = 0.0) -> None:
        """``flows[wire index]`` = +1/-1/0; advances the moving squares by ``dt`` seconds."""
        self.flow_phase = (self.flow_phase + dt * FLOW_SPEED * 6.0) % 6.0  # one dash period = 6 units
        for k, w in enumerate(self.wires):
            d = flows.get(k, 0)
            if d or w.flow:
                w.flow = d
                w.update()

    def show_values(self, values: dict[str, tuple[str, dict]]) -> None:
        """``values[comp_id] = (text, state)``: label under the symbol and drawing state."""
        for cid, (text, state) in values.items():
            item = self.items_by_id.get(cid)
            if item is None:
                continue
            if state != item.state:
                item.state = state
                item.update()
            label = self.labels[cid][1]
            if label.text() != text:
                label.setText(text)
                self.place_labels(item)


# parameters filled in when a component is dropped from the palette
NEW_DEFAULTS = {
    "tank": {"fluid": "nitrogen", "volume": "10 L", "pressure": "1000 psi", "temperature": "20 C"},
    "pressurized_tank": {"liquid": "water", "pressurant": "nitrogen", "volume": "50 L", "fill": 0.8,
                         "pressure": "100 psi"},
    "boundary": {"fluid": "nitrogen"},
    "volume": {"volume": "0.5 L"},
    "fitting": {"kind": "elbow90", "diameter": "10 mm"},
    "orifice": {"cda": "10 mm2"},
    "valve": {"cv": 0.5},
    "throttle_valve": {"cv": 0.5},
    "check_valve": {"cv": 0.5},
    "relief_valve": {"cv": 0.3, "set_pressure": "500 psi"},
    "regulator": {"cv": 0.3, "setpoint": "300 psi"},
}
REQUIRED: dict[str, tuple] = {}


class View(QGraphicsView):
    def __init__(self, scene: Scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setAcceptDrops(True)
        self.setBackgroundBrush(QBrush(T.qc(T.BACKGROUND)))

    def wheelEvent(self, e):
        f = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self.scale(f, f)

    def drawBackground(self, p: QPainter, rect: QRectF) -> None:
        super().drawBackground(p, rect)
        p.setPen(QPen(T.qc(T.GRAY_800), 0))
        step = GRID * 5
        x = int(rect.left() // step) * step
        while x < rect.right():
            p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            x += step
        y = int(rect.top() // step) * step
        while y < rect.bottom():
            p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            y += step

    def fit(self) -> None:
        r = self.scene().itemsBoundingRect().adjusted(-60, -60, 60, 60)
        if r.isValid():
            self.fitInView(r, Qt.KeepAspectRatio)

    def dragEnterEvent(self, e):
        if e.mimeData().hasText():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        e.acceptProposedAction()

    def dropEvent(self, e):
        sc = self.scene()
        if sc.running:
            return
        sc.add_component(e.mimeData().text(), self.mapToScene(e.position().toPoint()))
        e.acceptProposedAction()


def auto_layout(system: System, dx: float = 170.0, dy: float = 130.0) -> None:
    """Place components with no position in columns by distance from the nearest tank."""
    adj: dict[str, set[str]] = defaultdict(set)
    for a, b in system.connections:
        ca, cb = a.split(".")[0], b.split(".")[0]
        adj[ca].add(cb)
        adj[cb].add(ca)
    ids = [c.id for c in system.components]
    depth: dict[str, int] = {}
    roots = [c.id for c in system.components if c.type in ("tank", "pressurized_tank")] or ids[:1]
    q = deque()
    for r in roots:
        depth[r] = 0
        q.append(r)
    while True:
        while q:
            i = q.popleft()
            for j in sorted(adj[i], key=ids.index):
                if j not in depth:
                    depth[j] = depth[i] + 1
                    q.append(j)
        rest = [i for i in ids if i not in depth]
        if not rest:
            break
        depth[rest[0]] = 0
        q.append(rest[0])
    # order each column by the mean row of its neighbours (a few up/down sweeps)
    layers: dict[int, list[str]] = defaultdict(list)
    for i in ids:
        layers[depth[i]].append(i)
    row = {i: k for layer in layers.values() for k, i in enumerate(layer)}
    order = sorted(layers)
    for sweep in range(6):
        seq = order[1:] if sweep % 2 == 0 else order[-2::-1]
        for d in seq:
            ref = d - 1 if sweep % 2 == 0 else d + 1

            def key(i, ref=ref):
                nb = [row[j] for j in adj[i] if depth[j] == ref]
                return sum(nb) / len(nb) if nb else row[i]

            layers[d].sort(key=key)
            for k, i in enumerate(layers[d]):
                row[i] = k
    for c in system.components:
        if c.position is None:
            c.position = (depth[c.id] * dx, row[c.id] * dy)
