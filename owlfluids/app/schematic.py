"""The schematic editor: components you can place, move, rotate and wire together.

In run mode the same scene shows live values, and clicking a valve toggles it.
"""

from __future__ import annotations

from collections import defaultdict, deque

from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QPainter, QPainterPath, QPainterPathStroker, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsObject,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QMenu,
)

from ..components import get_type
from ..fluids.registry import canonical
from ..io.system import Component, System
from . import media as M
from . import routing, symbols
from . import theme as T

GRID = 10.0
WIRE = T.qc(T.GRAY_300)
WIRE_SELECTED = T.qc(T.ACCENT)
PORT = T.qc(T.GRAY_900)
PORT_HOVER = T.qc(T.ACCENT)
LABEL_DEPTH = 28.0  # id and value labels take this much room above or below a symbol
LABEL_WIDTH = 60.0  # ... or this much beside it
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
        self.setVisible(False)  # shown while editing, on the part under the mouse or selected

    @property
    def ref(self) -> str:
        return f"{self.parentItem().comp.id}.{self.name}"

    def hoverEnterEvent(self, e):
        self.setBrush(QBrush(PORT_HOVER))

    def hoverLeaveEvent(self, e):
        self.setBrush(QBrush(PORT))


def _overlap(a: routing.Rect, b: routing.Rect) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


class ComponentItem(QGraphicsObject):
    def __init__(self, comp: Component):
        super().__init__()
        self.comp = comp
        self.ctype = get_type(comp.type)
        self.size, port_pos = symbols.shape(self.ctype.type_name, self.ctype.ports)
        self.ports = {name: PortItem(self, name, port_pos[name]) for name in self.ctype.ports if name in port_pos}
        self.state: dict = {}
        self.value_text = ""
        self.label_side = "bottom"
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable
                      | QGraphicsItem.ItemSendsGeometryChanges)
        self.setRotation(comp.rotation)
        self.setZValue(1)
        self.setAcceptHoverEvents(True)
        self.setToolTip(f"{comp.id} ({self.ctype.label or self.ctype.type_name})")

    def hoverEnterEvent(self, e):
        sc = self.scene()
        if sc is not None:
            sc.refresh_ports(self)

    def hoverLeaveEvent(self, e):
        sc = self.scene()
        if sc is not None:
            sc.refresh_ports(self, hovered=False)

    def obstacle(self) -> routing.Rect:
        """The symbol's footprint in the scene and its labels below it, which wires route around."""
        w, h = self.size
        r = self.mapRectToScene(QRectF(-w / 2, -h / 2, w, h))
        left, top, right, bottom = r.left(), r.top(), r.right(), r.bottom()
        side = self.label_side
        if side not in ("bottom", "right", "left", "top"):  # a corner label is out of the lines' way already
            return (left, top, right, bottom)
        return (left - (LABEL_WIDTH if side == "left" else 0), top - (LABEL_DEPTH if side == "top" else 0),
                right + (LABEL_WIDTH if side == "right" else 0), bottom + (LABEL_DEPTH if side == "bottom" else 0))

    def boundingRect(self) -> QRectF:
        w, h = self.size
        return QRectF(-w / 2 - 6, -h / 2 - 6, w + 12, h + 12)

    def paint(self, p: QPainter, option, widget=None) -> None:
        symbols.paint(p, self.ctype.type_name, self.size,
                      {**self._static_state(), **self.state, "rotation": self.rotation()})
        if self.isSelected():
            p.setPen(QPen(WIRE_SELECTED, 1, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(self.boundingRect().adjusted(2, 2, -2, -2))

    def _static_state(self) -> dict:
        """What the drawing needs from the parameters: valve actuators, and media while editing."""
        prm = self.comp.params
        t = self.ctype.type_name
        sc = self.scene()
        media = sc.media if sc is not None else {}
        if t == "fitting":
            m = media.get(f"{self.comp.id}.in")
            return {"line_colour": symbols.fluid_colour(*m)} if m else {}
        if t == "volume":
            m = next((media[f"{self.comp.id}.{q}"] for q in self.ctype.ports if f"{self.comp.id}.{q}" in media), None)
            return {"fill_colour": symbols.fluid_colour(*m)} if m else {}
        if t == "tank":
            m = media.get(f"{self.comp.id}.port")
            if not m:
                return {}
            return ({"gas_colour": symbols.fluid_colour(*m)} if m[1] else
                    {"fill": float(prm.get("fill") or 1.0), "liquid_colour": symbols.fluid_colour(m[0])})
        if self.ctype.type_name == "valve":  # the "solenoid" type name is a valve drawn as one
            default = "solenoid" if self.comp.type == "solenoid" else "pneumatic"
            return {"actuator": prm.get("actuator", default)}
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
                sc.settle_labels([self])
        if change == QGraphicsItem.ItemSelectedHasChanged:
            sc = self.scene()
            if sc is not None:
                sc.refresh_ports(self)
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
        self.points: list[routing.Point] = []
        self._shape = QPainterPath()
        self._drag = None
        self.flow = 0  # +1 fluid moving from a to b, -1 from b to a, 0 still
        self.setPen(QPen(WIRE, 2))
        self.setFlags(QGraphicsItem.ItemIsSelectable)
        self.setZValue(0)

    def set_colour(self, colour) -> None:
        """The colour of the medium in the line; None for a line nothing fills yet."""
        colour = colour or WIRE
        if colour != self.colour:
            self.colour = colour
            self.update()

    def set_points(self, points: list[routing.Point]) -> None:
        self.points = points
        path = QPainterPath(QPointF(*points[0]))
        for pt in points[1:]:
            path.lineTo(QPointF(*pt))
        self.setPath(path)
        st = QPainterPathStroker()
        st.setWidth(9)
        self._shape = st.createStroke(path)

    def shape(self) -> QPainterPath:
        """Clickable along the line itself (not the area its bends enclose)."""
        return self._shape

    @property
    def key(self) -> str:
        return f"{self.a} -- {self.b}"

    # dragging a segment sideways reshapes the route by hand; a double click hands it back
    def mousePressEvent(self, e):
        sc = self.scene()
        if sc is None or sc.running or e.button() != Qt.LeftButton:
            return super().mousePressEvent(e)
        i = routing.segment_at(self.points, (e.scenePos().x(), e.scenePos().y()))
        self._drag = (list(self.points), i) if i is not None else None
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if getattr(self, "_drag", None) is None:
            return super().mouseMoveEvent(e)
        start, i = self._drag
        to = snap(e.scenePos())
        pts, _ = routing.drag_segment(start, i, (to.x(), to.y()))
        self.set_points(routing.simplify(pts))

    def mouseReleaseEvent(self, e):
        if getattr(self, "_drag", None) is not None:
            start, _ = self._drag
            self._drag = None
            if self.points != start:
                bends, horizontal = routing.to_bends(self.points)
                self.scene().set_route(self, {"h": horizontal, "bends": [[x, y] for x, y in bends]})
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):
        sc = self.scene()
        if sc is not None and not sc.running:
            sc.set_route(self, None)

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
        self.media: dict = {}  # port -> (fluid, is_gas), guessed while editing
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
        self.refresh_label_sides()
        self.update_wires()
        self.settle_labels()
        self.refresh_media()

    def refresh_media(self) -> None:
        """Colour wires and tanks by the media the parts set (editing only; a run colours from the solver)."""
        if self.running:
            return
        self.media = M.edit_media(self.system)
        for w in self.wires:
            m = self.media.get(w.a) or self.media.get(w.b)
            w.set_colour(symbols.fluid_colour(*m) if m else None)
        for it in self.items_by_id.values():
            it.update()

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

    SIDES = ("bottom", "right", "left", "top")
    CORNERS = ("bottom-right", "bottom-left", "top-right", "top-left")

    def _label_spots(self, item: ComponentItem, side: str, nw: float, vw: float):
        """Where the id and value labels go for ``side``: two top-left points."""
        w, h = item.size
        r = item.mapRectToScene(QRectF(-w / 2, -h / 2, w, h))
        gap, line = 4.0, 12.0
        cx, cy = r.center().x(), r.center().y()
        if side == "bottom":
            return (cx - nw / 2, r.bottom() + gap), (cx - vw / 2, r.bottom() + gap + line)
        if side == "top":
            return (cx - nw / 2, r.top() - gap - 2 * line), (cx - vw / 2, r.top() - gap - line)
        if side == "right":
            return (r.right() + gap + 2, cy - line), (r.right() + gap + 2, cy)
        if side == "left":
            return (r.left() - gap - 2 - nw, cy - line), (r.left() - gap - 2 - vw, cy)
        # corners, for when lines leave every side: diagonally off the symbol, clear of them
        y = r.bottom() + gap if side.startswith("bottom") else r.top() - gap - 2 * line
        if side.endswith("right"):
            return (r.right() + gap, y), (r.right() + gap, y + line)
        return (r.left() - gap - nw, y), (r.left() - gap - vw, y + line)

    def place_labels(self, item: ComponentItem) -> None:
        """Id and value labels on the part's label side (see refresh_label_sides)."""
        name, value = self.labels.get(item.comp.id, (None, None))
        if name is None:
            return
        npos, vpos = self._label_spots(item, item.label_side,
                                       name.boundingRect().width(), value.boundingRect().width())
        name.setPos(*npos)
        value.setPos(*vpos)

    def _port_sides(self, item: ComponentItem, connected: set) -> set:
        """The sides of a part that lines come out of."""
        c = item.scenePos()
        used = set()
        for pname, port in item.ports.items():
            if f"{item.comp.id}.{pname}" in connected:
                pt = port.scenePos()
                dx, dy = routing.port_direction((c.x(), c.y()), (pt.x(), pt.y()))
                used.add("right" if dx > 0 else "left" if dx < 0 else "bottom" if dy > 0 else "top")
        return used

    def refresh_label_sides(self, items=None) -> None:
        """Put each part's labels on the first side (bottom, right, left, top) no line comes out of,
        or off a corner when lines leave every side. Wires are routed around this side."""
        connected = {ref for conn in self.system.connections for ref in conn}
        for it in items if items is not None else self.items_by_id.values():
            used = self._port_sides(it, connected)
            it.label_side = next((sd for sd in self.SIDES if sd not in used), "bottom-right")
            self.place_labels(it)

    def settle_labels(self, items=None) -> None:
        """After routing: move labels that a wire or another part runs through to the first clear spot,
        trying the free sides, then the corners, then the sides lines come out of."""
        connected = {ref for conn in self.system.connections for ref in conn}
        segments = [seg for w in self.wires for seg in routing.segments(w.points)]
        parts = {}
        for cid, it in self.items_by_id.items():
            w, h = it.size
            r = it.mapRectToScene(QRectF(-w / 2, -h / 2, w, h))
            parts[cid] = (r.left(), r.top(), r.right(), r.bottom())
        taken = {cid: self._label_rect(it) for cid, it in self.items_by_id.items() if cid in self.labels}
        for it in items if items is not None else self.items_by_id.values():
            cid = it.comp.id
            if cid not in self.labels:  # still being added
                continue
            used = self._port_sides(it, connected)
            order = ([sd for sd in self.SIDES if sd not in used] + list(self.CORNERS)
                     + [sd for sd in self.SIDES if sd in used])
            best, best_hits = it.label_side, None
            for side in [it.label_side, *order]:
                r = self._label_rect(it, side)
                hits = sum(routing.hits(a, b, r) for a, b in segments)
                hits += sum(_overlap(r, q) for k, q in parts.items() if k != cid)
                hits += sum(_overlap(r, q) for k, q in taken.items() if k != cid)
                if best_hits is None or hits < best_hits:
                    best, best_hits = side, hits
                if hits == 0:
                    break
            it.label_side = best
            taken[cid] = self._label_rect(it)
            self.place_labels(it)

    def _label_rect(self, item: ComponentItem, side: str | None = None) -> routing.Rect:
        """The scene box both labels cover on ``side``, at least LABEL_WIDTH wide (values grow while running)."""
        name, value = self.labels[item.comp.id]
        nw = max(name.boundingRect().width(), 1.0)
        vw = max(value.boundingRect().width(), LABEL_WIDTH)
        (nx, ny), (vx, vy) = self._label_spots(item, side or item.label_side, nw, vw)
        return (min(nx, vx), ny, max(nx + nw, vx + vw), vy + 11.0)

    def _add_wire(self, a: str, b: str) -> WireItem:
        w = WireItem(a, b)
        self.addItem(w)
        self.wires.append(w)
        return w

    def update_wires(self, comp_id: str | None = None) -> None:
        todo = [w for w in self.wires
                if comp_id is None or w.a.split(".")[0] == comp_id or w.b.split(".")[0] == comp_id]
        if not todo:
            return
        obstacles = {cid: it.obstacle() for cid, it in self.items_by_id.items()}
        for w in todo:
            self.route_wire(w, obstacles)

    def route_wire(self, w: WireItem, obstacles: dict | None = None) -> None:
        """Route one wire: by hand if the system stores a route for it, otherwise around the parts."""
        ends = []
        for ref in (w.a, w.b):
            cid, _, port = ref.partition(".")
            item = self.items_by_id.get(cid)
            if item is None:
                return
            pt, c = item.port_scene_pos(port), item.scenePos()
            ends.append(((pt.x(), pt.y()), routing.port_direction((c.x(), c.y()), (pt.x(), pt.y()))))
        (pa, da), (pb, db) = ends
        manual = self.system.routes.get(w.key)
        if manual:
            w.set_points(routing.manual_route(pa, pb, [tuple(b) for b in manual.get("bends", [])],
                                              bool(manual.get("h", True))))
            return
        if obstacles is None:
            obstacles = {cid: it.obstacle() for cid, it in self.items_by_id.items()}
        w.set_points(routing.auto_route(pa, da, pb, db, list(obstacles.values())))

    def set_route(self, w: WireItem, route: dict | None) -> None:
        """Store a hand-made route for a wire, or drop it (None) to route it automatically again."""
        if route is None:
            if self.system.routes.pop(w.key, None) is None:
                return
        else:
            self.system.routes[w.key] = route
        self.route_wire(w)
        self.changed_system.emit()

    def refresh_ports(self, item=None, hovered: bool | None = None) -> None:
        """Ports show only while editing: on the part under the mouse, selected parts, or all while wiring."""
        items = [item] if item is not None else list(self.items_by_id.values())
        for it in items:
            under = it.isUnderMouse() if hovered is None or it is not item else hovered
            show = not self.running and (self._drag_from is not None or under or it.isSelected())
            for port in it.ports.values():
                if port.isVisible() != show:
                    port.setVisible(show)

    def _port_pos(self, ref: str) -> QPointF | None:
        cid, _, port = ref.partition(".")
        item = self.items_by_id.get(cid)
        return item.port_scene_pos(port) if item else None

    # ------------------------------------------------------------------ edits
    def add_component(self, type_name: str, pos: QPointF) -> Component:
        ct = get_type(type_name)
        prefix = {"valve": "V", "tank": "K", "pressurized_tank": "TK", "boundary": "ATM",
                  "orifice": "O", "fitting": "F", "tee": "T", "check_valve": "CV", "relief_valve": "RV",
                  "regulator": "REG", "volume": "VOL", "manual_valve": "MV"}.get(type_name, type_name.upper())
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
        self.refresh_media()
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
        self.refresh_label_sides()
        self.update_wires()
        self.settle_labels()
        self.refresh_media()
        self.changed_system.emit()

    def _remove_wire(self, w: WireItem) -> None:
        self.system.routes.pop(w.key, None)
        if (w.a, w.b) in self.system.connections:
            self.system.connections.remove((w.a, w.b))
        elif (w.b, w.a) in self.system.connections:
            self.system.connections.remove((w.b, w.a))
        self.wires.remove(w)
        self.removeItem(w)

    def rotate_selected(self, step: int = 90) -> None:
        """Turn the selected parts by ``step`` degrees (clockwise positive)."""
        if self.running:
            return
        for it in self.selectedItems():
            if isinstance(it, ComponentItem):
                it.comp.rotation = (it.comp.rotation + step) % 360
                it.setRotation(it.comp.rotation)
                self.refresh_label_sides([it])
                self.update_wires(it.comp.id)
        self.settle_labels()
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
        routes = {}
        for k, v in sysm.routes.items():
            a, _, b = k.partition(" -- ")
            routes[f"{fix(a)} -- {fix(b)}"] = v
        sysm.routes = routes
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
            self.refresh_ports()
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
            self.refresh_ports()
            if target is not None:
                self.connect_ports(src.ref, target.ref)
            return
        super().mouseReleaseEvent(e)

    def connect_ports(self, a: str, b: str) -> None:
        if a == b or (a, b) in self.system.connections or (b, a) in self.system.connections:
            return
        self.system.connect(a, b)
        w = self._add_wire(a, b)
        self.refresh_label_sides([it for it in (self.items_by_id.get(a.split(".")[0]),
                                                self.items_by_id.get(b.split(".")[0])) if it is not None])
        self.route_wire(w)
        self.settle_labels()
        self.refresh_media()
        self.changed_system.emit()

    # ------------------------------------------------------------------ run mode
    def set_running(self, running: bool) -> None:
        self.running = running
        self.refresh_ports()
        self.refresh_media()
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
    "manual_valve": {"cv": 0.5},
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

    def contextMenuEvent(self, e):
        sc = self.scene()
        if sc.running:
            return
        item = self.itemAt(e.pos())
        if item is not None and item.parentItem() is not None:
            item = item.parentItem()
        menu = QMenu(self)
        if isinstance(item, ComponentItem):
            if not item.isSelected():
                sc.clearSelection()
                item.setSelected(True)
            menu.addAction("Rotate clockwise\tR", lambda: sc.rotate_selected(90))
            menu.addAction("Rotate counter-clockwise\tShift+R", lambda: sc.rotate_selected(-90))
            menu.addAction("Rotate 180°", lambda: sc.rotate_selected(180))
            menu.addSeparator()
            menu.addAction("Delete\tDel", sc.delete_selected)
        elif isinstance(item, WireItem):
            a = menu.addAction("Reset route", lambda: sc.set_route(item, None))
            a.setEnabled(item.key in sc.system.routes)
            menu.addAction("Delete\tDel", lambda: (sc.clearSelection(), item.setSelected(True), sc.delete_selected()))
        else:
            return
        menu.exec(e.globalPos())

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
