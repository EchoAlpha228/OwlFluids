"""How each component type looks on the schematic: size, port positions, drawing."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPolygonF

from . import theme as T

LINE = T.qc(T.GRAY_100)
GAS = T.qc(T.GRAY_600)
LIQUID = T.qc(T.DARK_BLUE)
CRYO = T.qc(T.DARK_CYAN)
OPEN = T.qc(T.VALVE_OPEN)
CLOSED = T.qc(T.VALVE_CLOSED)
METAL = T.qc(T.GRAY_700)



def fluid_colour(fluid: str | None, gas: bool = False) -> QColor:
    """A fluid's material colour; gas is the same hue, partly transparent."""
    c = T.qc(T.FLUID_COLOURS.get(fluid or "", T.FLUID_UNKNOWN))
    if gas:
        c.setAlphaF(T.GAS_ALPHA)
    return c


# (width, height) and port positions relative to the centre, before rotation
SHAPES: dict[str, tuple[tuple[float, float], dict[str, tuple[float, float]]]] = {
    "tank": ((60, 110), {"top": (0, -55), "bottom": (0, 55), "port": (30, 0)}),
    "pressurized_tank": ((70, 120), {"gas": (0, -60), "liquid": (0, 60)}),
    "boundary": ((44, 40), {"port": (-22, 0)}),
    "volume": ((50, 50), {"a": (-25, 0), "b": (25, 0), "c": (0, -25), "d": (0, 25)}),
    "tee": ((20, 20), {"a": (-10, 0), "b": (10, 0), "c": (0, -10), "d": (0, 10)}),
    "fitting": ((24, 12), {"in": (-12, 0), "out": (12, 0)}),
    "orifice": ((40, 34), {"in": (-20, 0), "out": (20, 0)}),
    "valve": ((56, 40), {"in": (-28, 0), "out": (28, 0)}),
    "manual_valve": ((56, 46), {"in": (-28, 0), "out": (28, 0)}),
    "check_valve": ((40, 34), {"in": (-20, 0), "out": (20, 0)}),  # ports on the seats' line
    "relief_valve": ((50, 50), {"in": (-25, 0), "out": (25, 0)}),
    "regulator": ((50, 40), {"in": (-25, 0), "out": (25, 0)}),
    "air_pump": ((44, 44), {"out": (22, -18)}),  # the tip of the discharge arrow
    "pneumatic_supply": ((30, 30), {"a": (-15, 0), "b": (15, 0), "c": (0, -15), "d": (0, 15)}),
}
DEFAULT_SHAPE = ((50, 40), {})


def shape(type_name: str, ports: tuple[str, ...]):
    size, pos = SHAPES.get(type_name, DEFAULT_SHAPE)
    if not pos:  # unknown type: spread its ports left and right
        pos = {p: ((-size[0] / 2) if i % 2 == 0 else size[0] / 2, (i // 2) * 12) for i, p in enumerate(ports)}
    return size, pos


def valve_colour(position: float | None) -> QColor:
    if position is None:
        return METAL
    return OPEN if position > 0.001 else CLOSED  # anything off its seat passes flow: open


def _state_colour(state: dict, default: QColor) -> QColor:
    pos = state.get("position")
    return default if pos is None else valve_colour(pos)


def _body(p: QPainter, w: float, state: dict) -> None:
    """Valve body (two triangles meeting at the centre) between the ports, coloured by position."""
    p.setPen(QPen(LINE, 1.6))
    p.drawLine(QPointF(-w / 2, 0), QPointF(-w / 2 + 4, 0))
    p.drawLine(QPointF(w / 2 - 4, 0), QPointF(w / 2, 0))
    _bowtie(p, w, 30, valve_colour(state.get("position")))
    p.setPen(QPen(LINE, 1.6))


def _arrow_head(p: QPainter, tip: QPointF, direction: QPointF, size: float = 5.0) -> None:
    dx, dy = direction.x(), direction.y()
    n = (dx * dx + dy * dy) ** 0.5 or 1.0
    dx, dy = dx / n, dy / n
    base = QPointF(tip.x() - dx * size, tip.y() - dy * size)
    side = QPointF(-dy * size * 0.5, dx * size * 0.5)
    p.save()
    p.setBrush(QBrush(p.pen().color()))
    p.drawPolygon(QPolygonF([tip, base + side, base - side]))
    p.restore()


def _actuator(p: QPainter, h: float, rotation: float = 0.0) -> None:
    """Pneumatic actuator: a square on the stem, with the amber A at its top right."""
    p.drawLine(QPointF(0, 0), QPointF(0, -h / 2 + 12))
    p.setBrush(QBrush(METAL))
    p.drawRect(QRectF(-6, -h / 2, 12, 12))
    _pneumatic_badge(p, QPointF(10, -h / 2 - 1), rotation)


def _coil(p: QPainter, bottom: float) -> None:
    """Solenoid: a coil on the stem."""
    p.setBrush(Qt.NoBrush)
    p.drawLine(QPointF(0, 0), QPointF(0, bottom))
    coil = QPainterPath(QPointF(0, bottom))
    for i in range(3):
        y = bottom - i * 4
        coil.cubicTo(QPointF(7, y), QPointF(7, y - 4), QPointF(0, y - 4))
    p.drawPath(coil)


def _upright_text(p: QPainter, r: QRectF, text: str, font, rotation: float) -> None:
    """Text centred in ``r`` that stays upright however the symbol is turned."""
    p.save()
    p.translate(r.center())
    p.rotate(-rotation)
    p.setPen(QPen(T.qc(T.GRAY_950)))
    p.setFont(font)
    p.drawText(QRectF(-r.width() / 2, -r.height() / 2, r.width(), r.height()), Qt.AlignCenter, text)
    p.restore()


def _pneumatic_badge(p: QPainter, centre: QPointF, rotation: float = 0.0) -> None:
    """Amber disc lettered A beside the actuator: a pneumatically operated valve."""
    p.save()
    p.setPen(QPen(T.qc(T.DARK_AMBER), 0.8))
    p.setBrush(QBrush(T.qc(T.AMBER)))
    r = QRectF(centre.x() - 4.5, centre.y() - 4.5, 9, 9)
    p.drawEllipse(r)
    _upright_text(p, r, "A", T.ui_font(4.5, bold=True), rotation)
    p.restore()



def _bowtie(p: QPainter, w: float, h: float, fill: QColor) -> None:
    hw, hh = w / 2 - 4, h / 2 - 6
    poly = QPolygonF([QPointF(-hw, -hh), QPointF(0, 0), QPointF(-hw, hh)])
    poly2 = QPolygonF([QPointF(hw, -hh), QPointF(0, 0), QPointF(hw, hh)])
    p.setBrush(QBrush(fill))
    p.drawPolygon(poly)
    p.drawPolygon(poly2)


def paint(p: QPainter, type_name: str, size, state: dict) -> None:
    """Draw one component centred on (0, 0). ``state`` carries live values when running."""
    w, h = size
    rot = state.get("rotation", 0.0)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(LINE, 1.6))
    p.setBrush(QBrush(METAL))
    fill_frac = state.get("fill")
    liquid = state.get("liquid_colour", LIQUID)
    if type_name == "tank":
        r = QRectF(-w / 2, -h / 2, w, h)
        p.drawRoundedRect(r, 18, 18)
        if "gas_colour" in state:
            p.setBrush(QBrush(state["gas_colour"]))
            p.drawRoundedRect(r, 18, 18)
        if fill_frac:
            _fill_level(p, r, fill_frac, liquid)
    elif type_name == "pressurized_tank":
        r = QRectF(-w / 2, -h / 2, w, h)
        p.setBrush(QBrush(METAL))
        p.drawRoundedRect(r, 20, 20)
        p.setBrush(QBrush(state.get("gas_colour", GAS)))
        p.drawRoundedRect(r, 20, 20)
        _fill_level(p, r, fill_frac if fill_frac is not None else 0.8, liquid)
    elif type_name == "boundary":
        # vent / exhaust: the line ends in a cone of dashed rays leaving the open end
        p.setBrush(Qt.NoBrush)
        p.drawLine(QPointF(-w / 2, 0), QPointF(-6, 0))
        for dy in (-10.0, 0.0, 10.0):
            pen = QPen(LINE, 1.6, Qt.CustomDashLine)
            pen.setDashPattern([2.5, 1.5])
            p.setPen(pen)
            p.drawLine(QPointF(-4, dy * 0.25), QPointF(w / 2 - 2, dy))
        p.setPen(QPen(LINE, 1.6))
    elif type_name == "volume":
        p.drawEllipse(QRectF(-w / 2, -h / 2, w, h))
        if "fill_colour" in state:  # the medium inside
            p.setBrush(QBrush(state["fill_colour"]))
            p.drawEllipse(QRectF(-w / 2, -h / 2, w, h))
    elif type_name == "tee":
        p.setBrush(QBrush(LINE))
        p.drawEllipse(QRectF(-5, -5, 10, 10))
    elif type_name == "fitting":
        # a fitting (elbow, union...): a small square on the line
        p.setPen(QPen(state.get("line_colour", T.qc(T.GRAY_300)), 2))
        p.drawLine(QPointF(-w / 2, 0), QPointF(-5, 0))
        p.drawLine(QPointF(5, 0), QPointF(w / 2, 0))
        p.setPen(QPen(LINE, 1.4))
        p.setBrush(QBrush(METAL))
        p.drawRect(QRectF(-5, -5, 10, 10))
    elif type_name == "orifice":
        # restriction: two arcs bowing towards the flow path
        p.setBrush(Qt.NoBrush)
        p.drawLine(QPointF(-w / 2, 0), QPointF(-12, 0))
        p.drawLine(QPointF(12, 0), QPointF(w / 2, 0))
        top, bot = QPainterPath(QPointF(-12, -h / 2 + 4)), QPainterPath(QPointF(-12, h / 2 - 4))
        top.quadTo(QPointF(0, -2), QPointF(12, -h / 2 + 4))
        bot.quadTo(QPointF(0, 2), QPointF(12, h / 2 - 4))
        p.drawPath(top)
        p.drawPath(bot)
    elif type_name == "valve":
        _body(p, w, state)
        if state.get("actuator") == "solenoid":  # solenoid: coil on the stem
            _coil(p, -10)
        else:  # pneumatic, normally open or closed: square actuator with the A
            _actuator(p, h, rot)
    elif type_name == "manual_valve":
        # hand valve: a stem with a T handle, no actuator
        _body(p, w, state)
        p.setPen(QPen(LINE, 1.4))
        top = -h / 2 + 4
        p.drawLine(QPointF(0, 0), QPointF(0, top))
        p.setPen(QPen(LINE, 3))
        p.drawLine(QPointF(-10, top), QPointF(10, top))
    elif type_name == "check_valve":
        # check valve as on the reference P&ID: two seats joined by the flapper (a diagonal from
        # the top of the inlet seat to the foot of the outlet seat), and an arrow in the allowed
        # flow direction, in to out. The flapper takes the open / closed colour.
        sx, sh = 14.0, 9.0
        p.drawLine(QPointF(-w / 2, 0), QPointF(-sx, 0))
        p.drawLine(QPointF(sx, 0), QPointF(w / 2, 0))
        p.drawLine(QPointF(-sx, -sh), QPointF(-sx, sh))
        p.drawLine(QPointF(sx, -sh), QPointF(sx, sh))
        p.setPen(QPen(_state_colour(state, LINE), 2.4))
        p.drawLine(QPointF(-sx, -sh), QPointF(sx, sh))
        p.setPen(QPen(LINE, 1.4))
        p.drawLine(QPointF(-7, -h / 2 + 3), QPointF(4, -h / 2 + 3))
        _arrow_head(p, QPointF(9, -h / 2 + 3), QPointF(1, 0), 5)
    elif type_name == "relief_valve":
        # relief valve: body with an S across it
        _body(p, w, state)
        p.setBrush(Qt.NoBrush)
        s_path = QPainterPath(QPointF(6, -8))
        s_path.cubicTo(QPointF(-10, -10), QPointF(-8, 0), QPointF(0, 0))
        s_path.cubicTo(QPointF(8, 0), QPointF(10, 10), QPointF(-6, 8))
        p.drawPath(s_path)
    elif type_name == "regulator":
        # pressure regulator as on the reference NPR: an in-line body flanged at both ends with a
        # filled dot at its centre; the sensing line rises from the dot and runs downstream
        # through a small open circle to a bar across the outlet side. Turned to run the other
        # way (180 or 270 degrees) it is mirrored rather than turned over, so the sensing line
        # stays up (or right) and the bar always sits on the outlet side.
        if rot % 360 in (180, 270):
            p.scale(1, -1)
        hw, hh, stem, bar = 12.0, 8.0, -11.0, 17.0
        p.drawLine(QPointF(-w / 2, 0), QPointF(-hw - 3, 0))
        p.drawLine(QPointF(hw + 3, 0), QPointF(w / 2, 0))
        p.setBrush(QBrush(valve_colour(state.get("position"))))
        p.drawPolygon(QPolygonF([QPointF(-hw, -hh), QPointF(0, 0), QPointF(-hw, hh)]))
        p.drawPolygon(QPolygonF([QPointF(hw, -hh), QPointF(0, 0), QPointF(hw, hh)]))
        for x in (-hw, -hw - 3, hw, hw + 3):  # flanges: a double line at each end
            p.drawLine(QPointF(x, -hh), QPointF(x, hh))
        p.drawLine(QPointF(0, 0), QPointF(0, stem))
        p.drawLine(QPointF(0, stem), QPointF(4.5, stem))
        p.drawLine(QPointF(10.5, stem), QPointF(bar, stem))
        p.drawLine(QPointF(bar, stem - 6), QPointF(bar, hh + 4))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QRectF(4.5, stem - 3, 6, 6))
        p.setBrush(QBrush(LINE))
        p.drawEllipse(QRectF(-3.5, -3.5, 7, 7))
    elif type_name == "air_pump":
        # pump / compressor: casing on a base, intake arrow to the centre; the discharge arrow
        # along the top ends at the outlet port
        r = min(w, h) / 2 - 5
        p.setBrush(QBrush(METAL))
        p.drawPolygon(QPolygonF([QPointF(-r + 3, r - 2), QPointF(r - 3, r - 2), QPointF(r + 2, r + 4),
                                 QPointF(-r - 2, r + 4)]))
        p.drawEllipse(QRectF(-r, -r - 1, 2 * r, 2 * r))
        p.setPen(QPen(_state_colour(state, LINE), 1.6))
        p.drawLine(QPointF(-w / 2, -1), QPointF(-2, -1))
        _arrow_head(p, QPointF(0, -1), QPointF(1, 0), 5)
        p.drawLine(QPointF(0, -r - 1), QPointF(w / 2 - 5, -r - 1))
        _arrow_head(p, QPointF(w / 2, -r - 1), QPointF(1, 0), 5)
    elif type_name == "pneumatic_supply":
        # pneumatic supply node: amber disc lettered A
        p.setPen(QPen(T.qc(T.DARK_AMBER), 1.2))
        p.setBrush(QBrush(T.qc(T.AMBER)))
        p.drawEllipse(QRectF(-w / 2 + 2, -h / 2 + 2, w - 4, h - 4))
        _upright_text(p, QRectF(-w / 2, -h / 2, w, h), "A", T.ui_font(9, bold=True), rot)
    else:
        p.drawRoundedRect(QRectF(-w / 2, -h / 2, w, h), 6, 6)


def _fill_level(p: QPainter, r: QRectF, frac: float, colour: QColor) -> None:
    frac = min(max(frac, 0.0), 1.0)
    clip = QPainterPath()
    clip.addRoundedRect(r, 18, 18)
    p.save()
    p.setClipPath(clip)
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(colour))
    p.drawRect(QRectF(r.left(), r.bottom() - r.height() * frac, r.width(), r.height() * frac))
    p.restore()
    p.setBrush(Qt.NoBrush)
    p.setPen(QPen(LINE, 1.6))
    p.drawRoundedRect(r, 18, 18)
