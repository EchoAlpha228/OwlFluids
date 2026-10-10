"""How each component type looks on the schematic: size, port positions, drawing."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPolygonF

LINE = QColor("#2b3440")
GAS = QColor("#cfe3f7")
LIQUID = QColor("#5b9bd5")
CRYO = QColor("#8fd3ff")
OPEN = QColor("#3fae5a")
CLOSED = QColor("#d64545")
PARTIAL = QColor("#e0a030")
METAL = QColor("#eef1f4")

# (width, height) and port positions relative to the centre, before rotation
SHAPES: dict[str, tuple[tuple[float, float], dict[str, tuple[float, float]]]] = {
    "tank": ((60, 110), {"top": (0, -55), "bottom": (0, 55), "port": (30, 0)}),
    "pressurized_tank": ((70, 120), {"gas": (0, -60), "liquid": (0, 60)}),
    "boundary": ((44, 40), {"port": (-22, 0)}),
    "volume": ((50, 50), {"a": (-25, 0), "b": (25, 0), "c": (0, -25), "d": (0, 25)}),
    "pipe": ((100, 16), {"in": (-50, 0), "out": (50, 0)}),
    "tee": ((20, 20), {"a": (-10, 0), "b": (10, 0), "c": (0, -10), "d": (0, 10)}),
    "fitting": ((40, 30), {"in": (-20, 0), "out": (20, 0)}),
    "orifice": ((40, 34), {"in": (-20, 0), "out": (20, 0)}),
    "valve": ((56, 40), {"in": (-28, 0), "out": (28, 0)}),
    "throttle_valve": ((56, 46), {"in": (-28, 0), "out": (28, 0)}),
    "check_valve": ((50, 34), {"in": (-25, 0), "out": (25, 0)}),
    "relief_valve": ((50, 50), {"in": (-25, 0), "out": (25, 0)}),
    "regulator": ((56, 50), {"in": (-28, 0), "out": (28, 0)}),
    "air_pump": ((44, 44), {"out": (22, 0)}),
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
    if position >= 0.999:
        return OPEN
    if position <= 0.001:
        return CLOSED
    return PARTIAL


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
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(LINE, 1.6))
    p.setBrush(QBrush(METAL))
    fill_frac = state.get("fill")
    liquid = state.get("liquid_colour", LIQUID)
    if type_name == "tank":
        r = QRectF(-w / 2, -h / 2, w, h)
        p.drawRoundedRect(r, 18, 18)
        if fill_frac:
            _fill_level(p, r, fill_frac, liquid)
    elif type_name == "pressurized_tank":
        r = QRectF(-w / 2, -h / 2, w, h)
        p.setBrush(QBrush(GAS))
        p.drawRoundedRect(r, 20, 20)
        _fill_level(p, r, fill_frac if fill_frac is not None else 0.8, liquid)
    elif type_name == "boundary":
        p.setBrush(Qt.NoBrush)
        p.drawLine(QPointF(-w / 2, 0), QPointF(w / 2 - 12, 0))
        for i in range(3):
            x = w / 2 - 12 + i * 5
            p.drawLine(QPointF(x, -h / 2 + 6 + i * 4), QPointF(x, h / 2 - 6 - i * 4))
    elif type_name == "volume":
        p.drawEllipse(QRectF(-w / 2, -h / 2, w, h))
    elif type_name == "pipe":
        p.setBrush(QBrush(state.get("pipe_colour", METAL)))
        p.drawRect(QRectF(-w / 2, -h / 2, w, h))
    elif type_name == "tee":
        p.setBrush(QBrush(LINE))
        p.drawEllipse(QRectF(-5, -5, 10, 10))
    elif type_name == "fitting":
        p.setBrush(Qt.NoBrush)
        path = QPainterPath(QPointF(-w / 2, 0))
        path.quadTo(QPointF(0, -h / 2), QPointF(w / 2, 0))
        p.drawPath(path)
        p.drawLine(QPointF(-w / 2, 0), QPointF(w / 2, 0))
    elif type_name == "orifice":
        p.drawLine(QPointF(-w / 2, 0), QPointF(-3, 0))
        p.drawLine(QPointF(3, 0), QPointF(w / 2, 0))
        p.setPen(QPen(LINE, 3))
        p.drawLine(QPointF(0, -h / 2 + 2), QPointF(0, -4))
        p.drawLine(QPointF(0, 4), QPointF(0, h / 2 - 2))
    elif type_name in ("valve", "throttle_valve"):
        _bowtie(p, w, h if type_name == "valve" else h - 8, valve_colour(state.get("position")))
        p.setPen(QPen(LINE, 1.6))
        top = -h / 2 + (2 if type_name == "valve" else 0)
        p.drawLine(QPointF(0, 0), QPointF(0, top + 4))
        p.setBrush(QBrush(METAL))
        p.drawRect(QRectF(-8, top, 16, 6))
        if type_name == "throttle_valve":
            p.drawLine(QPointF(-14, 12), QPointF(14, -10))
    elif type_name == "check_valve":
        hw, hh = w / 2 - 6, h / 2 - 4
        p.setBrush(QBrush(valve_colour(state.get("position"))))
        p.drawPolygon(QPolygonF([QPointF(-hw, -hh), QPointF(hw - 6, 0), QPointF(-hw, hh)]))
        p.setPen(QPen(LINE, 3))
        p.drawLine(QPointF(hw, -hh), QPointF(hw, hh))
    elif type_name == "relief_valve":
        _bowtie(p, w, h - 14, valve_colour(state.get("position")))
        p.setPen(QPen(LINE, 1.4))
        p.setBrush(Qt.NoBrush)
        path = QPainterPath(QPointF(0, 0))
        for i in range(5):
            path.lineTo(QPointF(-6 if i % 2 == 0 else 6, -6 - i * 3.5))
        p.drawPath(path)
    elif type_name == "air_pump":
        # compressor: a circle with a wedge pointing at its outlet, green while it runs
        p.setBrush(QBrush(valve_colour(state.get("position"))))
        p.drawEllipse(QRectF(-w / 2 + 2, -h / 2 + 2, w - 4, h - 4))
        p.drawPolyline(QPolygonF([QPointF(-w / 2 + 6, -h / 2 + 10), QPointF(w / 2 - 2, 0),
                                  QPointF(-w / 2 + 6, h / 2 - 10)]))
    elif type_name == "regulator":
        p.drawRect(QRectF(-w / 2 + 6, -6, w - 12, h / 2 + 2))
        p.setBrush(QBrush(valve_colour(state.get("position"))))
        p.drawEllipse(QRectF(-12, -h / 2, 24, 16))
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
