"""Orthogonal wire routing for the schematic.

Points are (x, y) tuples in scene units. A route leaves each port along the
direction the port faces (away from its component), then joins the two stubs
with the cheapest of a few orthogonal shapes: fewest bends, shortest, and not
through any component. A route the user has shaped by hand is kept as its bend
points and only re-squared to the ports when components move.
"""

from __future__ import annotations

STUB = 15.0  # how far a wire runs straight out of a port before it may turn
BEND_COST = 25.0
HIT_COST = 2000.0
REVERSAL_COST = 200.0  # doubling straight back on itself

Point = tuple[float, float]
Rect = tuple[float, float, float, float]  # left, top, right, bottom


def _segments(points: list[Point]):
    return zip(points[:-1], points[1:], strict=True)


def port_direction(centre: Point, port: Point) -> Point:
    """The axis direction a port faces: from the component's centre out through the port."""
    dx, dy = port[0] - centre[0], port[1] - centre[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return (1.0, 0.0)
    if abs(dx) >= abs(dy):
        return (1.0 if dx > 0 else -1.0, 0.0)
    return (0.0, 1.0 if dy > 0 else -1.0)


def simplify(points: list[Point]) -> list[Point]:
    """Drop repeated points and points in the middle of a straight run."""
    out: list[Point] = []
    for p in points:
        if out and abs(out[-1][0] - p[0]) < 1e-6 and abs(out[-1][1] - p[1]) < 1e-6:
            continue
        if len(out) >= 2:
            a, b = out[-2], out[-1]
            same_x = abs(a[0] - b[0]) < 1e-6 and abs(b[0] - p[0]) < 1e-6
            same_y = abs(a[1] - b[1]) < 1e-6 and abs(b[1] - p[1]) < 1e-6
            # merge a straight run, but keep a doubling back so it is seen (and costed) as one
            if (same_x and (b[1] - a[1]) * (p[1] - b[1]) >= 0) or (same_y and (b[0] - a[0]) * (p[0] - b[0]) >= 0):
                out[-1] = p
                continue
        out.append(p)
    return out


CLEARANCE = 6.0  # wires keep this far from parts they pass


def _hits(a: Point, b: Point, r: Rect) -> bool:
    """Does the axis-aligned segment a-b pass through r, or closer to it than CLEARANCE?"""
    left, top, right, bottom = r[0] - CLEARANCE, r[1] - CLEARANCE, r[2] + CLEARANCE, r[3] + CLEARANCE
    if abs(a[1] - b[1]) < 1e-6:  # horizontal
        y = a[1]
        x0, x1 = sorted((a[0], b[0]))
        return top < y < bottom and x0 < right and x1 > left
    x = a[0]
    y0, y1 = sorted((a[1], b[1]))
    return left < x < right and y0 < bottom and y1 > top


def _inside(p: Point, r: Rect) -> bool:
    return r[0] - CLEARANCE <= p[0] <= r[2] + CLEARANCE and r[1] - CLEARANCE <= p[1] <= r[3] + CLEARANCE


def cost(points: list[Point], obstacles: list[Rect]) -> float:
    length = sum(abs(b[0] - a[0]) + abs(b[1] - a[1]) for a, b in _segments(points))
    bends = max(len(points) - 2, 0)
    reversals = sum(1 for (a, b), (_, c) in zip(_segments(points), list(_segments(points))[1:], strict=False)
                    if (b[0] - a[0]) * (c[0] - b[0]) < 0 or (b[1] - a[1]) * (c[1] - b[1]) < 0)
    last = len(points) - 2
    hits = 0
    for i, (a, b) in enumerate(_segments(points)):
        for r in obstacles:
            # a wire's first and last runs leave the part they connect to; that part doesn't count
            if (i == 0 and _inside(points[0], r)) or (i == last and _inside(points[-1], r)):
                continue
            hits += _hits(a, b, r)
    return length + BEND_COST * bends + REVERSAL_COST * reversals + HIT_COST * hits


def auto_route(pa: Point, da: Point, pb: Point, db: Point, obstacles: list[Rect]) -> list[Point]:
    """Best orthogonal route from port a (facing da) to port b (facing db), endpoints included."""
    sa = (pa[0] + da[0] * STUB, pa[1] + da[1] * STUB)
    sb = (pb[0] + db[0] * STUB, pb[1] + db[1] * STUB)
    mx, my = (sa[0] + sb[0]) / 2, (sa[1] + sb[1]) / 2
    middles = [  # on a tie the earlier wins: a jog halfway reads better than one at a port
        [(mx, sa[1]), (mx, sb[1])],
        [(sa[0], my), (sb[0], my)],
        [(sb[0], sa[1])],  # across then down
        [(sa[0], sb[1])],  # down then across
    ]
    # detours around the obstacles' edges, for when every simple shape runs through a part
    for left, top, right, bottom in obstacles:
        for x in (left - STUB, right + STUB):
            middles.append([(x, sa[1]), (x, sb[1])])
        for y in (top - STUB, bottom + STUB):
            middles.append([(sa[0], y), (sb[0], y)])
    best, best_cost = None, float("inf")
    for mid in middles:
        pts = simplify([pa, sa, *mid, sb, pb])
        if any(abs(a[0] - b[0]) > 1e-6 and abs(a[1] - b[1]) > 1e-6 for a, b in _segments(pts)):
            continue  # not orthogonal
        c = cost(pts, obstacles)
        if c < best_cost:
            best, best_cost = pts, c
    if best is None:  # direct ports that face each other on one line
        best = simplify([pa, sa, (sb[0], sa[1]), sb, pb])
    return best


def manual_route(pa: Point, pb: Point, bends: list[Point], horizontal_first: bool) -> list[Point]:
    """Through the user's bend points, kept square: segments alternate horizontal and vertical."""
    pts = [pa]
    horizontal = horizontal_first
    for b in bends:
        prev = pts[-1]
        pts.append((b[0], prev[1]) if horizontal else (prev[0], b[1]))
        horizontal = not horizontal
    last = pts[-1]
    if abs(last[0] - pb[0]) > 1e-6 and abs(last[1] - pb[1]) > 1e-6:
        pts.append((pb[0], last[1]) if horizontal else (last[0], pb[1]))
    pts.append(pb)
    return simplify(pts)


def drag_segment(points: list[Point], index: int, to: Point) -> tuple[list[Point], int]:
    """Move segment ``index`` sideways to pass through ``to``; the ends stay on their ports.

    Dragging the first or last segment first splits off a zero-length piece at the
    port, which then grows into a short stub. Returns the new points and the index
    of the dragged segment in them.
    """
    pts = list(points)
    if index == 0:
        pts.insert(0, pts[0])
        index = 1
    if index == len(pts) - 2:
        pts.append(pts[-1])
    a, b = pts[index], pts[index + 1]
    if abs(a[1] - b[1]) < 1e-6:  # horizontal: moves up or down
        pts[index], pts[index + 1] = (a[0], to[1]), (b[0], to[1])
    else:
        pts[index], pts[index + 1] = (to[0], a[1]), (to[0], b[1])
    return pts, index


def segment_at(points: list[Point], at: Point, tolerance: float = 6.0) -> int | None:
    best, best_d = None, tolerance
    for i, (a, b) in enumerate(_segments(points)):
        x0, x1 = sorted((a[0], b[0]))
        y0, y1 = sorted((a[1], b[1]))
        dx = max(x0 - at[0], 0.0, at[0] - x1)
        dy = max(y0 - at[1], 0.0, at[1] - y1)
        d = (dx * dx + dy * dy) ** 0.5
        if d <= best_d:
            best, best_d = i, d
    return best


def to_bends(points: list[Point]) -> tuple[list[Point], bool]:
    """A route as stored: its interior bend points and whether it leaves port a horizontally."""
    pts = simplify(points)
    horizontal = len(pts) < 2 or abs(pts[0][1] - pts[1][1]) < 1e-6
    return pts[1:-1], horizontal
