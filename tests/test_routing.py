"""Schematic wire routing (pure geometry, no Qt)."""
from owlfluids.app import routing as R


def _square(points):
    return all(a[0] == b[0] or a[1] == b[1] for a, b in zip(points, points[1:], strict=False))


def test_facing_ports_on_one_line_are_joined_straight():
    assert R.auto_route((0, 0), (1, 0), (100, 0), (-1, 0), []) == [(0, 0), (100, 0)]


def test_offset_ports_jog_halfway():
    pts = R.auto_route((0, 0), (1, 0), (100, 50), (-1, 0), [])
    assert _square(pts) and pts == [(0, 0), (50, 0), (50, 50), (100, 50)]


def test_routes_around_a_part_in_the_way():
    part = (80, -20, 120, 20)
    pts = R.auto_route((0, 0), (1, 0), (200, 0), (-1, 0), [part])
    assert _square(pts)
    assert R.cost(pts, [part]) < R.HIT_COST


def test_leaves_and_enters_ports_along_their_direction():
    pts = R.auto_route((0, 0), (0, -1), (100, -100), (-1, 0), [])
    assert _square(pts)
    assert pts[1][0] == 0 and pts[1][1] < 0  # straight up out of a
    assert pts[-2][1] == -100 and pts[-2][0] < 100  # into b from its left


def test_dragged_segment_is_kept_when_parts_move():
    pts = [(0, 0), (50, 0), (50, 40), (100, 40)]
    moved, _ = R.drag_segment(pts, 1, (70, 0))
    bends, horizontal = R.to_bends(moved)
    assert R.manual_route((0, 0), (100, 40), bends, horizontal) == [(0, 0), (70, 0), (70, 40), (100, 40)]
    # the parts move: the route stays square and keeps its hand-placed jog
    again = R.manual_route((0, 10), (120, 60), bends, horizontal)
    assert _square(again) and (70, 10) in again


def test_dragging_an_end_segment_keeps_the_port():
    pts = [(0, 0), (50, 0), (50, 40), (100, 40)]
    moved, i = R.drag_segment(pts, 0, (0, -20))
    route = R.manual_route((0, 0), (100, 40), *R.to_bends(moved))
    assert route[0] == (0, 0) and route[-1] == (100, 40) and _square(route)


def test_port_direction():
    assert R.port_direction((0, 0), (28, 0)) == (1.0, 0.0)
    assert R.port_direction((0, 0), (0, -60)) == (0.0, -1.0)
    assert R.port_direction((0, 0), (22, -18)) == (1.0, 0.0)
