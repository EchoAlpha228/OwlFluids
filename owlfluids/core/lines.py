"""Lines, and what a sensor tapped onto one reads.

Sensors are parts of their own: a pressure transducer, thermocouple or flow meter is put
anywhere on a line and reads what the fluid is doing at that spot. In the model a line has
no length, friction or volume of its own, so all the lines joining a set of ports are one
point (a node: a tank's or volume's, or a small junction). Every sensor on those lines reads
that node's pressure, temperature, density, quality or purge gas.

The mass flow along one line is found by cutting the line there: whatever is on the far side
takes in exactly what the line carries (a junction holds next to nothing), so it is the sum
of the flows into the flow elements over there. If a tank or volume is on that side, the near
side is used instead. A line that is part of a loop of lines has no flow of its own.

Some sensors belong to a part rather than to a line, and stay on it: a tank's level, a load
cell's mass, a valve's position and a wall thermocouple (``PART_TYPES``).
"""

from __future__ import annotations

from collections import defaultdict

from ..components import get_type

# sensors placed on lines, and the letters in their bubble (new ids add the line's commodity
# in front, see app/naming.py): pressure transducer, thermocouple, flow meter, density
# meter, vapor quality, purge gas fraction
LINE_TYPES = {"pressure": "PT", "temperature": "TC", "mass_flow": "FM", "density": "DT", "quality": "QT",
              "purge_fraction": "PF"}
# sensors that belong to a part: level (tanks), mass (load cell on a tank or volume), valve
# position (valves and pumps), wall temperature (a tank's or volume's shell)
PART_TYPES = ("level", "mass", "valve_position", "wall_temperature")


def split(ref: str) -> tuple[str, str]:
    cid, _, port = ref.rpartition(".")
    return cid, port


def part_types(type_name: str) -> tuple[str, ...]:
    """The part sensors a component of this type can carry."""
    ct = get_type(type_name)
    if ct.type_name == "tank":
        return ("level", "mass", "wall_temperature")
    if ct.type_name in ("volume", "pneumatic_supply"):
        return ("mass", "wall_temperature")
    if ct.is_valve or ct.type_name == "air_pump":
        return ("valve_position",)
    return ()


def taps_parts(type_name: str) -> bool:
    """A sensor dropped on a part of this type reads at its nearest port: tanks and volumes."""
    ct = get_type(type_name)
    return ct.category == "Volumes" and ct.type_name != "boundary"


class Lines:
    """How the lines of a system join its ports, for working out what a line sensor reads."""

    def __init__(self, system):
        self.types = {}
        for c in system.components:
            try:
                self.types[c.id] = get_type(c.type)
            except ValueError:
                continue
        self.adj: dict[str, list[tuple[str, int]]] = defaultdict(list)  # port -> [(port, line)]
        self.lines = list(system.connections)
        for k, (a, b) in enumerate(self.lines):
            self.adj[a].append((b, k))
            self.adj[b].append((a, k))
        for cid, ct in self.types.items():
            if ct.merges_ports:  # a tee's ports are all one point
                refs = [f"{cid}.{p}" for p in ct.ports]
                for r in refs[1:]:
                    self.adj[refs[0]].append((r, -1))
                    self.adj[r].append((refs[0], -1))

    def index(self, a: str, b: str) -> int | None:
        for k, (x, y) in enumerate(self.lines):
            if (x, y) in ((a, b), (b, a)):
                return k
        return None

    def kind(self, ref: str) -> str:
        """storage (a tank's, volume's or vent's port), element (a flow element's), point (a tee's), or none."""
        cid, _ = split(ref)
        ct = self.types.get(cid)
        if ct is None:
            return "none"
        if ct.merges_ports:
            return "point"
        return "storage" if ct.category == "Volumes" else "element"

    def _reach(self, start: str, cut: int) -> set[str]:
        seen = {start}
        todo = [start]
        while todo:
            r = todo.pop()
            for q, k in self.adj.get(r, ()):
                if k != cut and q not in seen:
                    seen.add(q)
                    todo.append(q)
        return seen

    def sides(self, a: str, b: str) -> tuple[set[str], set[str]] | None:
        """The ports still joined to each end once line a--b is cut there: (a's, b's).
        None when the line is part of a loop of lines (or isn't a line of the system)."""
        k = self.index(a, b)
        if k is None:
            return None
        near = self._reach(a, k)
        if b in near:
            return None
        return near, self._reach(b, k)

    def flow_side(self, a: str, b: str) -> tuple[set[str], float] | None:
        """The ports whose flow elements add up to the flow along a--b, from a toward b, and
        the sign to give the sum of what they take from the line. None if no flow is
        readable there: the line is in a loop, or has a tank or volume on both sides."""
        sd = self.sides(a, b)
        if sd is None:
            return None
        near, far = sd
        if not any(self.kind(r) == "storage" for r in far):
            return far, 1.0  # all the line carries goes into the far side's elements
        if not any(self.kind(r) == "storage" for r in near):
            return near, -1.0  # ... or comes out of the near side's
        return None

    def flow_order(self, a: str, b: str) -> tuple[str, str]:
        """Which way a flow meter on line a--b reads positive: (from, to).

        The way a flow element at one end passes fluid (in to out), else out of a tank or
        volume and into a vent; a line between two tees reads positive from a to b.
        """
        for here, there in ((a, b), (b, a)):
            if self.kind(here) == "element":
                cid, port = split(here)
                return (here, there) if port == self.types[cid].ports[-1] else (there, here)
        for here, there in ((a, b), (b, a)):
            if self.kind(here) == "storage":
                cid, _ = split(here)
                return (there, here) if self.types[cid].type_name == "boundary" else (here, there)
        return (a, b)
