"""Lines, and what a sensor tapped onto one reads.

Sensors are parts of their own: a pressure transducer, thermocouple or flow meter is put
anywhere on a line and reads what the fluid is doing at that spot. In the model a line is a
length of tubing (``length``, inner ``diameter``; the ``line_length`` and ``line_diameter``
settings by default) whose fluid has inertia: its mass flow is a state of its own, which the
pressure difference between its ends accelerates (water hammer, the time a line takes to
get flowing). It has no friction: losses are in the valves, orifices and fittings. The
fluid it holds sits at its ends, next to the parts it joins. Each end is a point (a node: a
tank's or volume's, or a small junction at a valve's port); a tee joins the lines on its
ports into one point.

A pressure, temperature, density, quality or purge sensor reads the end of its line it sits
nearer (with no place of its own, the end at a tank or volume, else the first). A flow meter
reads the line's own flow. A line of length 0 joins two parts bolted straight together into
one point: it has no flow of its own, so a meter on it reads the flow element at one of its
ends.

Some sensors belong to a part rather than to a line, and stay on it: a tank's level, a load
cell's mass, a valve's position and a wall thermocouple (``PART_TYPES``).
"""

from __future__ import annotations

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
        self.system = system
        self.types = {}
        for c in system.components:
            try:
                self.types[c.id] = get_type(c.type)
            except ValueError:
                continue
        self.lines = list(system.connections)

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

    def flow_readable(self, a: str, b: str) -> bool:
        """Whether a flow meter on line a--b reads a flow: every line with a length carries its
        own, and one of length 0 reads the flow element at one of its ends."""
        if self.index(a, b) is None:
            return False
        if self.length(a, b) > 0.0:
            return True
        return self.kind(a) == "element" or self.kind(b) == "element"

    def length(self, a: str, b: str) -> float:
        from .. import units

        spec = self.system.line_spec(a, b)
        raw = spec["length"] if spec.get("length") is not None else self.system.setting("line_length")
        try:
            return units.parse(raw, "length")
        except ValueError:
            return 1.0

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
