"""Default names for new parts and line sensors: commodity letter, part letter, dash, number.

An oxidizer valve is OV-1, the next one OV-2, a fuel tank FT-1, a helium regulator HR-1.
A relief valve takes two letters (ORV-1), and a line sensor its instrument tag
after the commodity: OPT-1 for a pressure transducer on an oxidizer line. A part
with no fluid reaching it yet is X (XV-1) until it is wired to one.

Fittings, tees, volumes, boundaries and pneumatic supplies are plumbing, not named
parts: they get a plain id the schematic doesn't draw (TEE-1).
"""

from __future__ import annotations

import re

from ..components import get_type
from . import media as M

# canonical fluid -> commodity letter
COMMODITY = {
    "oxygen": "O", "nitrous": "O",  # oxidizers
    "methane": "F", "ethanol": "F",  # fuels
    "helium": "H",
    "nitrogen": "N",
    "air": "A",
    "water": "W",
}
UNKNOWN = "X"

# part type -> part letter
PART = {
    "valve": "V",
    "manual_valve": "M",
    "check_valve": "C",
    "regulator": "R",
    "tank": "T",
    "relief_valve": "RV",
    "orifice": "O",
    "air_pump": "P",
}
# part type -> the plain id prefix of a part with no drawn name
UNNAMED = {"fitting": "FIT", "tee": "TEE", "volume": "VOL", "boundary": "BND", "pneumatic_supply": "SUP"}

# line sensor type -> instrument tag
SENSOR = {"pressure": "PT", "temperature": "TC", "mass_flow": "FM"}

_DEFAULT = re.compile(r"^([A-Z]+)-(\d+)$")


def commodity(fluid: str | None) -> str:
    return COMMODITY.get(M._fluid(fluid) or "", UNKNOWN)


def part_commodity(comp, media: dict) -> str:
    """A tank is named for what it holds (its liquid, under any pressurant); anything else for what reaches it."""
    t = get_type(comp.type).type_name
    if t == "tank" and M._fluid(comp.params.get("fluid")):
        return commodity(comp.params.get("fluid"))
    for port in get_type(comp.type).ports:
        m = media.get(f"{comp.id}.{port}")
        if m:
            return commodity(m[0])
    return commodity(comp.params.get("fluid"))


def shows_name(type_name: str) -> bool:
    return get_type(type_name).type_name not in UNNAMED


def part_prefix(comp, media: dict) -> str:
    t = get_type(comp.type).type_name
    if t in UNNAMED:
        return UNNAMED[t]
    return part_commodity(comp, media) + PART.get(t, t[:1].upper())


def sensor_prefix(stype: str, fluid: str | None) -> str:
    return commodity(fluid) + SENSOR.get(stype, stype[:2].upper())


def next_name(prefix: str, taken) -> str:
    """The first free ``prefix-n``, counting from 1."""
    taken = set(taken)
    n = 1
    while f"{prefix}-{n}" in taken:
        n += 1
    return f"{prefix}-{n}"


def prefix_of(name: str) -> str | None:
    m = _DEFAULT.match(name)
    return m.group(1) if m else None
