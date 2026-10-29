"""What fills each line while editing, so the drawing shows media before a run.

A guess from the parts that set a fluid: tanks, bottles, pumps, supplies and,
last, boundaries. It spreads along wires and through every part that passes
flow. A run replaces it with the solver's own fluids and phases.
"""

from __future__ import annotations

from collections import deque

from ..components import get_type
from ..fluids.registry import FLUIDS, canonical

SOURCES = ("tank", "air_pump", "pneumatic_supply")  # set what they feed, first
WEAK_SOURCES = ("boundary",)  # a vent takes what flows into it; only fills lines nothing else reaches


def _fluid(name) -> str | None:
    try:
        return canonical(str(name)) if name else None
    except (KeyError, ValueError):
        return None


def _liquid_name(name) -> bool:
    """LOX, LN2, LCH4 or a fluid that is only ever liquid here (water, ethanol)."""
    raw = str(name or "").strip().lower()
    f = _fluid(raw)
    return f is not None and (FLUIDS[f][1] == "liquid" or (raw.startswith("l") and raw[1:] in ("ox", "n2", "ch4")))


def _seeds(comp) -> dict[str, tuple[str, bool]]:
    """port -> (fluid, is_gas) for a part that sets its own fluid."""
    t = get_type(comp.type).type_name
    prm = comp.params
    if t == "tank" and _fluid(prm.get("pressurant")):  # liquid under a separate gas
        out = {}
        if _fluid(prm.get("fluid")):
            out["bottom"] = (_fluid(prm.get("fluid")), False)
        out["top"] = out["port"] = (_fluid(prm.get("pressurant")), True)
        return out
    name = prm.get("fluid", "air" if t in ("air_pump",) else None)
    f = _fluid(name)
    if f is None:
        return {}
    if t == "tank" and prm.get("fill") is not None:  # saturated: vapor over liquid
        return {"top": (f, True), "bottom": (f, False), "port": (f, False)}
    gas = not _liquid_name(name)
    return {p: (f, gas) for p in get_type(comp.type).ports}


def edit_media(system) -> dict[str, tuple[str, bool]]:
    """Port reference ("V1.in") -> (fluid, is_gas) for every port a fluid reaches."""
    wires: dict[str, list[str]] = {}
    for a, b in system.connections:
        wires.setdefault(a, []).append(b)
        wires.setdefault(b, []).append(a)
    comps = {c.id: c for c in system.components}
    media: dict[str, tuple[str, bool]] = {}

    def spread(order):
        q = deque()
        for c in system.components:
            if get_type(c.type).type_name in order:
                for port, m in _seeds(c).items():
                    ref = f"{c.id}.{port}"
                    if ref not in media:
                        media[ref] = m
                        q.append(ref)
        while q:
            ref = q.popleft()
            m = media[ref]
            nxt = list(wires.get(ref, []))
            cid, _, _port = ref.partition(".")
            c = comps.get(cid)
            if c is not None and get_type(c.type).type_name not in SOURCES + WEAK_SOURCES:
                nxt += [f"{cid}.{p}" for p in get_type(c.type).ports]  # flows through the part
            for r in nxt:
                if r not in media:
                    media[r] = m
                    q.append(r)

    spread(SOURCES)
    spread(WEAK_SOURCES)
    return media
