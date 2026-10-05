"""Small unit parser so system files can say "3000 psi" or "-196 C".

Every quantity is converted to SI on load: Pa, K, m, m^2, m^3, kg, s, kg/s.
A bare number is taken as already being in SI.
"""

from __future__ import annotations

import re

PSI = 6894.757293168
ATM = 101325.0

# multiplicative units: name -> (factor to SI, dimension)
_UNITS: dict[str, tuple[float, str]] = {
    # pressure (absolute)
    "pa": (1.0, "pressure"),
    "kpa": (1e3, "pressure"),
    "mpa": (1e6, "pressure"),
    "bar": (1e5, "pressure"),
    "bara": (1e5, "pressure"),
    "mbar": (100.0, "pressure"),
    "psi": (PSI, "pressure"),
    "psia": (PSI, "pressure"),
    "atm": (ATM, "pressure"),
    # length
    "m": (1.0, "length"),
    "cm": (0.01, "length"),
    "mm": (1e-3, "length"),
    "um": (1e-6, "length"),
    "in": (0.0254, "length"),
    "ft": (0.3048, "length"),
    # area
    "m2": (1.0, "area"),
    "mm2": (1e-6, "area"),
    "cm2": (1e-4, "area"),
    "in2": (0.0254**2, "area"),
    # volume
    "m3": (1.0, "volume"),
    "l": (1e-3, "volume"),
    "ml": (1e-6, "volume"),
    "cc": (1e-6, "volume"),
    "in3": (0.0254**3, "volume"),
    "ft3": (0.3048**3, "volume"),
    "gal": (3.785411784e-3, "volume"),
    # mass
    "kg": (1.0, "mass"),
    "g": (1e-3, "mass"),
    "lb": (0.45359237, "mass"),
    "lbm": (0.45359237, "mass"),
    # time
    "s": (1.0, "time"),
    "ms": (1e-3, "time"),
    "min": (60.0, "time"),
    # mass flow
    "kg/s": (1.0, "mass_flow"),
    "g/s": (1e-3, "mass_flow"),
    "lb/s": (0.45359237, "mass_flow"),
    "lbm/s": (0.45359237, "mass_flow"),
    # heat transfer coefficient
    "w/m2k": (1.0, "htc"),
}

_NUM = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([^\s]*)\s*$")


def parse(value, dimension: str | None = None) -> float:
    """Convert a number or a "<number> <unit>" string to SI.

    ``dimension`` is checked when given, so "3 in" passed as a pressure fails loudly.
    Gauge pressures ("psig", "barg") add one standard atmosphere.
    Temperatures accept K, C and F.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise TypeError(f"cannot read a quantity from {value!r}")
    m = _NUM.match(value)
    if not m:
        raise ValueError(f"cannot read a quantity from {value!r}")
    num = float(m.group(1))
    unit = m.group(2).lower().replace("³", "3").replace("²", "2").replace("·", "")
    if unit == "":
        return num
    if unit in ("k", "c", "degc", "°c", "f", "degf", "°f", "r"):
        _check(dimension, "temperature", value)
        if unit == "k":
            return num
        if unit in ("c", "degc", "°c"):
            return num + 273.15
        if unit == "r":
            return num * 5.0 / 9.0
        return (num - 32.0) * 5.0 / 9.0 + 273.15
    if unit in ("psig", "barg"):
        _check(dimension, "pressure", value)
        return num * (PSI if unit == "psig" else 1e5) + ATM
    if unit not in _UNITS:
        raise ValueError(f"unknown unit {m.group(2)!r} in {value!r}")
    factor, dim = _UNITS[unit]
    _check(dimension, dim, value)
    return num * factor


def _check(expected: str | None, actual: str, value) -> None:
    if expected is not None and expected != actual:
        raise ValueError(f"{value!r} is a {actual}, expected a {expected}")


def to_psi(pa: float) -> float:
    return pa / PSI


def to_celsius(k: float) -> float:
    return k - 273.15


DEFAULT_UNITS = {"pressure": "psi", "temperature": "K", "mass_flow": "kg/s", "valve_position": "",
                 "quality": "", "density": "kg/m3", "mass": "kg", "wall_temperature": "K"}


def convert(value_si: float, unit: str) -> float:
    """SI value expressed in ``unit`` (the inverse of ``parse``); "" leaves it as is."""
    u = unit.lower().replace("°", "").replace("deg", "")
    if u in ("", "si", "k"):
        return value_si
    if u == "c":
        return value_si - 273.15
    if u == "f":
        return (value_si - 273.15) * 9.0 / 5.0 + 32.0
    if u == "r":
        return value_si * 9.0 / 5.0
    if u == "psig":
        return (value_si - ATM) / PSI
    if u == "barg":
        return (value_si - ATM) / 1e5
    if u == "kg/m3":
        return value_si
    if u not in _UNITS:
        raise ValueError(f"unknown unit {unit!r}")
    return value_si / _UNITS[u][0]
