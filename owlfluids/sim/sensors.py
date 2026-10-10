"""Realistic sensor behaviour: noise and response lag by sensor type and what the sensor sees.

These are display-side effects only: they shape what sensors read (the app, logs and
Synnax), never the simulated physics. Numbers are typical of test-stand
instrumentation on a 16-bit DAQ, not of any one transducer; give a sensor its own
``noise:`` (standard deviation, SI) or ``lag:`` to override them, or set
``sensor_noise: off`` in the system settings to read exact values.

- Pressure: 0.5 psi plus 0.05% of reading always; up to 0.3% more while fluid flows
  through the node (turbulence); more again while pressure is changing fast
  (valve transients), capped at 0.5% + 1 psi so it never trips a redline.
- Temperature (thermocouples): 0.3 K (about 0.5 °F) of noise and a 0.3 s first-order lag, like an
  exposed-junction probe; wall thermocouples lag 0.5 s.
- Mass flow: 1% of reading, so a meter on a still line reads 0.
- Level: 0.2% of the tank plus slosh while the level is moving.
- Mass (load cell): 0.02 kg plus 0.01% of reading.
- Valve position, quality, density, purge fraction: exact.
"""

from __future__ import annotations

import math

from .. import units

PSI = units.PSI

DEFAULT_LAG = {"temperature": 0.3, "wall_temperature": 0.5}  # s

FLOW_REF = 0.05  # kg/s through a node at which flow noise is fully developed


def default_lag(stype: str) -> float:
    return DEFAULT_LAG.get(stype, 0.0)


def sigma(stype: str, value: float, rate: float, flow: float) -> float:
    """Noise standard deviation (SI) for a sensor reading ``value`` that changes at ``rate``
    per second, with ``flow`` (0-1) how strongly fluid is moving past it."""
    a = abs(value)
    if stype == "pressure":
        s = 0.5 * PSI + 5e-4 * a + 3e-3 * a * flow
        return s + min(2e-3 * abs(rate), 0.005 * a + PSI)  # ringing while pressure moves (kept below redlines)
    if stype in ("temperature", "wall_temperature"):
        return 0.3
    if stype == "mass_flow":
        return 0.01 * a
    if stype == "level":
        return 0.002 + min(0.05 * abs(rate), 0.005)
    if stype == "mass":
        return 0.02 + 1e-4 * a
    return 0.0


def flow_factor(total_mdot: float) -> float:
    if not math.isfinite(total_mdot):
        return 0.0
    return min(abs(total_mdot) / FLOW_REF, 1.0)
