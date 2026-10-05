"""Example systems. ``owlfluids example NAME -o file.yaml`` writes one out."""

from __future__ import annotations

from ..io.system import System
from .generate import engine_test_stand


def gn2_blowdown() -> System:
    s = System("GN2 bottle blowdown")
    s.add("K1", "tank", fluid="nitrogen", volume="49 L", pressure="2400 psi", temperature="20 C", at=[0, 0])
    s.add("V1", "valve", cv=0.3, open_time="50 ms", at=[150, 0])
    s.add("L1", "pipe", length="2 m", diameter="0.25 in", at=[300, 0])
    s.add("ATM", "boundary", fluid="nitrogen", at=[450, 0])
    s.connect("K1.port", "V1.in")
    s.connect("V1.out", "L1.in")
    s.connect("L1.out", "ATM.port")
    s.sense("PT-K1", "pressure", "K1.port", units="psi")
    s.sense("TC-K1", "temperature", "K1.port", units="K")
    s.sense("FM-V1", "mass_flow", "V1", units="kg/s")
    return s


def lox_feed() -> System:
    s = System("LOX feed and injector")
    s.add("TK", "pressurized_tank", liquid="oxygen", pressurant="nitrogen", volume="60 L", fill=0.85,
          pressure="300 psi", liquid_temperature="90 K", diameter="0.3 m", at=[0, 0])
    s.add("K-N2", "tank", fluid="nitrogen", volume="49 L", pressure="4000 psi", temperature="20 C", at=[0, -200])
    s.add("REG", "regulator", cv=0.4, setpoint="300 psi", droop="20 psi", at=[150, -200])
    s.add("V-P", "valve", cv=0.5, normally="open", at=[300, -200])
    s.add("L-P", "pipe", length="2 m", diameter="8 mm", at=[300, -100])
    s.add("L1", "pipe", length="1.5 m", diameter="19 mm", at=[0, 150])
    s.add("F1", "fitting", kind="elbow90", diameter="19 mm", at=[150, 150])
    s.add("V-M", "valve", cv=6, open_time="150 ms", at=[300, 150])
    s.add("L2", "pipe", length="1 m", diameter="19 mm", at=[450, 150])
    s.add("INJ", "orifice", cda="60 mm2", at=[600, 150])
    s.add("AMB", "boundary", fluid="oxygen", at=[750, 150])
    for a, b in [("K-N2.port", "REG.in"), ("REG.out", "V-P.in"), ("V-P.out", "L-P.in"), ("L-P.out", "TK.gas"),
                 ("TK.liquid", "L1.in"), ("L1.out", "F1.in"), ("F1.out", "V-M.in"), ("V-M.out", "L2.in"),
                 ("L2.out", "INJ.in"), ("INJ.out", "AMB.port")]:
        s.connect(a, b)
    s.sense("PT-TK", "pressure", "TK.gas", units="psi")
    s.sense("PT-INJ", "pressure", "INJ.in", units="psi")
    s.sense("TC-INJ", "temperature", "INJ.in", units="K")
    s.sense("FM-LOX", "mass_flow", "V-M", units="kg/s")
    s.sense("M-LOX", "mass", "TK", units="kg")
    return s


EXAMPLES = {
    "gn2-blowdown": gn2_blowdown,
    "lox-feed": lox_feed,
    "test-stand": engine_test_stand,
}
