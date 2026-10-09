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
    s.sense("TC-K1", "temperature", "K1.port", units="F")
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
    s.sense("TC-INJ", "temperature", "INJ.in", units="F")
    s.sense("FM-LOX", "mass_flow", "V-M", units="kg/s")
    s.sense("M-LOX", "mass", "TK", units="kg")
    s.sense("LL-LOX", "level", "TK", units="%")
    return s


def lox_purge() -> System:
    """LOX feed with a helium purge: flow, close the main valve, then purge the line clear.

    Helium, nitrogen or air can purge; change the purge bottle's fluid.
    """
    s = lox_feed()
    s.name = "LOX feed with helium purge"
    s.add("K-HE", "tank", fluid="helium", volume="15 L", pressure="2000 psi", temperature="20 C", at=[300, 300])
    s.add("REG-PG", "regulator", cv=0.2, setpoint="150 psi", droop="10 psi", at=[450, 300])
    s.add("V-PG", "valve", cv=0.2, open_time="50 ms", at=[600, 300])
    s.add("CK-PG", "check_valve", cv=0.2, at=[600, 230])
    for a, b in [("K-HE.port", "REG-PG.in"), ("REG-PG.out", "V-PG.in"), ("V-PG.out", "CK-PG.in"),
                 ("CK-PG.out", "L2.in")]:
        s.connect(a, b)
    s.sense("PT-PG", "pressure", "V-PG.out", units="psi")
    s.sense("FM-PG", "mass_flow", "CK-PG", units="kg/s")
    s.sense("PF-INJ", "purge_fraction", "INJ.in", units="%")
    return s


def nitrous_fill() -> System:
    """Nitrous fill cart with an air purge, after a typical oxidizer fill P&ID.

    Fill: open OFTV, OFV and the run tank vent RTV. Safe: close OFV, open the dump
    ODV and the purge solenoid OPGS so shop air sweeps the fill line out. Close
    OPGS and ODV and the fill can start again. The air compressor keeps its
    receiver between 100 and 115 psig.
    """
    s = System("Nitrous fill with air purge")
    # valve, filter and check valve bodies hold a few tens of cc each; saturated nitrous
    # flashing through 1 cc junctions is much stiffer than it needs to be
    s.settings["junction_volume"] = "20 cc"
    s.add("NOS", "tank", fluid="nitrous", volume="30 L", fill=0.8, temperature="20 C", wall_mass="15 kg",
          wall_material="aluminum", at=[0, 260])
    s.add("OFTV", "valve", cv=1.5, open_time="1 s", at=[0, 150])
    s.add("OFF", "orifice", cv=2.0, at=[60, 0])
    s.add("L-OF1", "pipe", length="3 m", diameter="0.43 in", at=[200, 0])
    s.add("OFRV-1", "relief_valve", cv=0.3, set_pressure="1000 psi", at=[370, -80])
    s.add("ATM-1", "boundary", fluid="nitrous", at=[370, -160])
    s.add("OFV", "valve", cv=1.2, open_time="300 ms", at=[420, 0])
    s.add("OFCV", "check_valve", cv=1.2, at=[540, 0])
    s.add("ODV", "valve", cv=0.5, open_time="200 ms", normally="closed", at=[560, -80])
    s.add("ATM-2", "boundary", fluid="nitrous", at=[560, -160])
    s.add("OFRV-2", "relief_valve", cv=0.3, set_pressure="1000 psi", at=[650, -80])
    s.add("ATM-3", "boundary", fluid="nitrous", at=[650, -160])
    s.add("OFIF", "orifice", cv=2.0, at=[660, 0])
    s.add("L-OF2", "pipe", length="3 m", diameter="0.43 in", at=[780, 0])
    s.add("OQD", "fitting", kind="quick_disconnect", diameter="0.43 in", at=[880, 0])
    s.add("RFCV", "check_valve", cv=1.2, at=[960, 0])  # the run tank's own fill check valve
    s.add("RT", "tank", fluid="nitrous", volume="15 L", pressure="1 atm", temperature="20 C", wall_mass="6 kg",
          wall_material="aluminum", at=[1040, 100])
    s.add("RTV", "valve", cv=0.15, open_time="200 ms", at=[1040, -80])
    s.add("ATM-4", "boundary", fluid="nitrous", at=[1040, -160])
    # shop air: compressor, receiver, regulator, isolation valve, filter, purge solenoid and check
    s.add("AC", "air_pump", flow="3 g/s", start_pressure="100 psig", stop_pressure="115 psig", at=[200, 340])
    s.add("AR", "tank", fluid="air", volume="20 L", pressure="115 psig", temperature="20 C", at=[280, 340])
    s.add("APRV", "relief_valve", cv=0.1, set_pressure="150 psi", at=[280, 250])
    s.add("ATM-A", "boundary", fluid="air", at=[280, 180])
    s.add("APR", "regulator", cv=0.3, setpoint="90 psig", droop="10 psi", at=[350, 340])
    s.add("AIV", "valve", cv=0.5, normally="open", at=[400, 340])
    s.add("ACF", "orifice", cv=0.8, at=[450, 340])
    s.add("L-A", "pipe", length="2 m", diameter="0.18 in", at=[470, 260])
    s.add("OPGS", "valve", cv=0.3, open_time="50 ms", at=[470, 180])
    s.add("OPGCV", "check_valve", cv=0.3, at=[470, 90])
    for a, b in [("NOS.bottom", "OFTV.in"), ("OFTV.out", "OFF.in"), ("OFF.out", "L-OF1.in"),
                 ("L-OF1.out", "OFV.in"), ("L-OF1.out", "OFRV-1.in"), ("OFRV-1.out", "ATM-1.port"),
                 ("OFV.out", "OFCV.in"), ("OFCV.out", "OFIF.in"), ("OFCV.out", "ODV.in"), ("ODV.out", "ATM-2.port"),
                 ("OFCV.out", "OFRV-2.in"), ("OFRV-2.out", "ATM-3.port"), ("OFIF.out", "L-OF2.in"),
                 ("L-OF2.out", "OQD.in"), ("OQD.out", "RFCV.in"), ("RFCV.out", "RT.bottom"), ("RT.top", "RTV.in"),
                 ("RTV.out", "ATM-4.port"),
                 ("AC.out", "AR.port"), ("AR.port", "APRV.in"), ("APRV.out", "ATM-A.port"), ("AR.port", "APR.in"),
                 ("APR.out", "AIV.in"), ("AIV.out", "ACF.in"), ("ACF.out", "L-A.in"), ("L-A.out", "OPGS.in"),
                 ("OPGS.out", "OPGCV.in"), ("OPGCV.out", "OFV.out")]:
        s.connect(a, b)
    s.sense("OFPT", "pressure", "L-OF1.out", units="psig")
    s.sense("OFT", "temperature", "L-OF1.out", units="F")
    s.sense("OIPT", "pressure", "OFIF.out", units="psig")
    s.sense("APT", "pressure", "ACF.out", units="psig")
    s.sense("ART", "pressure", "AR.port", units="psig")
    s.sense("FM-OF", "mass_flow", "OFV", units="kg/s")
    s.sense("M-RT", "mass", "RT", units="kg")
    s.sense("LL-RT", "level", "RT", units="%")
    s.sense("PF-OF", "purge_fraction", "OFIF.out", units="%")
    s.sense("AC-ON", "valve_position", "AC")
    return s


def ln2_wet_dress() -> System:
    """LN2 stand-in for LOX: loaded run tank, line chilldown through a bleed, then flow."""
    s = System("LN2 wet dress")
    s.add("K-N2", "tank", fluid="nitrogen", volume="49 L", pressure="3000 psi", temperature="20 C", at=[0, -220])
    s.add("REG", "regulator", cv=0.3, setpoint="250 psi", droop="20 psi", at=[150, -220])
    s.add("V-P", "valve", cv=0.3, at=[300, -220])
    s.add("L-P", "pipe", length="2 m", diameter="6 mm", at=[300, -120])
    s.add("TK", "pressurized_tank", liquid="nitrogen", pressurant="nitrogen", volume="100 L", fill=0.85,
          pressure="40 psi", liquid_temperature="78 K", diameter="0.4 m", at=[0, 0])
    s.add("V-V", "valve", cv=0.3, at=[150, -80])
    s.add("ATM-V", "boundary", fluid="nitrogen", at=[300, -40])
    s.add("V-M", "valve", cv=4, open_time="200 ms", at=[150, 160])
    s.add("L-F", "pipe", length="4 m", diameter="0.5 in", wall_thickness="0.049 in", segments=4, at=[300, 160])
    s.add("TEE", "tee", at=[450, 160])
    s.add("V-B", "valve", cv=0.2, at=[450, 260])
    s.add("ATM-B", "boundary", fluid="nitrogen", at=[600, 260])
    s.add("INJ", "orifice", cda="25 mm2", at=[600, 160])
    s.add("ATM", "boundary", fluid="nitrogen", at=[750, 160])
    for a, b in [("K-N2.port", "REG.in"), ("REG.out", "V-P.in"), ("V-P.out", "L-P.in"), ("L-P.out", "TK.gas"),
                 ("TK.gas", "V-V.in"), ("V-V.out", "ATM-V.port"), ("TK.liquid", "V-M.in"), ("V-M.out", "L-F.in"),
                 ("L-F.out", "TEE.a"), ("TEE.b", "INJ.in"), ("INJ.out", "ATM.port"), ("TEE.d", "V-B.in"),
                 ("V-B.out", "ATM-B.port")]:
        s.connect(a, b)
    s.sense("PT-TK", "pressure", "TK.gas", units="psi")
    s.sense("PT-INJ", "pressure", "INJ.in", units="psi")
    s.sense("TC-INJ", "temperature", "INJ.in", units="F")
    s.sense("TC-LINE", "wall_temperature", "L-F.s2", units="F")
    s.sense("FM-LN2", "mass_flow", "V-M", units="kg/s")
    s.sense("M-LN2", "mass", "TK", units="kg")
    s.sense("LL-LN2", "level", "TK", units="%")
    return s


def n2o_blowdown() -> System:
    """Self-pressurizing nitrous run tank drained through a valve and injector."""
    s = System("Nitrous blowdown")
    s.add("K", "tank", fluid="nitrous", volume="10 L", fill=0.8, temperature="20 C", at=[0, 0])
    s.add("V", "valve", cv=0.3, open_time="20 ms", at=[150, 50])
    s.add("INJ", "orifice", cda="10 mm2", at=[300, 50])
    s.add("ATM", "boundary", fluid="nitrous", at=[450, 50])
    s.connect("K.bottom", "V.in")
    s.connect("V.out", "INJ.in")
    s.connect("INJ.out", "ATM.port")
    s.sense("PT-K", "pressure", "K.port", units="psi")
    s.sense("TC-K", "temperature", "K.port", units="F")
    s.sense("PT-INJ", "pressure", "INJ.in", units="psi")
    s.sense("FM-N2O", "mass_flow", "INJ", units="kg/s")
    s.sense("M-N2O", "mass", "K", units="kg")
    s.sense("LL-N2O", "level", "K", units="%")
    return s


EXAMPLES = {
    "gn2-blowdown": gn2_blowdown,
    "lox-feed": lox_feed,
    "lox-purge": lox_purge,
    "ln2-wet-dress": ln2_wet_dress,
    "n2o-blowdown": n2o_blowdown,
    "nitrous-fill": nitrous_fill,
    "test-stand": engine_test_stand,
}
