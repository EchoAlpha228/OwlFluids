"""Example systems. ``owlfluids example NAME -o file.yaml`` writes one out."""

from __future__ import annotations

from ..io.system import System
from .generate import engine_test_stand


def gn2_blowdown() -> System:
    s = System("GN2 bottle blowdown")
    s.add("K1", "tank", fluid="nitrogen", volume="49 L", pressure="2400 psi", temperature="20 C", at=[0, 0])
    s.add("V1", "valve", cv=0.3, open_time="50 ms", at=[150, 0])
    s.add("ATM", "boundary", fluid="nitrogen", at=[300, 0])
    s.connect("K1.top", "V1.in", length="0.5 m")
    s.connect("V1.out", "ATM.port", length=0)  # straight out to the atmosphere
    s.sense("PT-K1", "pressure", line=("K1.top", "V1.in"), units="psi")
    s.sense("TC-K1", "temperature", line=("K1.top", "V1.in"), units="F")
    s.sense("FM-V1", "mass_flow", line=("V1.out", "ATM.port"), units="kg/s")
    return s


def lox_feed() -> System:
    s = System("LOX feed and injector")
    s.add("TK", "tank", fluid="oxygen", pressurant="nitrogen", volume="60 L", fill=0.85,
          pressure="300 psi", temperature="90 K", diameter="0.3 m", at=[0, 0])
    s.add("K-N2", "tank", fluid="nitrogen", volume="49 L", pressure="4000 psi", temperature="20 C", at=[0, -200])
    s.add("REG", "regulator", cv=0.4, setpoint="300 psi", droop="20 psi", at=[150, -200])
    s.add("V-P", "valve", cv=0.5, normally="open", at=[300, -200])
    s.add("F1", "fitting", kind="elbow90", diameter="19 mm", at=[150, 150])
    s.add("V-M", "valve", cv=6, open_time="150 ms", at=[300, 150])
    s.add("INJ", "orifice", cda="60 mm2", at=[600, 150])
    s.add("AMB", "boundary", fluid="oxygen", at=[750, 150])
    # line lengths in m; the injector faces straight into the chamber (here the atmosphere)
    for a, b, length in [("K-N2.top", "REG.in", 0.3), ("REG.out", "V-P.in", 0.5), ("V-P.out", "TK.top", 1),
                         ("TK.bottom", "F1.in", 0.5), ("F1.out", "V-M.in", 1), ("V-M.out", "INJ.in", 1),
                         ("INJ.out", "AMB.port", 0)]:
        s.connect(a, b, length=length)
    s.sense("PT-TK", "pressure", line=("V-P.out", "TK.top"), units="psi")
    s.sense("PT-INJ", "pressure", line=("V-M.out", "INJ.in"), along=1.0, units="psi")
    s.sense("TC-INJ", "temperature", line=("V-M.out", "INJ.in"), along=1.0, units="F")
    s.sense("FM-LOX", "mass_flow", line=("V-M.out", "INJ.in"), units="kg/s")
    s.sense("M-LOX", "mass", "TK", units="kg")
    s.sense("LL-LOX", "level", "TK", units="%")
    return s


def lox_purge() -> System:
    """LOX feed with a helium purge: flow, close the main valve, then purge the line clear.

    The purge comes in just below the main valve, so it pushes the whole feed line out through
    the injector. Helium, nitrogen or air can purge; change the purge bottle's fluid.
    """
    s = lox_feed()
    s.name = "LOX feed with helium purge"
    s.add("K-HE", "tank", fluid="helium", volume="15 L", pressure="2000 psi", temperature="20 C", at=[300, 300])
    s.add("REG-PG", "regulator", cv=0.2, setpoint="150 psi", droop="10 psi", at=[450, 300])
    s.add("V-PG", "valve", cv=0.2, open_time="50 ms", at=[600, 300])
    s.add("CK-PG", "check_valve", cv=0.2, at=[600, 230])
    for a, b, length in [("K-HE.top", "REG-PG.in", 0.3), ("REG-PG.out", "V-PG.in", 0.5),
                         ("V-PG.out", "CK-PG.in", 0.1), ("CK-PG.out", "V-M.out", 0.3)]:
        s.connect(a, b, length=length)
    s.sense("PT-PG", "pressure", line=("V-PG.out", "CK-PG.in"), units="psi")
    s.sense("FM-PG", "mass_flow", line=("CK-PG.out", "V-M.out"), units="kg/s")
    s.sense("PF-INJ", "purge_fraction", line=("V-M.out", "INJ.in"), along=1.0, units="%")
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
    s.add("OFRV-1", "relief_valve", cv=0.3, set_pressure="1000 psi", at=[370, -80])
    s.add("ATM-1", "boundary", fluid="nitrous", at=[370, -160])
    s.add("OFV", "valve", cv=1.2, open_time="300 ms", at=[420, 0])
    s.add("OFCV", "check_valve", cv=1.2, at=[540, 0])
    s.add("ODV", "valve", cv=0.5, open_time="200 ms", normally="closed", at=[560, -80])
    s.add("ATM-2", "boundary", fluid="nitrous", at=[560, -160])
    s.add("OFRV-2", "relief_valve", cv=0.3, set_pressure="1000 psi", at=[650, -80])
    s.add("ATM-3", "boundary", fluid="nitrous", at=[650, -160])
    s.add("OFIF", "orifice", cv=2.0, at=[660, 0])
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
    s.add("OPGS", "valve", cv=0.3, open_time="50 ms", at=[470, 180])
    s.add("OPGCV", "check_valve", cv=0.3, at=[470, 90])
    # line lengths in m: vents and reliefs exhaust straight to the atmosphere, the fill hose is 3 m
    for a, b, length in [("NOS.bottom", "OFTV.in", 0.3), ("OFTV.out", "OFF.in", 0.5), ("OFF.out", "OFV.in", 1),
                         ("OFF.out", "OFRV-1.in", 0.1), ("OFRV-1.out", "ATM-1.port", 0),
                         ("OFV.out", "OFCV.in", 0.2), ("OFCV.out", "OFIF.in", 0.5), ("OFCV.out", "ODV.in", 0.3),
                         ("ODV.out", "ATM-2.port", 0), ("OFCV.out", "OFRV-2.in", 0.1), ("OFRV-2.out", "ATM-3.port", 0),
                         ("OFIF.out", "OQD.in", 3), ("OQD.out", "RFCV.in", 0.3), ("RFCV.out", "RT.bottom", 0.5),
                         ("RT.top", "RTV.in", 0.3), ("RTV.out", "ATM-4.port", 0),
                         ("AC.out", "AR.top", 0.5), ("AR.top", "APRV.in", 0.1), ("APRV.out", "ATM-A.port", 0),
                         ("AR.top", "APR.in", 0.5), ("APR.out", "AIV.in", 0.3), ("AIV.out", "ACF.in", 0.3),
                         ("ACF.out", "OPGS.in", 1), ("OPGS.out", "OPGCV.in", 0.1), ("OPGCV.out", "OFV.out", 0.3)]:
        s.connect(a, b, length=length)
    s.sense("OFPT", "pressure", line=("OFF.out", "OFV.in"), units="psig")
    s.sense("OFT", "temperature", line=("OFF.out", "OFV.in"), units="F")
    s.sense("OIPT", "pressure", line=("OFIF.out", "OQD.in"), units="psig")
    s.sense("APT", "pressure", line=("ACF.out", "OPGS.in"), units="psig")
    s.sense("ART", "pressure", line=("AC.out", "AR.top"), units="psig")
    s.sense("FM-OF", "mass_flow", line=("OFF.out", "OFV.in"), units="kg/s")
    s.sense("M-RT", "mass", "RT", units="kg")
    s.sense("LL-RT", "level", "RT", units="%")
    s.sense("PF-OF", "purge_fraction", line=("OFV.out", "OFCV.in"), units="%")  # where the purge sweeps
    s.sense("AC-ON", "valve_position", "AC")
    return s


def ln2_wet_dress() -> System:
    """LN2 stand-in for LOX: loaded run tank, line chilldown through a bleed, then flow."""
    s = System("LN2 wet dress")
    s.add("K-N2", "tank", fluid="nitrogen", volume="49 L", pressure="3000 psi", temperature="20 C", at=[0, -220])
    s.add("REG", "regulator", cv=0.3, setpoint="250 psi", droop="20 psi", at=[150, -220])
    s.add("V-P", "valve", cv=0.3, at=[300, -220])
    s.add("TK", "tank", fluid="nitrogen", pressurant="nitrogen", volume="100 L", fill=0.85,
          pressure="40 psi", temperature="78 K", diameter="0.4 m", at=[0, 0])
    s.add("V-V", "valve", cv=0.3, at=[150, -80])
    s.add("ATM-V", "boundary", fluid="nitrogen", at=[300, -40])
    s.add("V-M", "valve", cv=4, open_time="200 ms", at=[150, 160])
    # the feed line as one volume with the tube's steel: 4 m of 1/2 in x 0.049 in tubing
    s.add("L-F", "volume", volume="0.33 L", wall_mass="1.4 kg", film_coefficient=2000, at=[300, 160])
    s.add("TEE", "tee", at=[450, 160])
    s.add("V-B", "valve", cv=0.2, at=[450, 260])
    s.add("ATM-B", "boundary", fluid="nitrogen", at=[600, 260])
    s.add("INJ", "orifice", cda="25 mm2", at=[600, 160])
    s.add("ATM", "boundary", fluid="nitrogen", at=[750, 160])
    # line lengths in m; the feed line is the volume L-F, so its ends are joined straight on
    for a, b, length in [("K-N2.top", "REG.in", 0.3), ("REG.out", "V-P.in", 0.5), ("V-P.out", "TK.top", 1),
                         ("TK.top", "V-V.in", 0.3), ("V-V.out", "ATM-V.port", 0), ("TK.bottom", "V-M.in", 0.5),
                         ("V-M.out", "L-F.a", 0), ("L-F.b", "TEE.a", 0), ("TEE.b", "INJ.in", 0.3),
                         ("INJ.out", "ATM.port", 0), ("TEE.d", "V-B.in", 0.3), ("V-B.out", "ATM-B.port", 0)]:
        s.connect(a, b, length=length)
    s.sense("PT-TK", "pressure", line=("V-P.out", "TK.top"), units="psi")
    s.sense("PT-INJ", "pressure", line=("TEE.b", "INJ.in"), along=1.0, units="psi")
    s.sense("TC-INJ", "temperature", line=("TEE.b", "INJ.in"), along=1.0, units="F")
    s.sense("TC-LINE", "wall_temperature", "L-F", units="F")
    s.sense("FM-LN2", "mass_flow", line=("V-M.out", "L-F.a"), units="kg/s")
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
    s.connect("K.bottom", "V.in", length="0.3 m")
    s.connect("V.out", "INJ.in", length="0.5 m")
    s.connect("INJ.out", "ATM.port", length=0)
    s.sense("PT-K", "pressure", line=("K.bottom", "V.in"), units="psi")
    s.sense("TC-K", "temperature", line=("K.bottom", "V.in"), units="F")
    s.sense("PT-INJ", "pressure", line=("V.out", "INJ.in"), along=1.0, units="psi")
    s.sense("FM-N2O", "mass_flow", line=("INJ.out", "ATM.port"), units="kg/s")
    s.sense("M-N2O", "mass", "K", units="kg")
    s.sense("LL-N2O", "level", "K", units="%")
    return s


def pneumatic_valves() -> System:
    """A compressor-fed air supply running a normally closed fill valve and a normally open vent."""
    s = System("Pneumatic valves")
    s.add("C", "air_pump", flow="2 g/s", start_pressure="90 psig", stop_pressure="110 psig", at=[0, -150])
    s.add("A", "pneumatic_supply", volume="2 L", fluid="air", pressure="105 psig", temperature="20 C",
          at=[150, -150])
    s.add("K", "tank", fluid="nitrogen", volume="49 L", pressure="2000 psi", temperature="20 C", at=[0, 0])
    s.add("V-FILL", "valve", cv=0.3, at=[150, 0])
    s.add("TK", "volume", volume="20 L", fluid="nitrogen", pressure="1 atm", temperature="20 C", at=[300, 0])
    s.add("V-VENT", "valve", cv=0.3, normally="open", at=[450, 0])
    s.add("ATM", "boundary", fluid="nitrogen", at=[600, 0])
    for a, b, length in [("C.out", "A.a", 0.5), ("K.top", "V-FILL.in", 0.3), ("V-FILL.out", "TK.a", 1),
                         ("TK.b", "V-VENT.in", 0.3), ("V-VENT.out", "ATM.port", 0)]:
        s.connect(a, b, length=length)
    s.sense("PT-A", "pressure", line=("C.out", "A.a"), units="psig")
    s.sense("PT-TK", "pressure", line=("V-FILL.out", "TK.a"), units="psi")
    s.sense("ZS-FILL", "valve_position", "V-FILL")
    s.sense("ZS-VENT", "valve_position", "V-VENT")
    return s


EXAMPLES = {
    "gn2-blowdown": gn2_blowdown,
    "lox-feed": lox_feed,
    "lox-purge": lox_purge,
    "ln2-wet-dress": ln2_wet_dress,
    "n2o-blowdown": n2o_blowdown,
    "nitrous-fill": nitrous_fill,
    "pneumatic-valves": pneumatic_valves,
    "test-stand": engine_test_stand,
}
