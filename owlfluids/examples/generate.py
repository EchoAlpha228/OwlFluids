"""Programmatic example systems, including the 60-component speed benchmark."""

from __future__ import annotations

from ..io.system import System


def engine_test_stand(n_purge: int = 4, pneumatic: bool = False) -> System:
    """A pressure-fed LOX/ethanol test stand with GN2 pressurization and purges.

    Roughly 60 components: supply bottles, regulators, pressurant lines with check
    valves, two pressurized propellant tanks with vents and reliefs, main valves,
    feed lines with fittings, injector orifices into a chamber, and GN2 purges of the
    feed lines and injector manifolds. With ``pneumatic`` its on/off valves run on a
    GN2 pneumatic supply regulated down from the high-pressure manifold.
    """
    s = System("LOX/ethanol test stand")
    s.settings["dt"] = "1 ms"
    add, con = s.add, s.connect

    # GN2 supply: two bottles into a manifold
    add("GN2-K1", "tank", fluid="nitrogen", volume="49 L", pressure="4500 psi", temperature="20 C")
    add("GN2-K2", "tank", fluid="nitrogen", volume="49 L", pressure="4500 psi", temperature="20 C")
    add("V-K1", "valve", cv=0.6, normally="open")
    add("V-K2", "valve", cv=0.6, normally="open")
    add("MAN-HP", "volume", volume="0.2 L")
    con("GN2-K1.port", "V-K1.in"), con("V-K1.out", "MAN-HP.a")
    con("GN2-K2.port", "V-K2.in"), con("V-K2.out", "MAN-HP.b")
    add("PT-HP-RV", "relief_valve", cv=0.2, set_pressure="4800 psi")
    add("ATM-HP", "boundary", fluid="nitrogen")
    con("MAN-HP.c", "PT-HP-RV.in"), con("PT-HP-RV.out", "ATM-HP.port")

    def press_leg(tag, setpoint):
        add(f"REG-{tag}", "regulator", cv=0.4, setpoint=setpoint, droop="30 psi")
        add(f"V-{tag}P", "valve", cv=0.5)
        add(f"CV-{tag}", "check_valve", cv=0.5)
        add(f"F-{tag}1", "fitting", kind="elbow90", diameter="8 mm")
        con(f"REG-{tag}.out", f"V-{tag}P.in"), con(f"V-{tag}P.out", f"CV-{tag}.in")
        con(f"CV-{tag}.out", f"F-{tag}1.in")

    add("TEE-HP", "tee")
    con("MAN-HP.d", "TEE-HP.a")
    press_leg("OX", "600 psi")
    press_leg("FU", "650 psi")
    con("TEE-HP.b", "REG-OX.in"), con("TEE-HP.c", "REG-FU.in")

    def prop_tank(tag, liquid, t_liq):
        add(f"TK-{tag}", "pressurized_tank", liquid=liquid, pressurant="nitrogen", volume="60 L", fill=0.85,
            pressure="80 psi", liquid_temperature=t_liq, diameter="0.3 m")
        add(f"TEE-{tag}U", "tee")
        con(f"F-{tag}1.out", f"TEE-{tag}U.a"), con(f"TEE-{tag}U.b", f"TK-{tag}.gas")
        add(f"V-{tag}V", "valve", cv=0.8, normally="open")
        add(f"ATM-{tag}", "boundary", fluid="nitrogen")
        con(f"TEE-{tag}U.c", f"V-{tag}V.in")
        con(f"V-{tag}V.out", f"ATM-{tag}.port")
        add(f"RV-{tag}", "relief_valve", cv=0.5, set_pressure="750 psi")
        add(f"ATM-{tag}R", "boundary", fluid="nitrogen")
        con(f"TEE-{tag}U.d", f"RV-{tag}.in"), con(f"RV-{tag}.out", f"ATM-{tag}R.port")
        # feed line
        add(f"F-{tag}F1", "fitting", kind="elbow90", diameter="19 mm")
        add(f"V-{tag}M", "valve", cv=6.0, open_time="150 ms")
        add(f"F-{tag}F2", "fitting", kind="tee_run", diameter="19 mm")
        add(f"INJ-{tag}", "orifice", cda="60 mm2")
        con(f"TK-{tag}.liquid", f"F-{tag}F1.in"), con(f"F-{tag}F1.out", f"V-{tag}M.in")
        con(f"V-{tag}M.out", f"F-{tag}F2.in"), con(f"F-{tag}F2.out", f"INJ-{tag}.in")
        # drain / dump
        add(f"V-{tag}D", "valve", cv=1.0)
        add(f"DUMP-{tag}", "boundary", fluid=liquid)
        add(f"TEE-{tag}D", "tee")
        con(f"V-{tag}M.in", f"TEE-{tag}D.a")  # shares the junction ahead of the main valve
        con(f"TEE-{tag}D.b", f"V-{tag}D.in")
        con(f"V-{tag}D.out", f"DUMP-{tag}.port")

    prop_tank("OX", "oxygen", "90 K")
    prop_tank("FU", "ethanol", "20 C")

    # injector manifolds feed separate chamber volumes (one fluid per volume)
    for tag, fl in (("OX", "oxygen"), ("FU", "ethanol")):
        add(f"CH-{tag}", "volume", volume="0.5 L")
        add(f"NOZ-{tag}", "orifice", cda="200 mm2")
        add(f"ATM-C{tag}", "boundary", fluid=fl)
        con(f"INJ-{tag}.out", f"CH-{tag}.a")
        con(f"CH-{tag}.b", f"NOZ-{tag}.in")
        con(f"NOZ-{tag}.out", f"ATM-C{tag}.port")

    # GN2 purges (own GN2 supplies) through check valves into each feed line below the
    # main valve, then into each injector manifold
    targets = ["F-OXF2.out", "F-FUF2.out", "CH-OX.c", "CH-FU.c"]
    for i in range(n_purge):
        tag = f"P{i + 1}"
        add(f"K-{tag}", "tank", fluid="nitrogen", volume="10 L", pressure="1000 psi", temperature="20 C")
        add(f"V-{tag}", "valve", cv=0.1)
        add(f"CK-{tag}", "check_valve", cv=0.1)
        con(f"K-{tag}.port", f"V-{tag}.in"), con(f"V-{tag}.out", f"CK-{tag}.in")
        con(f"CK-{tag}.out", targets[i % len(targets)])

    if pneumatic:
        add("REG-PN", "regulator", cv=0.1, setpoint="100 psig", droop="10 psi")
        add("PN", "pneumatic_supply", volume="2 L", pressure="100 psig")
        con("MAN-HP.d", "REG-PN.in"), con("REG-PN.out", "PN.a")

    _place(s, n_purge)

    s.sense("PT-HP", "pressure", "MAN-HP.a", units="psi")
    s.sense("PT-OX", "pressure", "TK-OX.gas", units="psi")
    s.sense("PT-FU", "pressure", "TK-FU.gas", units="psi")
    s.sense("PT-OXI", "pressure", "INJ-OX.in", units="psi")
    s.sense("PT-FUI", "pressure", "INJ-FU.in", units="psi")
    s.sense("TC-OX", "temperature", "TK-OX.liquid")
    s.sense("FM-OX", "mass_flow", "V-OXM")
    s.sense("FM-FU", "mass_flow", "V-FUM")
    return s


def _place(s: System, n_purge: int, dx: float = 150.0, dy: float = 110.0) -> None:
    """Schematic positions: supply on the left, oxidizer above, fuel below, purges at the bottom."""
    grid = {
        "GN2-K1": (0, 0), "V-K1": (2, 0), "GN2-K2": (0, 1.4), "V-K2": (2, 1.4),
        "MAN-HP": (3, 0.7), "PT-HP-RV": (3, -0.6), "ATM-HP": (4, -0.6), "TEE-HP": (5, 0.7),
        "REG-PN": (3, 2.0), "PN": (4, 2.0),
    }
    for tag, y in (("OX", -1.5), ("FU", 3.5)):
        # a gap where each line runs
        row = ["REG", None, "V-{}P", "CV", None, "F-{}1", "TEE-{}U"]
        for i, name in enumerate(row):
            if name is None:
                continue
            cid = f"{name}-{tag}" if "{}" not in name else name.format(tag)
            grid[cid] = (6 + i, y)
        grid[f"RV-{tag}"] = (12, y - 0.9)
        grid[f"ATM-{tag}R"] = (13, y - 0.9)
        grid[f"V-{tag}V"] = (13, y)
        grid[f"ATM-{tag}"] = (15, y)
        grid[f"TK-{tag}"] = (12, y + 1.3)
        feed = [None, f"F-{tag}F1", f"TEE-{tag}D", f"V-{tag}M", None, f"F-{tag}F2", None,
                f"INJ-{tag}", f"CH-{tag}", f"NOZ-{tag}", f"ATM-C{tag}"]
        for i, cid in enumerate(feed):
            if cid is not None:
                grid[cid] = (13 + i, y + 2.3)
        grid[f"V-{tag}D"] = (15, y + 3.2)
        grid[f"DUMP-{tag}"] = (17, y + 3.2)
    for i in range(n_purge):
        y = 8.5 + 1.25 * i
        for j, cid in ((0, f"K-P{i + 1}"), (1, f"V-P{i + 1}"), (3, f"CK-P{i + 1}")):
            grid[cid] = (j, y)
    for c in s.components:
        if c.id in grid:
            gx, gy = grid[c.id]
            c.position = (gx * dx, gy * dy)
