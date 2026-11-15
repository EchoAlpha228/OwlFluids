"""The Lyra feed system file builds, presses, and its N2O run tank behaves like a nitrous tank."""
from pathlib import Path

from owlfluids.sim.simulation import Simulation

PSI = 6894.757


def test_the_lyra_feed_system_builds_and_presses():
    path = Path(__file__).resolve().parents[1] / "systems" / "lyra" / "lyra-feed.yaml"
    sim = Simulation.from_file(path)
    assert sim.burst_disks() == ["OFBD"]
    sim.open("NI-F", "FPV")
    sim.run(3.0)
    assert sim.read("FTPT") > 100 * PSI and not sim.burst("OFBD")


def test_the_lyra_run_tank_boils_to_hold_its_vapor_pressure_when_vented():
    """N2O vapor and GN2 share the run tank's ullage: venting it boils the liquid, which cools."""
    path = Path(__file__).resolve().parents[1] / "systems" / "lyra" / "lyra-feed.yaml"
    sim = Simulation.from_file(path)
    p0 = sim.pressure("N2O.top")
    t0 = sim.temperature("N2O.bottom")
    sim.open("OTFS")
    sim.run(5.0)
    assert sim.pressure("N2O.top") > 0.85 * p0  # a GN2 ullage alone would have fallen to a third
    assert sim.temperature("N2O.bottom") < t0 - 1.0
