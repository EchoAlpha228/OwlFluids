"""Burst disks: shut below their burst pressure, then open for good, and a reset puts a new disk in."""
from pathlib import Path

from owlfluids.io.system import System
from owlfluids.sim.simulation import Simulation

PSI = 6894.757


def _line(burst):
    s = System("bd")
    s.add("K", "tank", fluid="nitrogen", volume="10 L", pressure="2000 psi", temperature="293 K")
    s.add("REG", "regulator", cv=0.2, setpoint="600 psi")
    s.add("BD", "burst_disk", cv=0.2, burst_pressure=burst)
    s.connect("K.top", "REG.in")
    s.connect("REG.out", "BD.in")  # nothing on its outlet: it vents to the atmosphere
    return Simulation(s)


def test_a_burst_disk_holds_below_its_burst_pressure():
    sim = _line("1000 psi")
    sim.run(1.0)
    assert not sim.burst("BD") and sim.mass_flow("BD") == 0.0


def test_a_burst_disk_ruptures_stays_open_and_resets():
    sim = _line("300 psi")
    sim.run(0.5)
    assert sim.burst("BD") and sim.mass_flow("BD") > 0.01
    sim.set_setpoint("REG", "100 psi")  # well under the burst pressure: a ruptured disk still flows
    sim.run(1.0)
    assert sim.burst("BD") and sim.mass_flow("BD") > 0.0
    sim.reset()
    assert not sim.burst("BD")
    assert sim.burst_disks() == ["BD"]


def test_the_lyra_feed_system_builds_and_presses():
    path = Path(__file__).resolve().parents[1] / "systems" / "lyra" / "lyra-feed.yaml"
    sim = Simulation.from_file(path)
    assert sim.burst_disks() == ["OFBD"]
    sim.open("NI-F", "FPV")
    sim.run(3.0)
    assert sim.read("FTPT") > 100 * PSI and not sim.burst("OFBD")
