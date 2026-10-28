"""Regulator behaviors: droop, lockup, creep and supply pressure effect, each on or off."""
import pytest

from owlfluids import units
from owlfluids.io.system import System
from owlfluids.sim.realtime import Runner
from owlfluids.sim.simulation import Simulation


def regulated(tank="50 L", **reg):
    """A bottle through a regulator into a 1 L volume, vented by a hand valve."""
    s = System("regulated")
    s.add("TK", "tank", fluid="nitrogen", volume=tank, pressure="3000 psi", temperature="300 K")
    s.add("REG", "regulator", cv=0.1, setpoint="300 psig", droop="20 psi", **reg)
    s.add("VOL", "volume", volume="1 L")
    s.add("MV", "manual_valve", cv=0.3, open_time="0.1 s")
    s.add("AMB", "boundary", fluid="nitrogen", pressure="1 bar")
    s.connect("TK.port", "REG.in")
    s.connect("REG.out", "VOL.a")
    s.connect("VOL.b", "MV.in")
    s.connect("MV.out", "AMB.port")
    s.sense("PT-OUT", "pressure", "VOL.a", noise=0.0, lag=0.0)
    s.sense("PT-IN", "pressure", "TK.port", noise=0.0, lag=0.0)
    return s


def psig(sim, sensor="PT-OUT"):
    return units.convert(sim.read(sensor), "psig")


def flowing_then_shut(sim):
    """Outlet pressure flowing, then 2 s after the vent shuts."""
    sim.run(0.5)
    sim.command("MV", 1.0)
    sim.run(2.0)
    flowing = psig(sim)
    sim.command("MV", 0.0)
    sim.run(2.0)
    return flowing, psig(sim)


def test_defaults_keep_droop_only():
    sim = Simulation(regulated())
    assert sim.regulator_behaviors("REG") == {"droop": True, "lockup": False, "creep": False,
                                              "supply_effect": False}


def test_droop_off_holds_the_setpoint_while_flowing():
    with_droop, _ = flowing_then_shut(Simulation(regulated(), step_budget=20))
    ideal, _ = flowing_then_shut(Simulation(regulated(droop_on="off"), step_budget=20))
    assert with_droop < 297.0
    assert ideal == pytest.approx(300.0, abs=0.5)


def test_lockup_climbs_above_setpoint_when_flow_stops():
    _, plain = flowing_then_shut(Simulation(regulated(), step_budget=20))
    flowing, locked = flowing_then_shut(Simulation(regulated(lockup_on="on", lockup="8 psi"), step_budget=20))
    assert plain == pytest.approx(300.0, abs=1.0)
    assert locked == pytest.approx(308.0, abs=1.5)
    assert flowing < 300.0  # flowing it droops as before


def test_creep_lets_a_dead_ended_outlet_rise():
    def rise(**reg):
        sim = Simulation(regulated(**reg), step_budget=20)
        sim.run(1.0)
        start = psig(sim)
        sim.run(30.0)
        return psig(sim) - start
    assert abs(rise()) < 0.2
    assert rise(creep_on="on") > 1.0  # default 5e-5 of its flow area
    assert rise(creep_on="on", creep=1e-4) > 3.0


def test_supply_effect_raises_the_outlet_as_the_bottle_blows_down():
    def blowdown(**reg):
        sim = Simulation(regulated(tank="2 L", droop_on="off", **reg), step_budget=20)
        sim.command("MV", 0.3)
        sim.run(1.0)
        p_in, p_out = psig(sim, "PT-IN"), psig(sim)
        sim.run(8.0)
        return p_in - psig(sim, "PT-IN"), psig(sim) - p_out
    drop, rise = blowdown()
    assert drop > 500.0 and abs(rise) < 1.0
    drop, rise = blowdown(supply_effect_on="on", supply_effect=0.02)
    assert rise == pytest.approx(0.02 * drop, rel=0.2)


def test_behaviors_turn_on_and_off_during_a_run_and_reset():
    sim = Simulation(regulated(), step_budget=20)
    sim.run(1.0)
    start = psig(sim)
    sim.set_regulator_behavior("REG", "creep", True)
    sim.set_regulator_behavior("REG", "creep", True)  # again: no change
    sim.run(20.0)
    assert psig(sim) - start > 1.0
    sim.set_regulator_behavior("REG", "creep", False)
    held = psig(sim)
    sim.run(5.0)
    assert psig(sim) == pytest.approx(held, abs=0.1)
    assert sim.regulator_behaviors("REG")["creep"] is False
    with pytest.raises(KeyError):
        sim.set_regulator_behavior("REG", "hysteresis", True)
    with pytest.raises(KeyError):
        sim.set_regulator_behavior("MV", "creep", True)
    sim.set_regulator_behavior("REG", "droop", False)
    sim.reset()
    assert sim.regulator_behaviors("REG")["droop"] is True


def test_turning_supply_effect_on_holds_the_outlet_where_it_is():
    sim = Simulation(regulated(tank="2 L", droop_on="off"), step_budget=20)
    sim.command("MV", 1.0)
    sim.run(3.0)
    before = psig(sim)
    sim.set_regulator_behavior("REG", "supply_effect", True)  # counts from the inlet pressure now
    sim.run(0.5)
    assert psig(sim) == pytest.approx(before, abs=2.0)


def test_switches_read_from_yaml_and_text():
    s = System.from_dict({"name": "r", "components": [
        {"id": "TK", "type": "tank", "fluid": "nitrogen", "volume": "5 L", "pressure": "1000 psi",
         "temperature": "300 K"},
        {"id": "REG", "type": "regulator", "cv": 0.1, "setpoint": "200 psig", "droop_on": False,
         "lockup_on": True, "creep_on": "yes", "supply_effect_on": "off"},
        {"id": "VOL", "type": "volume", "volume": "1 L"},
    ], "connections": [["TK.port", "REG.in"], ["REG.out", "VOL.a"]]})
    sim = Simulation(s)
    assert sim.regulator_behaviors("REG") == {"droop": False, "lockup": True, "creep": True,
                                              "supply_effect": False}
    s.component("REG").params["creep_on"] = "sometimes"
    with pytest.raises(ValueError, match="creep_on"):
        Simulation(s)


def test_runner_publishes_and_applies_behaviors():
    runner = Runner(Simulation(regulated()), rate=0)
    assert runner.frame.behaviors["REG"]["lockup"] is False
    runner.set_regulator_behavior("REG", "lockup", True)  # stopped: applied at once
    assert runner.frame.behaviors["REG"]["lockup"] is True
    with pytest.raises(KeyError):
        runner.set_regulator_behavior("REG", "wobble", True)
