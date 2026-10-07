"""Realistic sensor noise and lag, level sensors and default display units."""
import numpy as np
import pytest

from owlfluids import units
from owlfluids.examples import lox_feed, n2o_blowdown
from owlfluids.io.system import System
from owlfluids.sim.simulation import Simulation


def _reads(sim, sid, n=300):
    return np.array([sim.read(sid) for _ in range(n)])


@pytest.fixture(scope="module")
def lox():
    return Simulation(lox_feed(), step_budget=200)


def test_noise_is_repeatable_and_can_be_turned_off(lox):
    lox.reset()
    a = _reads(lox, "PT-TK", 20)
    lox.reset()
    assert np.array_equal(a, _reads(lox, "PT-TK", 20))  # same seed, same noise
    assert a.std() > 0
    quiet = lox_feed()
    quiet.settings["sensor_noise"] = "off"
    exact = Simulation(quiet, step_budget=200)
    assert np.all(_reads(exact, "PT-TK", 5) == exact.pressure("TK.gas"))
    one = lox_feed()
    one.sensors[0].noise = 0.0  # a sensor can opt out on its own
    assert np.all(_reads(Simulation(one, step_budget=200), "PT-TK", 5) == exact.pressure("TK.gas"))


def test_noise_follows_what_the_sensor_sees(lox):
    lox.reset()
    lox.run(0.1)
    still = _reads(lox, "PT-INJ").std()
    assert np.all(_reads(lox, "FM-LOX", 20) == 0.0)  # a meter on a still line reads 0
    lox.open("V-M")
    lox.run(1.0)
    flowing = _reads(lox, "PT-INJ").std()
    assert flowing > 1.5 * still
    fm = _reads(lox, "FM-LOX")
    assert 0.003 < fm.std() / fm.mean() < 0.03  # about 1% of reading


def test_thermocouples_lag_and_read_fahrenheit(lox):
    assert units.DEFAULT_UNITS["temperature"] == "F"
    assert next(s for s in lox_feed().sensors if s.id == "TC-INJ").units == "F"
    assert units.convert(77.15, "F") == pytest.approx(-320.8, abs=0.1)
    lox.reset()
    lox.open("V-M")
    lox.run(0.4)
    exact = lox.temperature("INJ.in")
    assert abs(np.mean(_reads(lox, "TC-INJ")) - exact) > 1.0  # still catching up with the cold LOX


def test_level_sensors():
    sim = Simulation(lox_feed(), step_budget=200)
    assert np.mean(_reads(sim, "LL-LOX")) == pytest.approx(0.85, abs=0.005)
    assert units.convert(0.85, "%") == pytest.approx(85.0)
    n2o = Simulation(n2o_blowdown(), step_budget=200)
    assert np.mean(_reads(n2o, "LL-N2O")) == pytest.approx(0.8, abs=0.01)
    bad = lox_feed()
    bad.sense("LL-BAD", "level", "V-M")
    with pytest.raises(ValueError, match="tank"):
        Simulation(bad, step_budget=200)


def test_sensor_noise_settings_round_trip():
    s = System.from_dict({"name": "x", "components": [], "sensors": [
        {"id": "A", "type": "pressure", "at": "K.port"},
        {"id": "B", "type": "pressure", "at": "K.port", "noise": "off"},
        {"id": "C", "type": "temperature", "at": "K.port", "lag": 0.1, "noise": 0.5}]})
    a, b, c = s.sensors
    assert (a.noise, a.lag) == (None, None) and b.noise == 0.0 and (c.lag, c.noise) == (0.1, 0.5)
    d = System.from_dict(s.to_dict())
    assert [(x.lag, x.noise) for x in d.sensors] == [(None, None), (None, 0.0), (0.1, 0.5)]
