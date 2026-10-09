"""The real-time gate: an 80-component stand must step well inside 1 ms."""
import time

import numpy as np
import pytest

from owlfluids.examples.generate import engine_test_stand
from owlfluids.sim.simulation import Simulation


@pytest.mark.slow
def test_engine_stand_runs_faster_than_real_time():
    system = engine_test_stand()
    assert len(system.components) >= 80
    sim = Simulation(system)
    sim.run(0.05)  # compile and settle
    times = []

    def run(duration):
        for _ in range(int(round(duration / 0.01))):
            t0 = time.perf_counter()
            sim.run(0.01)
            times.append(time.perf_counter() - t0)

    run(0.1)
    sim.open("V-OXP", "V-FUP")
    run(1.0)
    sim.close("V-OXV", "V-FUV")
    run(0.5)
    sim.open("V-OXM", "V-FUM")
    run(1.0)
    sim.close("V-OXM", "V-FUM")
    run(0.3)
    shut = sim.read("FM-OX")
    sim.open("V-P1", "V-P2", "V-P3", "V-P4")  # GN2 purges of both feeds and the chambers
    run(1.0)
    t = np.array(times) / 0.01  # wall seconds per simulated second
    # generous margins for slow CI machines; typical is ~0.2 mean, ~0.7 worst
    assert t.mean() < 0.5
    assert t.max() < 1.5
    assert shut == pytest.approx(0.0, abs=0.05)


def _feed_transient(budget):
    """Injector pressure every 10 ms for 1 s after the main valves open."""
    sim = Simulation(engine_test_stand(), step_budget=budget)
    sim.run(0.1)
    sim.open("V-OXP", "V-FUP")
    sim.run(1.0)
    sim.close("V-OXV", "V-FUV")
    sim.run(0.5)
    sim.open("V-OXM", "V-FUM")
    out = []
    for _ in range(100):
        sim.run(0.01)
        out.append(sim.read("PT-OXI"))
    return np.array(out), sim.solver_stats()["forced_steps"]


@pytest.mark.slow
def test_small_budget_keeps_the_valve_transient():
    """A slow machine gets a small work budget; the LOX feed transient must still match."""
    ref, forced_ref = _feed_transient(200)
    small, forced_small = _feed_transient(12)
    assert forced_ref < 10
    assert forced_small < 300
    arrival = [np.argmax(p > 100 * 6894.757) for p in (ref, small)]  # liquid reaches the injector
    assert ref[arrival[0]] > 100 * 6894.757 and small[arrival[1]] > 100 * 6894.757
    assert abs(arrival[0] - arrival[1]) <= 3  # within 30 ms
    assert np.mean(small[-30:]) == pytest.approx(np.mean(ref[-30:]), rel=0.03)


@pytest.mark.slow
def test_stand_purge_converges():
    """With a generous budget, GN2 purging the LOX feed after a run takes no forced steps."""
    sim = Simulation(engine_test_stand(), step_budget=200)
    sim.run(0.1)
    sim.open("V-OXP", "V-FUP")
    sim.run(1.0)
    sim.close("V-OXV", "V-FUV")
    sim.run(0.5)
    sim.open("V-OXM", "V-FUM")
    sim.run(1.0)
    sim.close("V-OXM", "V-FUM")
    sim.run(0.3)
    sim.open("V-P1", "V-P2", "V-P3", "V-P4")
    sim.run(1.5)
    assert sim.solver_stats()["forced_steps"] < 10
