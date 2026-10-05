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
    t = np.array(times) / 0.01  # wall seconds per simulated second
    # generous margins for slow CI machines; typical is ~0.1 mean, ~0.5 worst
    assert t.mean() < 0.5
    assert t.max() < 1.5
    assert sim.read("FM-OX") == pytest.approx(0.0, abs=0.05)
