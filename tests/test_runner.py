"""Real-time runner: pacing changes, command errors, pause and shutdown."""
import time

import pytest

from owlfluids.examples import gn2_blowdown
from owlfluids.io.logger import open_logger, read_log
from owlfluids.sim.realtime import Runner
from owlfluids.sim.simulation import Simulation


@pytest.fixture(scope="module")
def sim():
    return Simulation(gn2_blowdown())


def test_rate_change_keeps_pacing(sim):
    sim.reset()
    runner = Runner(sim, rate=0)  # as fast as possible first: sim time runs far ahead of the wall clock
    runner.start()
    try:
        runner.wait_until(lambda f: f.t >= 0.5)
        runner.set_rate(1.0)
        time.sleep(0.05)
        t0, w0 = runner.frame.t, time.perf_counter()
        time.sleep(0.4)
        advanced = runner.frame.t - t0
        assert advanced == pytest.approx(time.perf_counter() - w0, abs=0.12)  # real time, not frozen
    finally:
        t = time.perf_counter()
        runner.stop()
        assert time.perf_counter() - t < 0.5
    assert not runner.running


def test_bad_command_and_listener_are_reported_not_fatal(sim):
    sim.reset()
    runner = Runner(sim, rate=0)
    with pytest.raises(KeyError):
        runner.set_boundary("NOPE", pressure="1 atm")
    calls = []

    def broken(frame):
        calls.append(frame.t)
        raise ValueError("boom")

    runner.add_listener(broken)
    runner.start()
    try:
        runner.submit(lambda s: 1 / 0)
        runner.wait(0.05)
        assert runner.error is None and runner.running
        assert len(calls) == 1  # dropped after its first failure
        assert any("ZeroDivisionError" in w for w in runner.warnings)
        assert any("boom" in w for w in runner.warnings)
    finally:
        runner.stop()


def test_paused_runner_shows_commands(sim):
    sim.reset()
    runner = Runner(sim, rate=1.0)
    runner.start()
    try:
        runner.pause()
        time.sleep(0.05)
        runner.open("V1")
        deadline = time.perf_counter() + 1.0
        while runner.frame.commands["V1"] != 1.0 and time.perf_counter() < deadline:
            time.sleep(0.01)
        assert runner.frame.commands["V1"] == 1.0
    finally:
        runner.stop()


def test_stop_runs_queued_log_close(sim, tmp_path):
    sim.reset()
    runner = Runner(sim, rate=0)
    path = tmp_path / "run.csv"
    log = open_logger(path, runner)
    runner.start()
    runner.wait(0.05)
    runner.submit(lambda s: log.close())
    runner.stop()
    assert log._f.closed
    names, data = read_log(path)
    assert data.shape[0] > 0


def test_boundary_changes_apply_in_order(sim):
    sim.reset()
    runner = Runner(sim, rate=0)
    runner.start()
    try:
        runner.set_boundary("ATM", pressure="50 psi")
        runner.set_boundary("ATM", temperature="250 K")  # must keep the 50 psi
        runner.wait(0.02)
        i, p, t, _, _ = sim.boundary_state("ATM")
        assert p == pytest.approx(50 * 6894.757, rel=1e-6)
        assert t == pytest.approx(250.0)
    finally:
        runner.stop()
