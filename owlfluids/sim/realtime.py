"""Run a Simulation against the wall clock.

The runner owns the simulation and advances it on a background thread in small
chunks (5 ms by default), sleeping between chunks so simulated time tracks wall
time. The compiled kernel releases the GIL, so a GUI or a driver on other
threads stays responsive.

Everything that changes the simulation goes through the runner (``open``,
``close``, ``command``, ``submit``): commands are queued and applied between
chunks, so they never race with a step. Readers use ``frame`` (the latest values,
replaced atomically after every chunk) or ``history`` (a ring buffer).

    runner = Runner(Simulation.from_file("stand.yaml"))
    runner.start()
    runner.open("V-OXM")
    runner.wait(2.0)
    print(runner.frame.sensors["PT-OX"])
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .simulation import Simulation


@dataclass(frozen=True)
class Frame:
    """One consistent view of the simulation, taken after a chunk."""

    t: float  # simulated time, s
    wall: float  # wall time since start, s
    sensors: dict[str, float]  # SI units
    valves: dict[str, float]  # actual positions 0-1
    commands: dict[str, float]  # commanded positions 0-1
    lag: float = 0.0  # how far simulated time trails the wall clock, s
    load: float = 0.0  # share of wall time spent computing (recent average)
    stats: dict = field(default_factory=dict)


class History:
    """Fixed-length ring buffer of sensor and valve values, sampled every ``interval`` s."""

    def __init__(self, names: list[str], seconds: float, interval: float):
        self.names = list(names)
        self.interval = interval
        self.n = max(2, int(round(seconds / interval)))
        self._t = np.full(self.n, np.nan)
        self._v = np.full((self.n, len(names)), np.nan)
        self._i = 0
        self._count = 0
        self._lock = threading.Lock()

    def add(self, t: float, values) -> None:
        with self._lock:
            self._t[self._i] = t
            self._v[self._i] = values
            self._i = (self._i + 1) % self.n
            self._count = min(self._count + 1, self.n)

    def clear(self) -> None:
        with self._lock:
            self._t[:] = np.nan
            self._v[:] = np.nan
            self._i = self._count = 0

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """(t, values) in time order; values has one column per name."""
        with self._lock:
            if self._count < self.n:
                return self._t[:self._count].copy(), self._v[:self._count].copy()
            idx = np.r_[self._i:self.n, 0:self._i]
            return self._t[idx], self._v[idx]


class Runner:
    def __init__(self, sim: Simulation, rate: float = 1.0, chunk: float = 0.005,
                 history_seconds: float = 120.0, history_interval: float = 0.01, max_lag: float = 0.25):
        self.sim = sim
        self.rate = rate  # simulated seconds per wall second; 0 = as fast as possible
        self.chunk_steps = max(1, int(round(chunk / sim.dt)))
        self.max_lag = max_lag
        self._cmds: queue.SimpleQueue = queue.SimpleQueue()
        self._listeners: list[Callable[[Frame], None]] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._tick = threading.Condition()
        self.error: BaseException | None = None
        self.slips = 0  # times the runner fell behind by more than max_lag and let the clock slip
        names = list(sim.sensors()) + [f"valve:{v}" for v in sim.valves()]
        self.history = History(names, history_seconds, history_interval)
        self._next_record = 0.0
        self.frame = self._make_frame(0.0, 0.0, 0.0)

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="owlfluids-runner", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._paused.clear()
        if self._thread is not None:
            self._thread.join(timeout)
        self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def set_rate(self, rate: float) -> None:
        """1.0 = real time, 0.5 = half speed, 0 = as fast as the CPU allows."""
        self.submit(lambda sim: setattr(self, "rate", float(rate)))

    def reset(self) -> None:
        def do(sim: Simulation) -> None:
            sim.reset()
            self.history.clear()
            self._next_record = 0.0
        self.submit(do)

    # ------------------------------------------------------------------ commands
    def submit(self, fn: Callable[[Simulation], object]) -> None:
        """Run ``fn(sim)`` on the runner thread before the next chunk (or now, if stopped)."""
        if self.running:
            self._cmds.put(fn)
        else:
            fn(self.sim)
            self.frame = self._make_frame(self.frame.wall, 0.0, self.frame.load)

    def open(self, *valves: str) -> None:
        for v in valves:
            self.sim._valve(v)  # fail now, in the caller, on a bad name
        self.submit(lambda sim: sim.open(*valves))

    def close(self, *valves: str) -> None:
        for v in valves:
            self.sim._valve(v)
        self.submit(lambda sim: sim.close(*valves))

    def command(self, valve: str, position) -> None:
        self.sim._valve(valve)
        self.submit(lambda sim: sim.command(valve, position))

    def set_boundary(self, comp: str, pressure=None, temperature=None) -> None:
        self.submit(lambda sim: sim.set_boundary(comp, pressure, temperature))

    def add_listener(self, fn: Callable[[Frame], None]) -> None:
        """Call ``fn(frame)`` on the runner thread after every chunk (loggers, drivers)."""
        self._listeners.append(fn)

    def remove_listener(self, fn: Callable[[Frame], None]) -> None:
        if fn in self._listeners:
            self._listeners.remove(fn)

    # ------------------------------------------------------------------ waiting (scripts)
    def wait(self, seconds: float) -> None:
        """Block until ``seconds`` of simulated time have passed."""
        end = self.frame.t + seconds
        self.wait_until(lambda f: f.t >= end - 1e-9, timeout=None)

    def wait_until(self, predicate: Callable[[Frame], bool], timeout: float | None = None) -> bool:
        """Block until ``predicate(frame)`` holds; False if ``timeout`` simulated seconds pass first."""
        end = None if timeout is None else self.frame.t + timeout
        with self._tick:
            while True:
                f = self.frame
                if predicate(f):
                    return True
                if end is not None and f.t >= end:
                    return False
                if self.error is not None:
                    raise RuntimeError("the simulation stopped with an error") from self.error
                if not self.running:
                    raise RuntimeError("the runner is not running")
                self._tick.wait(0.1)

    # ------------------------------------------------------------------ loop
    def _apply_commands(self) -> None:
        while True:
            try:
                fn = self._cmds.get_nowait()
            except queue.Empty:
                return
            fn(self.sim)

    def _run(self) -> None:
        _high_resolution_timer(True)
        try:
            sim = self.sim
            wall0 = time.perf_counter()
            sim_t0 = sim.t
            busy = 0.0
            load = 0.0
            while not self._stop.is_set():
                self._apply_commands()
                if self._paused.is_set():
                    time.sleep(0.01)
                    wall0 = time.perf_counter()
                    sim_t0 = sim.t
                    continue
                t_start = time.perf_counter()
                sim.step(self.chunk_steps)
                t_end = time.perf_counter()
                busy = t_end - t_start
                elapsed = t_end - wall0
                load = 0.95 * load + 0.05 * (busy / max(self.chunk_steps * sim.dt / max(self.rate, 1e-9), 1e-9))
                lag = 0.0
                if self.rate > 0:
                    target_wall = (sim.t - sim_t0) / self.rate
                    ahead = target_wall - elapsed
                    lag = max(-ahead * self.rate, 0.0)
                    if -ahead > self.max_lag:
                        # too far behind to catch up: let the clock slip
                        self.slips += 1
                        wall0 = t_end - (sim.t - sim_t0) / self.rate
                        lag = 0.0
                frame = self._make_frame(t_end - wall0, lag, load)
                self._publish(frame)
                if self.rate > 0:
                    ahead = (sim.t - sim_t0) / self.rate - (time.perf_counter() - wall0)
                    _sleep(ahead)
        except BaseException as e:  # surfaced to waiters and the GUI
            self.error = e
            with self._tick:
                self._tick.notify_all()
        finally:
            _high_resolution_timer(False)

    def _make_frame(self, wall: float, lag: float, load: float) -> Frame:
        sim = self.sim
        valves = {v: sim.valve_position(v) for v in sim.valves()}
        return Frame(t=sim.t, wall=wall, sensors=sim.sensors(), valves=valves,
                     commands={v: sim.valve_command(v) for v in sim.valves()}, lag=lag, load=load,
                     stats=sim.solver_stats())

    def _publish(self, frame: Frame) -> None:
        self.frame = frame
        if frame.t + 1e-12 >= self._next_record:
            self.history.add(frame.t, list(frame.sensors.values()) + list(frame.valves.values()))
            self._next_record = frame.t + self.history.interval
        for fn in list(self._listeners):
            fn(frame)
        with self._tick:
            self._tick.notify_all()


def _sleep(seconds: float) -> None:
    """Sleep precisely: coarse sleep, then spin for the last fraction of a millisecond."""
    if seconds <= 0:
        return
    end = time.perf_counter() + seconds
    if seconds > 0.002:
        time.sleep(seconds - 0.0015)
    while time.perf_counter() < end:
        pass


def _high_resolution_timer(on: bool) -> None:
    """Ask Windows for 1 ms timer resolution while running (no-op elsewhere)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        winmm = ctypes.WinDLL("winmm")
        (winmm.timeBeginPeriod if on else winmm.timeEndPeriod)(1)
    except OSError:
        pass
