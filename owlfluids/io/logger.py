"""Record runs to CSV or HDF5, and read them back for replay and comparison.

A logger is a runner listener: ``runner.add_listener(CsvLogger("run.csv", runner))``.
Columns are simulated time, every sensor (in its display units) and every
valve's command and position.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .. import units


def _columns(runner) -> list[tuple[str, str]]:
    sim = runner.sim
    cols = []
    for s in sim.system.sensors:
        unit = s.units or units.DEFAULT_UNITS.get(s.type, "")
        cols.append((s.id, unit))
    for v in sim.valves():
        cols.append((f"{v}.cmd", ""))
        cols.append((f"{v}.pos", ""))
    return cols


def _row(frame, runner, cols) -> list[float]:
    sim = runner.sim
    out = [frame.t]
    for (_name, unit), s in zip(cols, sim.system.sensors, strict=False):
        out.append(units.convert(frame.sensors[s.id], unit))
    for v in sim.valves():
        out.append(frame.commands[v])
        out.append(frame.valves[v])
    return out


class CsvLogger:
    def __init__(self, path: str | Path, runner, interval: float = 0.01):
        self.path = Path(path)
        self.runner = runner
        self.interval = interval
        self.cols = _columns(runner)
        self._f = self.path.open("w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(["t [s]"] + [f"{n} [{u}]" if u else n for n, u in self.cols])
        self._next = -1.0

    def __call__(self, frame) -> None:
        if frame.t + 1e-12 < self._next:
            return
        self._next = frame.t + self.interval
        self._w.writerow([f"{v:.6g}" for v in _row(frame, self.runner, self.cols)])

    def close(self) -> None:
        self.runner.remove_listener(self)
        self._f.close()


class Hdf5Logger:
    """Same columns as CsvLogger, in an extendable HDF5 dataset ``data``."""

    def __init__(self, path: str | Path, runner, interval: float = 0.01, flush_rows: int = 500):
        import h5py

        self.runner = runner
        self.interval = interval
        self.cols = _columns(runner)
        self._h = h5py.File(path, "w")
        names = ["t"] + [n for n, _ in self.cols]
        self._d = self._h.create_dataset("data", shape=(0, len(names)), maxshape=(None, len(names)),
                                         dtype="f8", chunks=(1024, len(names)))
        self._d.attrs["columns"] = names
        self._d.attrs["units"] = ["s"] + [u for _, u in self.cols]
        self._h.attrs["system"] = runner.sim.system.name
        self._buf: list[list[float]] = []
        self._flush_rows = flush_rows
        self._next = -1.0

    def __call__(self, frame) -> None:
        if frame.t + 1e-12 < self._next:
            return
        self._next = frame.t + self.interval
        self._buf.append(_row(frame, self.runner, self.cols))
        if len(self._buf) >= self._flush_rows:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        a = np.asarray(self._buf)
        n = self._d.shape[0]
        self._d.resize(n + len(a), axis=0)
        self._d[n:] = a
        self._buf.clear()
        self._h.flush()

    def close(self) -> None:
        self.runner.remove_listener(self)
        self.flush()
        self._h.close()


def open_logger(path: str | Path, runner, interval: float = 0.01):
    """CSV or HDF5 by file extension; registered with the runner."""
    p = Path(path)
    log = Hdf5Logger(p, runner, interval) if p.suffix.lower() in (".h5", ".hdf5") else CsvLogger(p, runner, interval)
    runner.add_listener(log)
    return log


def read_log(path: str | Path) -> tuple[list[str], np.ndarray]:
    """(column names, data) from a CSV or HDF5 log, or any CSV with a header row.

    Real test data in CSV form loads the same way, so it can be plotted over a run.
    """
    p = Path(path)
    if p.suffix.lower() in (".h5", ".hdf5"):
        import h5py

        with h5py.File(p, "r") as h:
            return [str(c) for c in h["data"].attrs["columns"]], h["data"][:]
    with p.open(newline="") as f:
        header = next(csv.reader(f))
    data = np.genfromtxt(p, delimiter=",", skip_header=1, ndmin=2)
    names = [h.split(" [")[0].strip() for h in header]
    return names, data
