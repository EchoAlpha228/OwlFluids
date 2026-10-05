"""Headless simulation: build a network, command valves, advance time, read sensors.

This is the scripting API that tests use directly. ``Simulation`` runs as fast as
the CPU allows; ``owlfluids.sim.realtime`` paces the same object to the wall clock.

    sim = Simulation.from_file("examples/gn2_blowdown.yaml")
    sim.open("V-1")
    sim.run_until(lambda s: s.read("PT-1") < 500 * PSI, timeout=60)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .. import units
from ..core import kernel
from ..core import layout as L
from ..core.network import Network, build_network
from ..io.system import System

SENSOR_TYPES = ("pressure", "temperature", "mass_flow", "valve_position", "quality", "density", "mass",
                "wall_temperature")


class Simulation:
    def __init__(self, system: System, seed: int = 0):
        self.system = system
        self.net: Network = build_network(system)
        self.dt = self.net.dt
        self.rng = np.random.default_rng(seed)
        self._alloc()
        self.reset()

    @classmethod
    def from_file(cls, path: str | Path, **kw) -> Simulation:
        return cls(System.load(path), **kw)

    # ------------------------------------------------------------------ setup
    def _alloc(self) -> None:
        net = self.net
        n = net.n_states
        nn, nl = net.NI.shape[0], net.LI.shape[0]
        self.W = np.zeros((nn, kernel.N_W))
        self.nout = np.zeros((nn, L.N_NO))
        self.lout = np.zeros((nl, L.N_LO))
        self.FN = np.full((nl, 4), -1, dtype=np.int64)
        self.FV = np.zeros((nl, 9))
        self.NS = np.zeros((nn, 2))
        self.jvals = np.zeros(len(net.col_rows))
        self.f0 = np.zeros(n)
        self.st = (net.col_ptr, net.col_rows, net.color_ptr, net.color_cols, net.sowner, net.nl_ptr, net.nl_idx)
        self.sc = (np.zeros(n), np.zeros(n), np.zeros(nn, dtype=np.int64), np.zeros(nl, dtype=np.int64),
                   np.zeros(nn, dtype=np.int64), np.zeros(nl, dtype=np.int64), np.zeros((nn, kernel.N_W)),
                   np.zeros(4, dtype=np.int64), np.zeros(9), np.zeros(2), np.zeros(1, dtype=np.int64))
        self.ab = np.zeros((n, 2 * net.kl + net.ku + 1))
        self.piv = np.zeros(n, dtype=np.int64)
        self.bvec = np.zeros(n)
        self.xnew = np.zeros(n)
        self._empty_h = np.zeros((0, max(nn, 1)))
        self._empty_l = np.zeros((0, max(nl, 1)))
        self.fl = self.net.fluids.arrays()
        self.step_budget = self._step_budget()

    def _step_budget(self, share: float = 0.35) -> float:
        """Work one step may spend, in right-hand-side evaluations.

        Sized so a step uses at most ``share`` of dt on a typical desktop CPU
        (about 0.1 us per node or link per evaluation). It depends only on the
        network, so runs are reproducible on any machine.
        """
        elements = self.net.NI.shape[0] + self.net.LI.shape[0]
        return float(min(max(share * self.dt / (1e-7 * elements), 30.0), 1000.0))

    def reset(self) -> None:
        """Back to the initial conditions of the system."""
        net = self.net
        self.x = net.x0.copy()
        self.LV = net.LV.copy()
        self.NF = net.NF.copy()
        self.t = 0.0
        self.steps = 0
        self.ctl = np.zeros(10)
        self.ctl[1] = -1.0
        self.ctl[4] = 1.0
        self.ctl[9] = self.step_budget
        self.stats = np.zeros(6)
        self._sensors = {}
        for s in self.system.sensors:
            self._sensors[s.id] = _SensorState(s, self._sensor_source(s))
        self._advance(0)
        for st in self._sensors.values():
            st.value = st.held = self._raw(st)

    # ------------------------------------------------------------------ time
    def step(self, n: int = 1) -> None:
        self._advance(n)

    def run(self, duration: float, chunk: float = 0.01) -> None:
        """Advance simulated time by ``duration`` seconds."""
        steps = int(round(units.parse(duration, "time") / self.dt))
        per = max(1, int(round(chunk / self.dt)))
        while steps > 0:
            k = min(per, steps)
            self._advance(k)
            steps -= k

    def run_until(self, predicate: Callable[[Simulation], bool], timeout: float, chunk: float = 0.001) -> bool:
        """Advance until ``predicate(sim)`` is true; False if ``timeout`` (s) passes first."""
        end = self.t + units.parse(timeout, "time")
        per = max(1, int(round(chunk / self.dt)))
        while self.t < end - 1e-12:
            if predicate(self):
                return True
            self._advance(per)
        return bool(predicate(self))

    def _advance(self, n: int) -> None:
        net = self.net
        kernel.advance(self.x, n, self.dt, net.NI, self.NF, net.LI, net.LF, self.LV, self.fl, self.W,
                       self.nout, self.lout, self.FN, self.FV, self.NS, net.xscale, self.st, self.sc,
                       net.iperm, net.kl, net.ku, self.jvals, self.f0, self.ab, self.piv, self.bvec, self.xnew,
                       self.ctl, self.stats,
                       self._empty_h, self._empty_h, self._empty_l, 0)
        self.t = (self.steps + n) * self.dt
        self.steps += n
        if n:
            for st in self._sensors.values():
                st.update(self._raw(st), n * self.dt, self.t)

    @property
    def substeps(self) -> int:
        return int(self.stats[0])

    def solver_stats(self) -> dict:
        """Counters since reset: sub-steps, Newton iterations, Jacobians, failed and forced steps."""
        s = self.stats
        return {"steps": self.steps, "substeps": int(s[0]), "newton_iterations": int(s[1]),
                "jacobians": int(s[2]), "failed_substeps": int(s[3]), "forced_steps": int(s[4]),
                "unconverged": int(s[5])}

    # ------------------------------------------------------------------ commands
    def valves(self) -> list[str]:
        return list(self.net.valves)

    def command(self, valve: str, position) -> None:
        """Command a valve: True/"open", False/"closed", or a position 0-1."""
        k = self._valve(valve)
        if isinstance(position, str):
            position = {"open": 1.0, "opened": 1.0, "close": 0.0, "closed": 0.0}[position.lower()]
        self.LV[k, L.V_CMD] = min(max(float(position), 0.0), 1.0)

    def open(self, *valves: str) -> None:
        for v in valves:
            self.command(v, 1.0)

    def close(self, *valves: str) -> None:
        for v in valves:
            self.command(v, 0.0)

    def valve_command(self, valve: str) -> float:
        return float(self.LV[self._valve(valve), L.V_CMD])

    def valve_position(self, valve: str) -> float:
        return float(self.LV[self._valve(valve), L.V_POS])

    def set_boundary(self, comp: str, pressure=None, temperature=None) -> None:
        """Change a pressure boundary during a run (a supply regulator setting, say)."""
        nodes = self.net.comp_nodes.get(comp, [])
        i = next((i for i in nodes if self.net.NI[i, L.N_KIND] == L.NODE_BOUNDARY), None)
        if i is None:
            raise KeyError(f"{comp} is not a pressure boundary")
        p = units.parse(pressure, "pressure") if pressure is not None else self.NF[i, L.NF_BP]
        t = units.parse(temperature, "temperature") if temperature is not None else self.NF[i, L.NF_BT]
        rho, u = self.net.fluids.rho_u_from_pt(int(self.net.NI[i, L.N_FLUID]), p, t)
        self.NF[i, L.NF_BP], self.NF[i, L.NF_BT], self.NF[i, L.NF_BRHO], self.NF[i, L.NF_BU] = p, t, rho, u

    def _valve(self, valve: str) -> int:
        try:
            return self.net.valves[valve]
        except KeyError:
            raise KeyError(f"no commandable valve {valve!r}; valves: {', '.join(self.net.valves)}") from None

    # ------------------------------------------------------------------ reading
    def node(self, ref: str) -> int:
        """Node index for "COMP.port", "COMP" (its first volume) or "COMP.s2" (pipe segment)."""
        net = self.net
        if ref in net.port_nodes:
            return net.port_nodes[ref]
        if ref in net.node_names:
            return net.node_names.index(ref)
        if ref in net.comp_nodes:
            return net.comp_nodes[ref][0]
        raise KeyError(f"no node {ref!r}")

    def link(self, ref: str) -> int:
        net = self.net
        if ref in net.link_names:
            return net.link_names.index(ref)
        if ref in net.comp_links:
            ks = net.comp_links[ref]
            return ks[len(ks) // 2]
        raise KeyError(f"no flow element {ref!r}")

    def pressure(self, ref: str) -> float:
        return float(self.nout[self.node(ref), L.O_P])

    def temperature(self, ref: str) -> float:
        return float(self.nout[self.node(ref), L.O_T])

    def quality(self, ref: str) -> float:
        return float(self.nout[self.node(ref), L.O_X])

    def density(self, ref: str) -> float:
        return float(self.nout[self.node(ref), L.O_RHO])

    def mass(self, ref: str) -> float:
        """Fluid mass in a component's volumes (both regions for a pressurized tank)."""
        if ref in self.net.comp_nodes:
            return float(sum(self.nout[i, L.O_M] for i in self.net.comp_nodes[ref]))
        return float(self.nout[self.node(ref), L.O_M])

    def mass_flow(self, ref: str) -> float:
        return float(self.lout[self.link(ref), L.LO_MDOT])

    def read(self, sensor: str) -> float:
        """Sensor value in SI units, with its lag, sample rate and noise applied."""
        st = self._sensors.get(sensor)
        if st is None:
            raise KeyError(f"no sensor {sensor!r}; sensors: {', '.join(self._sensors)}")
        v = st.held
        if st.spec.noise:
            v += self.rng.normal(0.0, st.spec.noise)
        return float(v)

    def sensors(self) -> dict[str, float]:
        return {k: self.read(k) for k in self._sensors}

    def snapshot(self) -> dict:
        """Every node and link value, for plots and logs."""
        net = self.net
        return {
            "t": self.t,
            "nodes": {name: {"p": self.nout[i, L.O_P], "t": self.nout[i, L.O_T], "x": self.nout[i, L.O_X],
                             "rho": self.nout[i, L.O_RHO], "m": self.nout[i, L.O_M]}
                      for i, name in enumerate(net.node_names)},
            "links": {name: {"mdot": self.lout[k, L.LO_MDOT], "open": self.lout[k, L.LO_OPEN]}
                      for k, name in enumerate(net.link_names)},
            "sensors": self.sensors(),
            "valves": {v: self.valve_position(v) for v in net.valves},
        }

    # ------------------------------------------------------------------ sensors
    def _sensor_source(self, s) -> tuple[str, int]:
        if s.type not in SENSOR_TYPES:
            raise ValueError(f"sensor {s.id}: unknown type {s.type!r}; types: {', '.join(SENSOR_TYPES)}")
        if s.type == "mass_flow":
            return "link", self.link(s.at)
        if s.type == "valve_position":
            return "valve", self._valve(s.at)
        if s.type == "mass":
            return "mass", 0
        return "node", self.node(s.at)

    def _raw(self, st) -> float:
        kind, idx = st.source
        t = st.spec.type
        if kind == "link":
            return float(self.lout[idx, L.LO_MDOT])
        if kind == "valve":
            return float(self.LV[idx, L.V_POS])
        if kind == "mass":
            return self.mass(st.spec.at)
        col = {"pressure": L.O_P, "temperature": L.O_T, "quality": L.O_X, "density": L.O_RHO,
               "wall_temperature": L.O_TW}[t]
        return float(self.nout[idx, col])


class _SensorState:
    def __init__(self, spec, source):
        self.spec = spec
        self.source = source
        self.value = 0.0
        self.held = 0.0
        self.next_sample = 0.0

    def update(self, raw: float, elapsed: float, t: float) -> None:
        if self.spec.lag > 0:
            self.value += (1.0 - math.exp(-elapsed / self.spec.lag)) * (raw - self.value)
        else:
            self.value = raw
        if self.spec.rate <= 0:
            self.held = self.value
        elif t + 1e-12 >= self.next_sample:
            self.held = self.value
            self.next_sample = t + 1.0 / self.spec.rate
