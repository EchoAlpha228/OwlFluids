"""Headless simulation: build a network, command valves, advance time, read sensors.

This is the scripting API that tests use directly. ``Simulation`` runs as fast as
the CPU allows; ``owlfluids.sim.realtime`` paces the same object to the wall clock.

    sim = Simulation.from_file("examples/gn2_blowdown.yaml")
    sim.open("V-1")
    sim.run_until(lambda s: s.read("PT-1") < 500 * PSI, timeout=60)
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .. import units
from ..core import kernel
from ..core import layout as L
from ..core.network import Network, build_network
from ..fluids import props
from ..io.system import System
from . import sensors as sensor_model

MIN_BUDGET = 8.0  # smallest work per step: one Jacobian and two Newton iterations

SENSOR_TYPES = ("pressure", "temperature", "mass_flow", "valve_position", "quality", "density", "mass",
                "wall_temperature", "level", "purge_fraction")


class Simulation:
    def __init__(self, system: System, seed: int | None = None, step_budget: float | None = None):
        """``step_budget``: work per step in right-hand-side evaluations; by default it is
        timed on this machine (see ``_step_budget``). Give a number for runs that must
        come out the same on every machine. ``seed``: sensor noise seed (default: the
        system's ``noise_seed`` setting); every reset restarts the same noise."""
        self.system = system
        self._fixed_budget = step_budget
        self.net: Network = build_network(system)
        self.dt = self.net.dt
        self.seed = int(system.setting("noise_seed") if seed is None else seed)
        self.realistic = str(system.setting("sensor_noise")).strip().lower() not in ("off", "false", "none", "0")
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
        self._init_scratch()
        self.nout = np.zeros((nn, L.N_NO))
        self.lout = np.zeros((nl, L.N_LO))
        self.FN = np.full((nl, 4), -1, dtype=np.int64)
        self.FV = np.zeros((nl, kernel.N_FV))
        self.NS = np.zeros((nn, 2))
        self.jvals = np.zeros(len(net.col_rows))
        self.f0 = np.zeros(n)
        self.st = (net.col_ptr, net.col_rows, net.color_ptr, net.color_cols, net.sowner, net.nl_ptr, net.nl_idx)
        self.sc = (np.zeros(n), np.zeros(n), np.zeros(nn, dtype=np.int64), np.zeros(nl, dtype=np.int64),
                   np.zeros(nn, dtype=np.int64), np.zeros(nl, dtype=np.int64), np.zeros((nn, kernel.N_W)),
                   np.zeros(4, dtype=np.int64), np.zeros(kernel.N_FV), np.zeros(2), np.zeros(1, dtype=np.int64))
        self.ab = np.zeros((n, 2 * net.kl + net.ku + 1))
        self.piv = np.zeros(n, dtype=np.int64)
        self.bvec = np.zeros(n)
        self.xnew = np.zeros(n)
        self._empty_h = np.zeros((0, max(nn, 1)))
        self._empty_l = np.zeros((0, max(nl, 1)))
        self.fl = self.net.fluids.arrays()
        self.rhs_seconds = 0.0
        self.step_budget = float(self._fixed_budget) if self._fixed_budget else self._step_budget()
        # nodes whose fluid uses the property tables, which stop at tables.T_MAX
        from ..fluids import tables
        from ..fluids.registry import MODEL_TABLE

        table_node = (net.fluids.model[net.NI[:, L.N_FLUID]] == MODEL_TABLE) & (net.NI[:, L.N_KIND] != L.NODE_BOUNDARY)
        self._t_limit = np.where(table_node, tables.T_MAX - 1.0, np.inf)

    def _init_scratch(self) -> None:
        # a mixed node's volume split starts from the network's initial one
        mixed = self.net.NI[:, L.N_GX] >= 0
        for j, cols in enumerate(((kernel.W_U, kernel.W_U0), (kernel.W_LT, kernel.W_LT0), (kernel.W_T, kernel.W_T0),
                                  (kernel.W_REG, kernel.W_REG0))):
            for c in cols:
                self.W[mixed, c] = self.net.mix_init[mixed, j]

    def _step_budget(self, share: float = 0.35) -> float:
        """Work one step may spend, in right-hand-side evaluations.

        Sized so a step uses at most ``share`` of dt on this machine: one
        evaluation is timed when the simulation is built, and a Newton iteration
        costs about 1.5 evaluations once the linear solve is counted. A slower CPU
        gets a smaller budget (more forced steps) rather than falling behind real time.
        """
        net = self.net
        args = (net.x0.copy(), self.f0, net.NI, net.NF.copy(), net.LI, net.LF, net.LV.copy(), self.fl,
                self.W, self.nout, self.lout, self.FN, self.FV, self.NS)
        kernel.rhs(*args)  # compile
        runs = []
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(20):
                kernel.rhs(*args)
            runs.append((time.perf_counter() - t0) / 20)
        # the median, not the best: leaves margin when other threads (GUI, Synnax) share the CPU
        self.rhs_seconds = cost = sorted(runs)[len(runs) // 2]
        budget = share * self.dt / (1.5 * max(cost, 1e-9))
        if budget < MIN_BUDGET:
            # below this a step cannot build a Jacobian and still iterate: the floor holds,
            # so hard transients can run behind the wall clock on this machine
            self._slow_note = (f"this computer is too slow to guarantee real time for this system: a step costs "
                               f"{1.5 * cost * MIN_BUDGET / self.dt:.0%} of its time at the smallest work budget; "
                               "hard transients may run behind the clock")
        return float(min(max(budget, MIN_BUDGET), 1000.0))

    def reset(self) -> None:
        """Back to the initial conditions of the system."""
        net = self.net
        self.x = net.x0.copy()
        self._init_scratch()
        self.LV = net.LV.copy()
        self.NF = net.NF.copy()
        self.t = 0.0
        self.steps = 0
        self.ctl = np.zeros(12)
        self.ctl[1] = -1.0
        self.ctl[4] = 1.0
        self.ctl[9] = self.step_budget
        self.stats = np.zeros(6)
        self.notices: list[str] = []  # physics the model cannot represent, newest last
        if getattr(self, "_slow_note", None):
            self.notices.append(self._slow_note)
        self._limited = np.zeros(self.net.NI.shape[0], dtype=bool)
        self.rng = np.random.default_rng(self.seed)
        self._sensors = {}
        for s in self.system.sensors:
            st = _SensorState(s, self._sensor_source(s))
            st.lag = s.lag if s.lag is not None else (sensor_model.default_lag(s.type) if self.realistic else 0.0)
            st.modelled = s.noise is None and self.realistic
            st.links = self._sensor_links(st)
            self._sensors[s.id] = st
        self._advance(0)
        for st in self._sensors.values():
            st.value = st.held = self._raw(st)
            self._sample_noise(st)

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
        hot = self.nout[:, L.O_T] >= self._t_limit
        if hot.any() and (hot & ~self._limited).any():
            for i in np.nonzero(hot & ~self._limited)[0]:
                self.notices.append(f"{self.net.node_names[i]} reached the {self._t_limit[i] + 1:.0f} K limit of the "
                                    "property tables at t = {:.3f} s; hotter gas reads too cold and at too low a "
                                    "pressure".format(self.t))
            self._limited |= hot
        if n:
            mdot = self.lout[:, L.LO_MDOT]
            for st in self._sensors.values():
                flow = sensor_model.flow_factor(float(np.abs(mdot[st.links]).sum()) / 2) if len(st.links) else 0.0
                if st.update(self._raw(st), n * self.dt, self.t, flow):
                    self._sample_noise(st)

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

    def set_position(self, valve: str, position: float) -> None:
        """Put a valve at ``position`` at once, skipping its stroke (a measured position from hardware)."""
        k = self._valve(valve)
        p = min(max(float(position), 0.0), 1.0)
        self.LV[k, L.V_CMD] = p
        self.LV[k, L.V_POS] = p
        a = int(self.LV[k, L.V_AIR])
        if a >= 0:
            # the measured valve has its actuator charged or vented to match, so it stays there
            no = self.LV[a, L.V_NO] > 0.5
            f = 1.0 - p if no else p
            p_lo = units.ATM + self.net.LF[a, L.LF_P1]
            p_hi = units.ATM + self.net.LF[a, L.LF_P2]
            self.LV[a, L.V_PACT] = units.ATM if f <= 0.0 else p_lo + f * (p_hi - p_lo)
            self.LV[k, L.V_TGT] = p

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

    def actuator_pressure(self, valve: str) -> float | None:
        """Absolute pressure (Pa) in a pneumatic valve's actuator; None for a valve without an air supply."""
        k = self.net.actuators.get(valve)
        if k is None:
            self._valve(valve)
            return None
        return float(self.LV[k, L.V_PACT])

    def air_draw(self, valve: str) -> float:
        """Mass flow (kg/s) a pneumatic valve's actuator is drawing from its supply now."""
        k = self.net.actuators.get(valve)
        return 0.0 if k is None else float(self.LV[k, L.V_Q])

    def pumps(self) -> list[str]:
        return list(self.net.pumps)

    def pump_running(self, pump: str) -> bool:
        """Whether a pump's pressure switch has it on."""
        if pump not in self.net.pumps:
            raise KeyError(f"no pump {pump!r}; pumps: {', '.join(self.net.pumps)}")
        return bool(self.LV[self.net.pumps[pump], L.V_CMD] > 0.5)

    def set_boundary(self, comp: str, pressure=None, temperature=None) -> None:
        """Change a pressure boundary during a run (a supply regulator setting, say)."""
        self.apply_boundary(self.boundary_state(comp, pressure, temperature))

    def boundary_state(self, comp: str, pressure=None, temperature=None) -> tuple:
        """Validate a boundary change and work out its state, without applying it."""
        nodes = self.net.comp_nodes.get(comp, [])
        i = next((i for i in nodes if self.net.NI[i, L.N_KIND] == L.NODE_BOUNDARY), None)
        if i is None:
            raise KeyError(f"{comp} is not a pressure boundary")
        p = units.parse(pressure, "pressure") if pressure is not None else float(self.NF[i, L.NF_BP])
        t = units.parse(temperature, "temperature") if temperature is not None else float(self.NF[i, L.NF_BT])
        rho, u = self.net.fluids.rho_u_from_pt(int(self.net.NI[i, L.N_FLUID]), p, t)
        return i, p, t, rho, u

    def apply_boundary(self, state: tuple) -> None:
        i, p, t, rho, u = state
        self.NF[i, L.NF_BP], self.NF[i, L.NF_BT], self.NF[i, L.NF_BRHO], self.NF[i, L.NF_BU] = p, t, rho, u

    def _valve(self, valve: str) -> int:
        try:
            return self.net.valves[valve]
        except KeyError:
            raise KeyError(f"no commandable valve {valve!r}; valves: {', '.join(self.net.valves)}") from None

    # ------------------------------------------------------------------ reading
    def node(self, ref: str) -> int:
        """Node index for "COMP.port", or "COMP" (its first volume; for a line, the point it joins)."""
        net = self.net
        if ref in net.port_nodes:
            return net.port_nodes[ref]
        if ref in net.node_names:
            return net.node_names.index(ref)
        if net.comp_nodes.get(ref):
            return net.comp_nodes[ref][0]
        if f"{ref}.in" in net.port_nodes:
            return net.port_nodes[f"{ref}.in"]
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
        return float(st.reading)

    def _sample_noise(self, st) -> None:
        """Noise is drawn once per sample, so repeated reads agree and runs repeat exactly."""
        v = st.held
        sd = sensor_model.sigma(st.spec.type, v, st.rate, st.flow) if st.modelled else (st.spec.noise or 0.0)
        st.reading = v + self.rng.normal(0.0, sd) if sd > 0 and math.isfinite(v) else v

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
            # on a pump it reads how far it has spun up (0 stopped, 1 running)
            return "valve", self.net.pumps[s.at] if s.at in self.net.pumps else self._valve(s.at)
        if s.type == "mass":
            return "mass", 0
        if s.type == "level":
            return "level", self._level_source(s)
        return "node", self.node(s.at)

    def _level_source(self, s) -> int:
        comp = s.at.split(".")[0]
        nodes = self.net.comp_nodes.get(comp, [])
        ctype = next((c.type for c in self.system.components if c.id == comp), None)
        if ctype not in ("tank", "pressurized_tank") or not nodes:
            raise ValueError(f"sensor {s.id}: a level sensor goes on a tank, not {s.at!r}")
        return nodes[0]

    def _sensor_links(self, st) -> np.ndarray:
        """Links next to a pressure sensor's node, whose flow makes the reading noisier."""
        kind, idx = st.source
        if kind != "node" or st.spec.type != "pressure":
            return np.zeros(0, dtype=np.int64)
        net = self.net
        return net.nl_idx[net.nl_ptr[idx]:net.nl_ptr[idx + 1]]

    def _level(self, comp: str) -> float:
        """Liquid volume fraction 0-1 of a tank."""
        nodes = self.net.comp_nodes[comp]
        o = self.nout
        if len(nodes) > 1:  # pressurized tank: ullage, then liquid
            vol = self.net.NF[nodes[0], L.NF_VOL]
            liq = nodes[1]
            return min(max(o[liq, L.O_M] / max(o[liq, L.O_RHO], 1e-9) / vol, 0.0), 1.0)
        i = nodes[0]
        x = o[i, L.O_X]
        fl = self.fl
        # the saturation table, not CoolProp: this runs on the runner thread every chunk
        sat = props.saturation(int(self.net.NI[i, L.N_FLUID]), float(o[i, L.O_T]), fl[0], fl[1], fl[7], fl[8])
        if not sat[8] or not math.isfinite(x) or x >= 1.0:  # supercritical or all vapor: no liquid level
            return 0.0
        if x <= 0.0:
            return 1.0
        return min(max((1.0 - x) * o[i, L.O_RHO] / sat[1], 0.0), 1.0)

    def _raw(self, st) -> float:
        kind, idx = st.source
        t = st.spec.type
        if kind == "link":
            return float(self.lout[idx, L.LO_MDOT])
        if kind == "valve":
            return float(self.LV[idx, L.V_POS])
        if kind == "mass":
            return self.mass(st.spec.at)
        if kind == "level":
            return self._level(st.spec.at.split(".")[0])
        col = {"pressure": L.O_P, "temperature": L.O_T, "quality": L.O_X, "density": L.O_RHO,
               "wall_temperature": L.O_TW, "purge_fraction": L.O_GAS}[t]
        return float(self.nout[idx, col])


class _SensorState:
    def __init__(self, spec, source):
        self.spec = spec
        self.source = source
        self.value = 0.0
        self.held = 0.0
        self.next_sample = 0.0
        self.lag = 0.0
        self.modelled = False
        self.links = None
        self.rate = 0.0  # d(reading)/dt, for transient noise
        self.flow = 0.0  # 0-1, flow past the sensor
        self.reading = 0.0  # held value plus noise: what read() returns

    def update(self, raw: float, elapsed: float, t: float, flow: float = 0.0) -> bool:
        """Advance the lag; True when a new sample was taken."""
        before = self.value
        if self.lag > 0:
            self.value += (1.0 - math.exp(-elapsed / self.lag)) * (raw - self.value)
        else:
            self.value = raw
        if elapsed > 0 and math.isfinite(self.value) and math.isfinite(before):
            self.rate = (self.value - before) / elapsed
        self.flow = flow
        if self.spec.rate <= 0:
            self.held = self.value
            return True
        if t + 1e-12 >= self.next_sample:
            self.held = self.value
            self.next_sample = t + 1.0 / self.spec.rate
            return True
        return False
