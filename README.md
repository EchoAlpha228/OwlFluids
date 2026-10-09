# OwlFluids

A real-time, one-dimensional fluid network simulator for test-stand automation.
Build a system from tanks, tubing, fittings, valves, regulators and orifices, set
starting conditions, and run it against the wall clock while you (or a sequence,
or Synnax) open and close valves.

- **Real time first.** An 81-component LOX/ethanol stand takes about 0.08–0.1 ms
  per 1 ms step in steady operation and about 0.2 ms during valve transients and
  purges on a desktop CPU, and every step has a work budget, timed on your machine at start-up,
  so hard transients (LOX hitting a warm line) can't stall the clock. A slower CPU
  gets more forced steps instead of falling behind.
- **Real fluids.** Water, ethanol, oxygen (GOX/LOX), methane, nitrogen (GN2/LN2) and
  nitrous oxide, from ambient to ~3,000 psi and cryogenic to room temperature.
  Gases and cryogens use tables built once from CoolProp or REFPROP, including
  two-phase states (saturated tanks, flashing, chilldown). Helium, air, argon and
  CO2 are available as gases.
- **Purges.** Any of helium, nitrogen, argon, air or CO2 can purge a propellant or
  water line: the gas pushes the liquid out, LOX boils off into it, and a
  `purge_fraction` sensor shows how clear the line is.
- **Desktop app.** Drag parts onto a schematic, wire their ports, run, and click
  valves; live values on the drawing and strip charts below.
- **Scriptable and headless.** The same simulation runs from Python, the command
  line, or a sequence script; runs can be logged to CSV or HDF5.
- **HOOTL and HITL with Synnax.** A Synnax driver writes simulated sensors to
  Synnax and either takes valve commands and fakes the valve states (HOOTL) or
  takes the real hardware's valve states (HITL).

## Install (Windows)

Python 3.12 is recommended (the Synnax client needs 3.12+).

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e ".[app,hdf5,synnax]"
owlfluids tables                 # build property tables once (~5 s per fluid)
owlfluids app examples\lox-feed.yaml
```

To build tables from REFPROP instead of CoolProp's HEOS, point CoolProp at your
REFPROP folder and choose the backend:

```powershell
$env:RPPREFIX = "C:\Program Files (x86)\REFPROP"
owlfluids tables --backend REFPROP
```

then set **Edit > Settings > property backend** to REFPROP (or `backend: REFPROP`
under `settings:` in the system file).

## Using it

| Command | What it does |
| --- | --- |
| `owlfluids app [file.yaml]` | Desktop app (also `owlfluids` with no arguments) |
| `owlfluids console file.yaml` | Real-time run with typed commands: `open V-M`, `close V-M`, `set TV-1 0.4`, `status` |
| `owlfluids run file.yaml --script seq.py --log run.csv` | Headless run of a sequence; `--rate 0` runs as fast as possible |
| `owlfluids run file.yaml --synnax examples/synnax.yaml` | Run against Synnax (HOOTL, or `--synnax-mode hitl`) |
| `owlfluids check file.yaml` | Build the network and print a summary (catches wiring mistakes) |
| `owlfluids bench` | Time the 81-component example stand |
| `owlfluids example NAME -o file.yaml` | Write one of the examples |

In the app: drag parts from **Parts**, drag from one port (small circle) to another
to connect, **R** rotates, **Delete** removes. Select a part to edit its
parameters; quantities take units (`3000 psi`, `-196 C`, `0.5 in`, `49 L`).
Press **Run**; valves appear in the **Valves** tab and can also be clicked on the
schematic. **Record** logs to CSV/HDF5, and **File > Overlay a recorded log** draws
a log (for example real test data with matching column names) over the plots.

### System files

```yaml
name: GN2 bottle blowdown
components:
- {id: K1, type: tank, fluid: nitrogen, volume: 49 L, pressure: 2400 psi, temperature: 20 C}
- {id: V1, type: valve, cv: 0.3, open_time: 50 ms}
- {id: L1, type: pipe, length: 2 m, diameter: 0.25 in}
- {id: ATM, type: boundary, fluid: nitrogen}
connections:
- [K1.port, V1.in]
- [V1.out, L1.in]
- [L1.out, ATM.port]
sensors:
- {id: PT-K1, type: pressure, at: K1.port, units: psi}
```

Component types: `tank` (gas, or saturated liquid/vapor with `fill`),
`pressurized_tank` (propellant with a pressurant ullage), `boundary`, `volume`,
`pipe` (friction, inertia, optional wall for chilldown), `tee`, `fitting`,
`orifice`, `valve`, `throttle_valve`, `check_valve`, `relief_valve`, `regulator`.
`owlfluids app` shows every parameter with its default and meaning.

Sensor types: `pressure`, `temperature`, `wall_temperature`, `mass_flow`,
`valve_position`, `mass`, `level` (a tank's liquid fill, shown in %),
`purge_fraction` (share of a node's volume that is purge gas, in %), `quality`
and `density`. Display units default to psi, °F, kg/s, kg and %; set `units:`
to change one (temperatures defaulted to K before October 2026, so a sensor
without `units:` now logs and sends °F). Sensors read with realistic noise and lag for their type: a
pressure transducer gets noisier while fluid flows past it and while the pressure
swings, thermocouples lag about 0.3 s, a flow meter's noise scales with flow (so a
still line reads 0), and a level reading jitters as the level moves. Give a sensor
`noise: off` or its own `noise:` (standard deviation, SI) and `lag:` (s), or set
`sensor_noise: off` under `settings:` for exact values. Noise repeats run to run
for a given `noise_seed` setting. It changes only what sensors read, never the
simulation.

Lines start in equilibrium with the tank or volume they are open to: a feed line
on a LOX tank starts full of LOX at tank pressure, a regulated line starts at its
setpoint, and anything behind a closed valve starts at the system's initial
pressure and temperature (`prime_lines: false` turns this off).

### Purges

Connect a purge bottle (a `tank` of helium, nitrogen, argon, air or CO2), usually
through a regulator, a valve and a check valve, to any line; `examples/lox-purge.yaml`
purges a LOX feed with helium. Where the purge gas and the line's own fluid can
meet, each node carries both: with liquid present, the liquid fills its share of
the node and its vapor and the purge gas share the rest at one temperature
(pressure = vapor pressure + the gas's partial pressure), so LOX boils off into a
nitrogen or helium sweep and cools it; with no liquid, the two gases fill the node
together. Flow out of such a node is homogeneous (each fluid leaves in proportion
to its mass). The part of the line between a closed purge valve and its check valve
starts full of purge gas.

Only a purge gas may meet another fluid. Two propellants meeting (GOX in a methane
line) is still a build error, since that would need combustion. When two purge
gases meet (helium into a nitrogen line), the one listed first in helium,
nitrogen, argon, air, CO2 is the purge gas. A tank, pressurized tank or `boundary` holds only its
own fluid: purge gas flowing into it is counted as that fluid at the same
temperature, and the builder lists each place this can happen. Give a vent
boundary the line's own fluid.

### Python

```python
from owlfluids.sim.simulation import Simulation

sim = Simulation.from_file("examples/lox-feed.yaml")
sim.open("V-M")
sim.run(2.0)                       # as fast as the CPU allows
print(sim.read("PT-INJ") / 6894.76, "psi", sim.mass_flow("INJ"), "kg/s")
```

For real time, wrap it in `owlfluids.sim.realtime.Runner` (`start()`, `open()`,
`wait()`, `wait_until()`, `frame`, `history`); sequence scripts get a runner (see
`examples/lox_flow_sequence.py`).

## Synnax: HOOTL and HITL

`examples/synnax.yaml` maps simulated sensors to Synnax channels (written in each
sensor's display units, e.g. psi) and each valve to a command channel and a state
channel. Its `mode` picks the test:

| mode | valves driven by | valve state channels |
|---|---|---|
| `hootl` (default) | the command channels your Arc sequences and operators write | faked by the sim |
| `hitl` | the state channels the real hardware reports | left to the hardware |

`valve_state` says what a state channel means. `command` (the default) is a Synnax
digital output task's state, which echoes the command: HOOTL fakes it from the
command, and in HITL the sim strokes the valve itself. `position` is limit-switch
or position feedback: HOOTL fakes limit switches (1 once fully open, 0 once fully
closed, held mid-stroke), and in HITL the sim valve jumps to the reported position.
Use `position` in HITL only with real position feedback: limit switches alone would
hold the sim valve shut for the whole stroke, so stands with only limit switches
should keep `command`.

`--synnax-mode hitl` on `run` or `console` overrides the file. Point the channel
names at the stand's real channels so sequences and schematics work unchanged:
existing channels keep their own index channel, and missing ones are created on
`index` unless `create_channels: false`. In HITL, stop the hardware's sensor read
task first, since the sim writes those sensor channels; the driver refuses to
start if a sensor it writes shares an index with a valve state it reads, and it
errors rather than losing writes when another writer holds a channel. Only
`owlfluids/drivers/synnax.py`'s `SynnaxLink` touches the Synnax client, so a new
Synnax release means updating that one class.

## How it works

Each component becomes lumped **nodes** (volumes holding mass and internal energy,
plus a wall temperature for chilldown lines) and **links** (pipes carry their mass
flow as a state, so water hammer and line inertia are captured; orifices, valves,
fittings and regulators are quasi-steady). Every 1 ms step is backward Euler,
solved by chord Newton iterations on a banded LU of a finite-difference Jacobian
that is coloured, reordered (reverse Cuthill-McKee) and updated incrementally.
The kernel is compiled with Numba and releases the GIL. Fluid properties come from
(density, internal energy) tables with a boundary-fitted compressed-liquid table,
so subcooled LOX and two-phase states match CoolProp to about 1% in pressure
(0.5% for 99% of states).

Validation tests (`pytest`) cover isentropic gas blowdown, closed-system
equalization (mass and energy conserved), Torricelli drain, Colebrook pipe flow,
Joukowsky water hammer, table accuracy against CoolProp, LOX feed priming and
flow, a self-pressurizing LN2 dewar, nitrous blowdown, line chilldown, purges with
every purge gas (mass conserved, LOX, water, ethanol and methane pushed out), and the
real-time gate on the 81-component stand, purges included.

## Limits worth knowing

- One fluid per connected line, plus one purge gas (separate propellants with a
  tank or a pressurized tank). A purge gas is an ideal gas with constant heat
  capacity, does not dissolve in the liquid, and the liquid under it is slightly
  compressible with a constant bulk modulus.
- Two-phase flow is homogeneous equilibrium. Violent events (LOX into a warm line,
  geysering) are stable and conserve mass and energy but are not resolved in
  detail; the status bar counts "forced steps" when a step used its whole budget.
  For runs that must match across machines, pass `Simulation(system, step_budget=200)`.
- Property tables blend pressure within about one grid cell of saturated liquid,
  so a saturated tank filled above ~98 % can read a few percent high (up to ~13 % at
  99 %). States within a kelvin or so of the critical point are also approximate.
- Pipe heat transfer uses a fixed film coefficient (set per line).
