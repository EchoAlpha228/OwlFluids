# OwlFluids

A real-time, one-dimensional fluid network simulator for test-stand automation.
Build a system from tanks, tubing, fittings, valves, regulators and orifices, set
starting conditions, and run it against the wall clock while you (or a sequence,
or Synnax) open and close valves.

- **Real time first.** An 81-component LOX/ethanol stand takes about 0.06–0.1 ms
  per 1 ms step in steady operation and about 0.2 ms during valve transients on a
  desktop CPU, and every step has a work budget, timed on your machine at start-up,
  so hard transients (LOX hitting a warm line) can't stall the clock. A slower CPU
  gets more forced steps instead of falling behind.
- **Real fluids.** Water, ethanol, oxygen (GOX/LOX), methane, nitrogen (GN2/LN2) and
  nitrous oxide, from ambient to ~3,000 psi and cryogenic to room temperature.
  Gases and cryogens use tables built once from CoolProp or REFPROP, including
  two-phase states (saturated tanks, flashing, chilldown).
- **Desktop app.** Drag parts onto a schematic, wire their ports, run, and click
  valves; live values on the drawing and strip charts below.
- **Scriptable and headless.** The same simulation runs from Python, the command
  line, or a sequence script; runs can be logged to CSV or HDF5.
- **Hardware in the loop.** A Synnax driver writes simulated sensors to Synnax
  channels and takes valve commands (or real valve states) back.

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
| `owlfluids run file.yaml --synnax examples/synnax.yaml` | Hardware-in-the-loop with Synnax |
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

Lines start in equilibrium with the tank or volume they are open to: a feed line
on a LOX tank starts full of LOX at tank pressure, a regulated line starts at its
setpoint, and anything behind a closed valve starts at the system's initial
pressure and temperature (`prime_lines: false` turns this off).

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

## Synnax hardware-in-the-loop

`examples/synnax.yaml` maps simulated sensors to Synnax channels (written in each
sensor's display units, e.g. psi) and valves to the channels read back: the
command channels your Arc sequences write, or the state channels a real controller
reports. Missing channels are created unless `create_channels: false`. Only
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
flow, a self-pressurizing LN2 dewar, nitrous blowdown, line chilldown, and the
real-time gate on the 81-component stand.

## Limits worth knowing

- One fluid per connected line (separate gases from propellants with a tank or
  a pressurized tank); no mixing of species.
- Two-phase flow is homogeneous equilibrium. Violent events (LOX into a warm line,
  geysering) are stable and conserve mass and energy but are not resolved in
  detail; the status bar counts "forced steps" when a step used its whole budget.
  For runs that must match across machines, pass `Simulation(system, step_budget=200)`.
- Property tables blend pressure within about one grid cell of saturated liquid,
  so a saturated tank filled above ~98 % can read a few percent high (up to ~13 % at
  99 %). States within a kelvin or so of the critical point are also approximate.
- Pipe heat transfer uses a fixed film coefficient (set per line).
