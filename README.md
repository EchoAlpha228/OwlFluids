# OwlFluids

A real-time, one-dimensional fluid network simulator for test-stand automation.
Build a system from tanks, tubing, fittings, valves, regulators and orifices, set
starting conditions, and run it against the wall clock while you (or a sequence,
or Synnax) open and close valves.

- **Real time first.** A 60-component LOX/ethanol stand takes about 0.03–0.05 ms
  per 1 ms step in steady operation and about 0.1 ms during valve transients and
  purges on a desktop CPU, and every step has a work budget, timed on your machine at start-up,
  so hard transients (LOX hitting a warm line) can't stall the clock. A slower CPU
  gets more forced steps instead of falling behind; if a machine is too slow even
  for the smallest budget, the simulation says so when it starts.
- **Real fluids.** Water, ethanol, oxygen (GOX/LOX), methane, nitrogen (GN2/LN2) and
  nitrous oxide, from ambient to ~3,000 psi and cryogenic to room temperature.
  Gases and cryogens use tables built once from CoolProp or REFPROP, including
  two-phase states (saturated tanks, flashing, chilldown). Helium and air are
  available as gases.
- **Purges and shop air.** Helium, nitrogen or air can purge a propellant or
  water line: the gas pushes the liquid out, and a `purge_fraction` sensor shows
  how clear the line is. After a purge the line fills again normally. Air pumps
  (compressors) switch on at a low pressure and off at a high one.
- **Desktop app.** Drag parts onto a schematic, wire their ports, run, and click
  valves; live values on the drawing and strip charts below.
- **Scriptable and headless.** The same simulation runs from Python, the command
  line, or a sequence script; runs can be logged to CSV or HDF5.
- **HOOTL and HITL with Synnax.** A Synnax driver writes simulated sensors to
  Synnax and either takes valve commands and fakes the valve states (HOOTL) or
  takes the real hardware's valve states (HITL).

## Install (Windows)

### Installer

Download `OwlFluids-Setup-<version>.exe` from the repository's
[Releases](https://github.com/EchoAlpha228/OwlFluids/releases) and run it. It
installs for your user only (no admin prompt) into `%LOCALAPPDATA%\Programs\OwlFluids`
with its own Python, the Synnax client and prebuilt property tables, adds Start
menu entries, and puts `owlfluids` on your PATH for new terminals. Windows may
warn about an unknown publisher because the installer isn't code-signed: choose
**More info > Run anyway**.

**Updates.** The app checks for a newer release when it starts and offers to
install it (**Help > Check for updates**; `owlfluids update` from a terminal).
No sign-in is needed while the repository is public. A GitHub token is optional
(it lifts GitHub's limit on anonymous requests, and is required if the repository
goes private): paste one into **Help > Update settings**, or use
`owlfluids update --token <token>` or `OWLFLUIDS_GITHUB_TOKEN`.

**Synnax.** Each installer bundles the newest Synnax client at the time it was
built; it is private to OwlFluids and doesn't touch any
other Python install. The client's major.minor version should match your Synnax
Core: the driver warns on connect if it doesn't (`check_version: strict` under
`synnax:` refuses instead, `off` skips the check). For an older Core, run **Actions > Release > Run
workflow** with its `synnax_version` and install the build it attaches.

The installed program runs sequence scripts (`owlfluids run --script`) with its
bundled packages only. To `import owlfluids` from your own Python, use the
development install below; both can be installed side by side.

**Publishing a release:** every merge to `main` builds the installer on Windows and
publishes it as a new GitHub release, which installed copies then offer to update to.
`__version__` in `owlfluids/__init__.py` is the major.minor line: the first merge after
it changes is released as that version, and each later merge takes the next patch
(0.2.0, 0.2.1, ...). Bump it only to start a new line. Merges that change only Markdown
or tests are not released.
`packaging\build.ps1` builds the same installer locally (needs Inno Setup 6).

### From source

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
| `owlfluids bench` | Time the 60-component example stand |
| `owlfluids example NAME -o file.yaml` | Write one of the examples |
| `owlfluids update` | Install a newer release (Windows installer; `--check` only reports) |

In the app: drag parts from **Parts**, drag from one port (small circle) to another
to connect, **R** rotates, **Delete** removes, and the middle mouse button drags the
schematic around. Right-click a line to put a pressure, temperature or mass flow sensor on
it; its reading shows beside it during a run. Select a part to edit its
parameters; quantities take units (`3000 psi`, `-196 C`, `0.5 in`, `49 L`).
New parts are named commodity letter, part letter, number: `OV-1` is the first oxidizer
valve, `FT-1` a fuel tank, `HR-1` a helium regulator, and line sensors add their tag
(`OPT-1`). Commodities are O oxidizer (LOX, N2O), F fuel (ethanol, methane), H helium,
N nitrogen, A air, W water, and X while no fluid reaches the part yet. Parts are V valve,
M manual valve, C check valve, R regulator, T tank, RV relief valve, O orifice and P air
pump. Fittings, tees, volumes, boundaries and pneumatic supplies are left unnamed on the
schematic (their ids, like `TEE-1`, show in the properties panel). A default name follows the
fluid wired to the part until you rename it: **F2** or double-click a part or sensor
to type a new id.
Press **Run**; valves appear in the **Valves** tab and can also be clicked on the
schematic. Manual valves there also have a slider to throttle them to any opening, and
each regulator has a setpoint box (psig) to turn its knob mid-run. **Record** logs to CSV/HDF5, and **File > Overlay a recorded log** draws
a log (for example real test data with matching column names) over the plots.

### System files

```yaml
name: GN2 bottle blowdown
components:
- {id: K1, type: tank, fluid: nitrogen, volume: 49 L, pressure: 2400 psi, temperature: 20 C}
- {id: V1, type: valve, cv: 0.3, open_time: 50 ms}
- {id: ATM, type: boundary, fluid: nitrogen}
connections:
- [K1.port, V1.in]
- [V1.out, ATM.port]
sensors:
- {id: PT-K1, type: pressure, at: K1.port, units: psi}
```

Component types: `tank` (gas, saturated liquid/vapor with `fill`, optionally with a
`wall_mass` shell that stores heat, or liquid under a separate gas with `pressurant` and
`fill`; ports `top` (gas), `bottom` (liquid) and `port`; old `pressurized_tank` parts load
as one, their `gas` and `liquid` ports becoming `top` and `bottom`), `boundary`, `volume`
(also takes `wall_mass`, for example to chill down a cryogenic line), `tee`, `fitting`,
`orifice`, `valve`, `manual_valve` (a hand valve the operator opens, closes or throttles from the
valve panel; Synnax never commands it, and old `throttle_valve` parts load as one),
`check_valve`, `relief_valve`, `regulator` (its setpoint can be turned from the valve panel during a run),
`air_pump` (a compressor with a pressure switch: `flow`, `start_pressure`,
`stop_pressure`; a `valve_position` sensor on it reads 1 while it runs),
`pneumatic_supply` (the "A" air or GN2 header pneumatic valves run on; feed it from an
`air_pump` or a bottle through a regulator).

On/off valves have `actuator: pneumatic` (the default) or `solenoid` (also the default
for type `solenoid`). A pneumatic valve is spring return: it draws air from its supply
(`air_supply`, by default the only `pneumatic_supply` in the system) while it is held
away from its normal state, so a normally closed valve draws to open and a normally
open one draws to close, and it vents to the atmosphere on the way back. The actuator
fills through its pilot (`pilot_cv`) to `actuator_volume`, starts to move at
`stroke_pressure` and is fully stroked at `full_stroke_pressure`, so a weak supply
strokes it slowly, partway or not at all, and losing the supply springs it back.
Without a `pneumatic_supply` in the system valves move on command alone.
`examples/pneumatic-valves.yaml` shows a compressor running a fill and a vent valve;
`owlfluids bench --pneumatic` times the test stand with its valves on a GN2 supply.
`owlfluids app` shows every parameter with its default and meaning.

Regulators have four behaviors, each switched with an `_on` parameter (`on`/`off`) and
sized by the parameter of the same name:

| Behavior | Default | What it does |
| --- | --- | --- |
| `droop` (`20 psi`) | on | The outlet falls as the flow rises, by `droop` at full opening. Off, the outlet settles back at the setpoint at any flow within a fraction of a second. |
| `lockup` (`5 psi`) | off | As the flow stops, the outlet climbs up to `lockup` above the setpoint before the seat seals. |
| `creep` (`5e-5`) | off | A worn seat leaks this share of the flow area, so a dead-ended outlet slowly rises toward the inlet pressure (about 4 psi a minute into 1 L for a Cv 0.1 regulator off a 3000 psi bottle). |
| `supply_effect` (`0.01`) | off | The outlet rises by this much per unit fall of inlet pressure (0.01: 1 psi per 100 psi), counted from the inlet pressure when the setpoint was last set, so it shows as a bottle blows down. |

```yaml
- {id: REG, type: regulator, cv: 0.2, setpoint: 500 psig, lockup_on: on, creep_on: on, creep: 1e-4}
```

During a run they can be switched from a script (`sim.set_regulator_behavior("REG", "creep", True)`,
`sim.regulator_behaviors("REG")`) or the console (`reg REG creep on`), for example to
fail a regulator seat partway through a test. Synnax never switches them.

Sensor types: `pressure`, `temperature`, `wall_temperature`, `mass_flow`,
`valve_position`, `mass`, `level` (a tank's liquid fill, shown in %),
`purge_fraction` (share of a node's volume that is purge gas, in %; in LOX, nitrous,
LN2 and methane lines it is the gas's share of the pressure, so a node still full of
LOX can read 30 % while gas pushes on it), `quality`
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

Lines are not modelled and are not components: connect component ports directly
(`[V1.out, INJ.in]`), and in the app a line is the wire between two ports. The
valves, orifices, regulators and fittings are the restrictions that matter on a
stand, so their Cv sets every flow; tubing adds no pressure drop, volume or
inertia. A file from before October 2026 that still has `pipe` components loads
with each one folded into a direct connection (with a warning; save it to keep the
change), and a sensor on a pipe moves to a port at the same point. Each point where components meet holds a small volume (`junction_volume`,
1 cc by default; valve and fitting bodies hold more, and a flashing liquid such as
nitrous runs smoother with about 20 cc). To keep a line's volume or its steel (for
a chilldown), draw it as a `volume`. Line friction is usually a few percent of a
feed's pressure drop: the example stand's LOX flow comes out about 4 % higher than
with friction modelled.

Lines start in equilibrium with the tank or volume they are open to: a feed line
on a LOX tank starts full of LOX at tank pressure, a regulated line starts at its
setpoint, and anything behind a closed valve starts at the system's initial
pressure and temperature (`prime_lines: false` turns this off).

### Purges

Saturated tanks (`tank` with `fill`) hold liquid and vapor in equilibrium, so a
nitrous bottle's pressure follows its temperature as it empties and cools, and a run
tank filled from the bottom while its vent is open settles on the vapor pressure of
what it holds. `wall_mass` adds the tank shell, which warms the boiling nitrous;
`film_coefficient` (default 1000 W/m2K) sets how fast, and is the value to tune
against test data.

Purges are meant to run the way stands run them: vent the line to ambient first,
then flow purge gas for a few seconds. Points the purge gas has not reached cost
about the same as any other.

Connect a purge bottle (a `tank` of helium, nitrogen or air), usually
through a regulator, a valve and a check valve, to any line; `examples/lox-purge.yaml`
purges a LOX feed with helium, and `examples/nitrous-fill.yaml` is a nitrous fill
cart whose line is dumped, swept with shop air and filled again. Where the purge
gas and the line's own fluid can both flow (check valves, relief valves,
regulators and pumps pass flow one way), each node carries both at one
temperature. Water and ethanol fill their share of the node, compressed by the
total pressure, and the gas fills the rest. Other fluids (LOX, nitrous, LN2,
methane) keep their own property tables over the node, as in a line of that fluid
alone, and the purge gas adds its partial pressure. Flow out of such a node is
homogeneous (each fluid leaves in proportion to its mass).

Only a purge gas may meet another fluid. Two propellants meeting (GOX in a methane
line) is still a build error, since that would need combustion. When two purge
gases meet (helium into a nitrogen line), the one listed first in helium,
nitrogen, air is the purge gas. A tank or `boundary` holds only its
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

Manual valves and regulators get no channels in either mode: like hand valves and
regulator knobs on the stand, only the operator works them, from the OwlFluids window.
Listing a manual valve under `valves:` is an error.

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
plus a wall temperature for a tank or volume with a shell) and **links** (orifices,
valves, fittings, regulators and pumps, all quasi-steady: their flow follows from
the pressures on either side). Every 1 ms step is backward Euler,
solved by chord Newton iterations on a banded LU of a finite-difference Jacobian
that is coloured, reordered (reverse Cuthill-McKee) and updated incrementally.
The kernel is compiled with Numba and releases the GIL. Fluid properties come from
(density, internal energy) tables with a boundary-fitted compressed-liquid table,
so subcooled LOX and two-phase states match CoolProp to about 1% in pressure
(0.5% for 99% of states).

Validation tests (`pytest`) cover isentropic gas blowdown, closed-system
equalization (mass and energy conserved), Torricelli drain, table accuracy against CoolProp, LOX feed priming and
flow, a self-pressurizing LN2 dewar, nitrous blowdown, line chilldown, purges with
every purge gas (mass conserved, LOX, water, ethanol and methane pushed out), the
nitrous fill-purge-refill sequence, air pump switching, and the real-time gate on
the 60-component stand, purges included.

## Limits worth knowing

- One fluid per connected line, plus one purge gas (separate propellants with a
  tank). A purge gas is an ideal gas with constant heat
  capacity and does not dissolve in the liquid. In LOX, nitrous, LN2 and methane
  lines it spreads over the whole node, so a gas pocket pushing on liquid there
  presses less hard than a real one: purges clear these lines, a little more
  gently. For the same reason, LOX trapped at saturation under a purge gas (a dead
  leg behind a closed valve) is stiff, so a purge into a LOX line can take some
  forced steps even with a large budget. Near zero flow, nodes on either side of a
  link swap a small stream (this keeps Newton from circling the flow reversal), so
  a purge gas and a line's fluid sitting still across an open valve slowly blend.
  The purge gas never condenses: GN2 in a line still at LOX temperature would
  really liquefy above about 50 psi. An air pump delivers its rated mass flow at a
  fixed temperature (after its aftercooler) whatever the outlet pressure.
- Two-phase flow is homogeneous equilibrium. Violent events (LOX into a warm line,
  geysering) are stable and conserve mass and energy but are not resolved in
  detail; the status bar counts "forced steps" when a step used its whole budget.
  For runs that must match across machines, pass `Simulation(system, step_budget=200)`.
- Property tables blend pressure within about one grid cell of saturated liquid,
  so a saturated tank filled above ~98 % can read a few percent high (up to ~13 % at
  99 %). States within a kelvin or so of the critical point are also approximate.
- Lines have no friction, volume or inertia, so there is no water hammer and no
  line chilldown unless a line is drawn as a `volume` with a `wall_mass`. Wall heat
  transfer uses a fixed film coefficient.
