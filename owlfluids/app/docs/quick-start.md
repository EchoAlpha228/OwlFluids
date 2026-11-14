# OwlFluids quick start

OwlFluids pretends to be your test stand. It simulates a fluid system (tanks,
valves, regulators, orifices) in real time, and you open and close its valves
yourself, from a sequence script, or from Synnax. With Synnax, its simulated
sensors stream into Synnax channels and Synnax valve channels drive the
simulated valves, so the Console, schematics and Arc sequences see ordinary
channels.

This guide is also in the app: **Help > Quick start guide** (F1).

## 1. Install and update

1. Download `OwlFluids-Setup-<version>.exe` from the
   [latest release](https://github.com/EchoAlpha228/OwlFluids/releases/latest).
2. Run it. It installs for your user only (no admin prompt) into
   `%LOCALAPPDATA%\Programs\OwlFluids`. If Windows SmartScreen says it protected
   your PC, click **More info**, then **Run anyway**: the installer isn't
   code-signed.
3. Keep **Add OwlFluids to PATH** ticked so `owlfluids` works in new PowerShell
   windows. The installer brings its own Python, the Synnax client and prebuilt
   fluid property tables, so there is nothing else to install.
4. Open **OwlFluids** from the Start menu, or type `owlfluids` in a new
   PowerShell window.

**Updates.** The app checks for a new release each time it starts and offers to
install it. **Help > Check for updates…** checks on demand; it downloads the
installer, closes OwlFluids, upgrades it and reopens it. A check during a run
only leaves a note in the status bar. Every merge to `main` publishes a new
release. **Help > Update settings…** turns the start-up check off or takes an
optional GitHub token (only needed if GitHub limits anonymous checks).

**Your own files.** An update replaces the installed examples, so copy them to a
folder of your own before editing them:

```powershell
New-Item -ItemType Directory C:\FRL\OwlFluids -Force | Out-Null
Copy-Item "$env:LOCALAPPDATA\Programs\OwlFluids\examples" C:\FRL\OwlFluids -Recurse -Force
```

**From source** (to develop OwlFluids, or `import owlfluids` from your own
Python), with Python 3.12 or newer:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[app,hdf5,synnax]"
owlfluids tables
```

`owlfluids tables` builds the property tables once (a few seconds per fluid).
After updating the code, run the `pip install` line again.

## 2. Your first run

1. Open the app and pick **File > Examples > lox-feed**: a nitrogen-pressurized
   LOX tank feeding an injector through the main valve `V-M`. **File > Open…**
   opens any `.yaml` system and **File > Recent** reopens one.
2. Press **Run** (F5). The plots open under the schematic, and the **Speed** box
   on the toolbar runs it at real time or faster.
3. Click `V-M` on the schematic, or its button in the **Valves** tab, to open
   it. Open valves turn green and closed ones red; flowing lines carry moving
   white squares. Live values show beside each part and in the plot list.
4. **Pause** (F6), **Reset** (Ctrl+R) and **Stop** (Shift+F5) are in the
   **Run** menu. **Record** (Ctrl+L) logs sensors and valves to CSV or HDF5, and
   **File > Overlay a recorded log** draws a log, such as real test data, over
   the plots.

The plot panel's **LIVE** button follows the run; press it to hold the view
still (PAUSED) while you pan and zoom, and press it again to follow.

| Example | What it shows |
| --- | --- |
| `lox-feed` | N2-pressurized LOX tank feeding an injector (used in this guide) |
| `gn2-blowdown` | GN2 bottle blowdown |
| `ln2-wet-dress` | LN2 wet dress |
| `n2o-blowdown` | Saturated nitrous tank blowdown |
| `nitrous-fill` | Nitrous fill cart with an air purge, dump and refill |
| `lox-purge` | Purging a LOX line with helium |
| `pneumatic-valves` | A compressor feeding the air supply that strokes a fill and a vent valve |
| `test-stand` | Full 60-component LOX/ethanol stand with purge valves |

## 3. Building a system

- **Add parts** by dragging them from **Parts**. Drag from one port (the small
  circle) to another to connect them. The Cv of the valves, regulators, fittings
  and orifices sets every flow. Each line is 1 m of tubing by default, and its
  fluid has inertia, so a fast close shows water hammer; click a line to set its
  `length` and inner `diameter` in **Properties** (`0` for parts bolted
  together), or the defaults for every line in **Edit > Settings…**.
- **Edit** a selected part in **Properties**. Values take units: `3000 psi`,
  `-196 C`, `0.5 in`, `49 L`. A value that can't be read turns red with the
  reason under it.
- **R** rotates clockwise, **Shift+R** back, **Delete** removes, **F**
  fits the drawing to the window, and the middle mouse button drags it around.
  Drag a line segment sideways to reshape it; double-click it to reset.
- **Sensors:** drag a pressure transducer, thermocouple or flow meter from
  **Parts > Sensors** onto any point of any line (or right-click a line). It
  reads what the fluid does right there; drag it along the line or onto another
  to move it. Its reading shows beside it during a run. A tank's level, mass and
  a valve's position are added from that part's properties.
- **Names:** new parts are named commodity letter, part letter, number: `OV-1` is
  the first oxidizer valve, `FT-1` a fuel tank, `HR-1` a helium regulator, and
  line sensors add their tag (`OPT-1`). Commodities are O oxidizer, F fuel, H
  helium, N nitrogen, A air, W water, and X until a fluid reaches the part. Parts
  are V valve, M manual valve, C check valve, R regulator, T tank, RV relief
  valve, O orifice and P air pump. **F2** or a double-click renames a part or
  sensor.
- **Edit > Settings…** (Ctrl+,) holds the time step, ambient conditions, the
  property backend and the sensor noise switch.

Sensors read with realistic noise and lag for their type (pressure
transducers get noisier while fluid flows, thermocouples lag about 0.3 s).
Temperatures default to °F, pressures to psi. Set `sensor_noise: off` under
`settings:` in the system file for exact values, or `units:` on a sensor to
change its units.

## 4. Valves, regulators and pneumatics

- **On/off valves** are opened and closed by clicking them, by a script, or by
  Synnax.
- **Manual valves** are hand valves: the Valves tab has a slider to throttle
  them to any opening. Synnax never commands them.
- **Regulators** have a setpoint box (psig) in the Valves tab that you can change
  mid-run, like turning the knob. Each can also droop, lock up, creep or follow
  its supply pressure; see `README.md` for the settings.
- **Pneumatic valves.** Add a `pneumatic_supply` (the "A" air or GN2 header),
  fed from an `air_pump` or a bottle through a regulator, and pneumatic valves
  draw air from it to stroke. A weak supply strokes them slowly or partway, and
  losing it springs them back. With no pneumatic supply in the system, every
  valve moves on command alone.

## 5. Connecting to Synnax

Synnax gives you two ways to run a Core, and OwlFluids works with both.

| Your Synnax | Pick as the core | Login |
| --- | --- | --- |
| **Synnax Desktop 1.1 or newer** on this PC (Core 0.61) | *Synnax Desktop on this computer* | None: OwlFluids reads Desktop's API key |
| A **standalone Core** (`synnax start`, or the stand's server) | *a Synnax Core at an address* | Username and password, default `synnax` / `seldon` |

### Synnax Desktop (easiest)

Synnax Desktop runs its own Core inside the app with a random password it never
saves. Desktop 1.1 and newer share that Core with programs on the same PC by
writing its address and an API key to `~/.synnax/desktop.json` each time it
opens. OwlFluids reads that file at every connect.

1. Install **Synnax Desktop 1.1** or newer from the
   [Synnax docs](https://docs.synnaxlabs.com) and open it. Wait for its
   workspace to load.
2. In OwlFluids, open your system (for example **File > Examples > lox-feed**),
   then **Run > Synnax setup…**. This works while editing.
3. Under **Connection**, set **core** to *Synnax Desktop on this computer*. It is
   already picked when Desktop's file exists. The server and login rows go away,
   and the note under them says where Desktop's Core is.
4. Press **Test connection**. Green means connected, with the Core's version and
   how many of your channels already exist. Amber is a warning, such as a version
   mismatch. Red means it couldn't connect, with the reason.
5. Carry on with **Setting up the test** below.

If OwlFluids says Desktop isn't running, open Desktop and try again: it leaves
its file behind when it quits. **Reset key** in Desktop's **Connect from code**
window ends OwlFluids' session; connect again to pick up the new key. Desktop
1.0.x and older don't share their Core at all: update Desktop, or use a
standalone Core.

### A standalone Core

1. Start the Core, for example `synnax start` in its own PowerShell window
   (`--help` lists the flags for running insecure or in memory). It runs in the
   foreground and is ready when it prints `Synnax is running and available at
   localhost:9090`. Warnings about NI-DAQmx, LabJack or Npcap are harmless.
2. Connect the Synnax Console to it to look at data.
3. In OwlFluids, **Run > Synnax setup…**, set **core** to *a Synnax Core at an
   address*, and enter the server and port (`localhost` : `9090` for one on this
   PC), username and password. Tick **TLS** if the Core uses certificates.
4. Press **Test connection**, as above.

The login is kept in your own app settings on this PC and never goes in the
system file, so it stays out of git.

### Setting up the test

Still in **Run > Synnax setup…**:

1. Under **Test**, pick the **mode**:
   - **HOOTL** (no hardware): the sim reads each valve's command channel and
     fakes its state channel.
   - **HITL** (real valves): the sim reads the state channels the hardware
     reports and never writes them; only sensors are faked.
2. Leave **valve state** on *echoes the command* unless your stand has real
   position feedback. **rate** (100 Hz), **index channel** (`owl_time`),
   **create channels** and **version mismatch** can usually stay as they are.
3. The **Sensors** and **Valves** tables list every sensor and automated valve
   with its default channel: a sensor's id in lower case (`PT-TK` to `pt_tk`),
   and `<valve>_cmd` / `<valve>_state` for valves. Type over a name to use the
   stand's real channel, or untick a row to leave that part out. A blank name or
   two parts on one channel turns red. The line under the tables says what will
   be written and read.
4. Press **Save**, then save the system file (Ctrl+S). Everything except the
   login goes in the file's `synnax:` section.

**Load file…** fills the window from a YAML file with a `synnax:` section, such
as `examples\synnax.yaml`, and **Save file…** writes one for the command line.

### Running against Synnax

1. Press **Run** (F5).
2. **Run > Connect to Synnax…** shows the same window with a **Connect** button.
   Check it and press **Connect**.
3. The status bar shows `Synnax ↑` (writes sent) and `↓` (valve channel updates
   received). Both counting up means the link works.
4. In Synnax Desktop or the Console, plot `pt_tk`, `pt_inj`, `tc_inj` (°F),
   `fm_lox` and `ll_lox` (%) against `owl_time`. Tank pressure sits near 300 psi
   with a little noise.
5. Write 1 to `mpv_cmd` (a valve symbol on `mpv_cmd` / `mpv_state` in a
   schematic is easiest). `mpv_state` goes to 1, `fm_lox` jumps, `pt_inj` rises
   and `ll_lox` starts falling. Write 0 to close it.

Clicking a valve in OwlFluids still works while connected; the next Synnax write
to that valve takes over. Click **Run > Connect to Synnax…** again to
disconnect.

## 6. Commanding valves from Synnax

Anything that writes a valve's command channel moves the simulated valve:

- **Schematics:** a valve symbol with command channel `mpv_cmd` and state channel
  `mpv_state`. In HOOTL the sim fakes the state, so the symbol behaves as it does
  on the stand.
- **Arc sequences:** point the valves at the channels your sequences already use,
  for example `V-M: {cmd: ox_mpv_cmd, state: ox_mpv_state}`. Sequences that wait
  on valve states, and aborts triggered by simulated pressures, run unchanged.
- **Python:** for a standalone Core,

```python
import synnax as sy

client = sy.Synnax(host="localhost", port=9090, username="synnax", password="seldon", secure=False)
with client.open_writer(sy.TimeStamp.now(), ["mpv_cmd"]) as w:
    w.write({"mpv_cmd": 1.0})  # 1 = open, 0 = close
```

For Synnax Desktop, copy the connection code from Desktop's **Connect from
code** window. A value above 0.5 opens a valve and below closes it. On connect
OwlFluids reads each channel's last value, so valves start where Synnax says.
Manual valves and regulators get no channels.

## 7. Hardware in the loop

1. Set **mode** to **HITL** and map each sensor and valve to the stand's real
   channels. Existing channels keep their own index, such as `ai_time`.
2. **Stop the hardware's sensor read task** first, since the sim writes those
   channels. The driver stops with an error rather than lose writes if another
   writer holds a channel.
3. Point each valve's state channel at the one the hardware reports. The sim
   reads it and never writes it.
4. Keep faked sensors off the valve state index: the driver refuses to start if a
   sensor it writes shares an index channel with a valve state it reads.
5. Keep **valve state** on *echoes the command* unless the stand has real
   position feedback; with limit switches only, the sim valve would stay shut for
   the whole stroke.
6. Untick **create channels**, so a typo in a channel name stops the run with a
   list of missing channels instead of creating a new one.

Set `units:` on each sensor in the system file to match the real channel (psi vs
psig, °F vs K), and keep sensor noise on so alarms and abort limits see
realistic signals.

## 8. The command line

| Command | What it does |
| --- | --- |
| `owlfluids app [file.yaml]` | The desktop app (also plain `owlfluids`) |
| `owlfluids console file.yaml` | Real-time run with typed commands: `open V-M`, `close V-M`, `set TV-1 0.4`, `status`, `help` |
| `owlfluids run file.yaml --script seq.py --log run.csv` | Headless run of a sequence; `--rate 0` runs as fast as possible |
| `owlfluids run file.yaml --synnax` | Run against Synnax with the system file's `synnax:` section |
| `owlfluids run file.yaml --synnax examples\synnax.yaml` | The same, with that file's settings on top |
| `owlfluids check file.yaml` | Build the network and print a summary (catches wiring mistakes) |
| `owlfluids example NAME -o file.yaml` | Write one of the examples (`owlfluids example` lists them) |
| `owlfluids bench` | Time the 60-component example stand |
| `owlfluids update` | Install a newer release (`--check` only reports) |

An example name works anywhere a file does (`owlfluids run lox-feed --synnax examples\synnax.yaml`).
`--synnax-mode hitl` overrides the mode, `--duration 60` stops after 60
simulated seconds, and `--log run.csv` keeps a local copy. For Synnax Desktop on
the command line, set `desktop: true` in the `synnax:` section.

### The synnax: section

| Setting | Default | What it does |
| --- | --- | --- |
| `desktop` | `false` | `true`: Synnax Desktop on this PC, read from `desktop.json`; host and login are ignored |
| `host`, `port` | `localhost`, `9090` | Where a standalone Core is |
| `username`, `password` | `synnax`, `seldon` | Its login |
| `secure` | `false` | `true` for a TLS Core |
| `mode` | `hootl` | `hootl` or `hitl` |
| `valve_state` | `command` | `command` (echoes the command) or `position` (real position feedback) |
| `rate` | `100` | Hz written to Synnax |
| `index` | `owl_time` | Index channel for channels OwlFluids creates |
| `create_channels` | `true` | `false` stops with a list of missing channels |
| `sensors` | every sensor, id in lower case | Sensor id to channel |
| `valves` | every valve, `<valve>_cmd` and `<valve>_state` | Valve id to `{cmd: ..., state: ...}` |
| `exclude` | none | Sensors and valves that get no channels (the unticked rows in Synnax setup) |
| `valve_threshold` | `0.5` | A value above this opens an on/off valve |
| `check_version` | `warn` | Core and client major.minor differ: `warn`, `strict` (refuse) or `off` |

If you list any sensors or valves, only the listed ones are connected.

## 9. Keyboard shortcuts

| Keys | Does |
| --- | --- |
| F1 | This guide |
| Ctrl+N, Ctrl+O, Ctrl+S | New, open, save |
| F5, F6, Shift+F5, Ctrl+R | Run, pause, stop, reset |
| Ctrl+L | Record |
| R, Shift+R | Rotate clockwise, counter-clockwise |
| F2 or double-click | Rename |
| Delete | Delete |
| F | Fit the schematic |
| Ctrl+, | Settings |
| Middle mouse drag | Pan |

## 10. Troubleshooting

| What you see | Likely cause | Fix |
| --- | --- | --- |
| *Synnax Desktop hasn't shared its Core* (no `desktop.json`) | Desktop 1.0 or older, or Desktop never opened | Update to Synnax Desktop 1.1 or newer and open it |
| *Synnax Desktop isn't running* | Desktop is closed; its file stays behind | Open Desktop, wait for the workspace, try again |
| *turned down the API key* | **Reset key** was pressed in Desktop | Connect again to read the new key |
| *cannot reach the Synnax Core* | Core not running, or wrong host or port | Start it, and use the host and port the Console connects to |
| Login error | Wrong username, password or TLS setting | Match what the Console uses |
| Warning: Core and client *differ in major.minor* | OwlFluids' bundled client and the Core are on different releases | Upgrade the side the warning names; Help > Check for updates brings the newest client |
| *refused because check_version is strict* | Versions differ and **version mismatch** is *refuse to connect* | Match the versions, or set it to *warn* |
| *missing Synnax channels: ...* | **create channels** is off and a name is wrong | Fix the name, or create the channel in Synnax |
| *Synnax index ... stamps channels the driver writes and also ...* | HITL: a faked sensor shares an index with a valve state the sim reads | Put those sensors on another index, or stop that task |
| *Synnax link stopped* in the app, or *Synnax driver error* on the command line | Core dropped, or another writer holds a channel | Stop the other writer (often the hardware read task) and connect again |
| Valve doesn't move | HOOTL: writing a channel other than its command. HITL: hardware state not changing. Value at or below 0.5 | Check the valve's channels; write 1 to open |
| A manual valve under `valves:` is an error | Manual valves take no Synnax channels | Remove it from the Synnax settings |
| `owlfluids` not recognized in PowerShell | Window opened before the install, or PATH unticked | Open a new window, or reinstall with the box ticked |
| `synnax start` seems to hang | It runs in the foreground | Wait for *Synnax is running*, leave it open, use another window |
| Console log repeats `invalid token: auth error` | The Console is logged in to an older server's database | Remove and re-add the cluster in the Console |
| GitHub refused the update check | GitHub limits anonymous checks | Try later, or add a token in Help > Update settings |

**Updating Synnax.** OwlFluids needs its Synnax client and the Core on the same
major.minor version. Each installer bundles the newest client when it is built,
so after upgrading Synnax, take the latest OwlFluids release (**Help > Check for
updates…**). From source, run `pip install -U synnax`. If a stand has to stay on
an older Core, a maintainer can run **Actions > Release > Run workflow** with its
`synnax_version` (for example `0.60.*`) and install that build.

More detail on every part, fluid and setting is in `README.md` in the
[repository](https://github.com/EchoAlpha228/OwlFluids).
