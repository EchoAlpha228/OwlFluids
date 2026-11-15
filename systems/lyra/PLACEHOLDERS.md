# Lyra feed system: placeholders and modelling notes

`lyra-feed.yaml` is built from the P&ID *Lyra Feed System Cleaner* (FRL, 11/8/2026). Every
component and sensor keeps its P&ID tag. Values marked `PLACEHOLDER` in the file are guesses
made so the system runs; this page lists them, roughly in order of how much they change what
the simulation shows.

Taken from the P&ID (not placeholders): orifice sizes FPO 0.082", OPO 0.144", IFO 0.04",
IOO 0.035", OTVO 3/32"; set pressures FPRV 1000 psig, EPRV 375 psig, OTRV 1070 psig,
OFRV 940 psi, OFBD 1500 psi; tube sizes (1/4", 3/8", 1/2", 3/4" with the drawing's standard
walls); which valves are pneumatic (NC or NO), solenoid or manual.

## Numbers that matter most

| What | Placeholder | Why it matters |
| --- | --- | --- |
| GN2 supply to the fuel side (GN2-F) | 700 psig | Sets the ethanol tank pressure the press settles at (there is no regulator on the P&ID's press line, so it is the facility supply's regulated pressure) |
| GN2 supply to the ox side (GN2-O) | 900 psig | Sets the run tank's supercharge pressure |
| FMV, OMV Cv | 3.0, 4.0 | Feed flow and the pressure drop to the injector |
| Injector CdA (INJ-F, INJ-O) | 20 mm², 30 mm² | Mass flow and mixture ratio; with no combustion the "chamber" is the atmosphere |
| ETH tank volume, fill, diameter | 15 L, 90 %, 0.15 m | Run time, ullage blowdown |
| N2O run tank volume, fill, temperature, wall | 20 L, 85 %, 20 °C (about 720 psig), 8 kg aluminum | Run time, vapor pressure, how fast it cools as it boils |
| Venturis FV, OV (permanent loss) | Cv 4, Cv 6 | Pressure drop through the flowmeters |
| FPV, OPV Cv | 0.4, 0.6 | Press rate (the P&ID orifices FPO and OPO usually limit it anyway) |

## Everything else

- **Volumes and charges:** N2O supply bottle (N2O-SUP, 30 L, 80 % full, 20 °C, 15 kg aluminum),
  purge bottle GN2 (5 L at 300 psig), GCH4 and GOX igniter bottles (1 L at 1500 psi each), air
  receiver AR (20 L at 120 psig), pneumatic header A (2 L).
- **Setpoints:** EPPR 300 psig, IFPR and IOPR 150 psig, APR 100 psig; compressor AC 3 g/s, on at
  110 psig, off at 125 psig.
- **Cv of every other valve, check valve, relief valve, filter and the burst disk**, and the
  discharge coefficient (0.7) of the five P&ID orifices.
- **Valve stroke times:** solenoids 20-30 ms, pneumatic mains 150 ms, vents 100 ms, OFV 300 ms.
  Pneumatic valves use the default actuators (100 cc, stroke from 30 to 60 psig, pilot Cv 0.1).
- **Line lengths:** all of them (flex hoses 1.5 m). Line diameters come from the P&ID.
- **Starting state:** ethanol loaded and unpressurized; run tank loaded and self-pressurized at 20 °C
  (`fill: 0, pressure: 1 atm` starts it empty for a fill); NI-F, NTI-O, OFTV and every automated valve closed; ODV-M closed; AIV, IFTV and IOTV open.

## Names added or changed

- **OPCV appears twice on the P&ID.** The one on the run tank press line keeps `OPCV`; the one
  after OPGS (purging the fill line) is `OPGCV` here, the name the nitrous-fill example uses.
- The four igniter pressure gauges have no tags: `IFPG-1`, `IFPG-2`, `IOPG-1`, `IOPG-2`.
- Untagged parts get descriptive ids: `GN2-F`, `GN2-O` (the "GN2" supplies), `N2O-SUP` (the
  supply bottle; the run tank is `N2O`), `AC` and `AR` (compressor and "Air" receiver), `A`
  (pneumatic header), `TEE-*` junctions, `VENT-*` vents, `INJ-F`, `INJ-O`, `ATM-F`, `ATM-O`
  (the engine), `IGN-F`, `IGN-O` (the igniter ports).
- `FV` and `OV` are the venturis the flowmeters FFM and OFM read. FVPT-1/OVPT-1 read the inlet and
  FVPT-2/OVPT-2 the outlet: a lumped model has no throat pressure.

## Modelling notes and limits

- **Run tank ullage.** It holds N2O vapor and any GN2 pressed in, at one temperature: venting it
  boils the liquid to hold its vapor pressure (and cools it), and a supercharge adds the GN2's
  partial pressure. GN2 does not dissolve in the liquid N2O. A fill from empty works, but this
  solver needs a fresh Jacobian every step while the tank fills, so on a slow computer the status
  bar counts forced steps then (they converge to the same answer).
- **ODV-M is a 3-way valve with a DES port.** It is modelled as a hand valve from ODV to the
  vent; the DES path is left out until we know what it does.
- **No combustion.** The injectors flow to the atmosphere, as on a cold flow, and the igniter
  lines end at the engine.
- Opening a press valve into a short dead-ended line at 1 atm briefly heats the trapped gas
  past the 500 K property limit (the app notes it). It is real compression heating, without the
  line walls that cool it in practice.
