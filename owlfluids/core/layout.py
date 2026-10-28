"""Column layout of the packed network arrays shared by the compiler and kernel."""

# ---- node integer columns
N_KIND, N_FLUID, N_PARTNER, N_SX, N_WALL, N_GAS, N_GX = range(7)
N_NI = 7
# N_GAS: the purge gas mixed into the node's own fluid (-1 for a node holding one fluid);
# N_GX: index of its mass state (its energy follows), after the node's other states
# node kinds
NODE_VOLUME = 0
NODE_BOUNDARY = 1
NODE_TANK_LIQUID = 2  # liquid region of a pressurized tank; pressure comes from its ullage partner

# ---- node float columns
(NF_VOL, NF_RHOREF, NF_AREA, NF_HA, NF_WALLMC, NF_HAAMB, NF_TAMB,
 NF_BP, NF_BT, NF_BRHO, NF_BU, NF_MREF) = range(12)
N_NF = 12

# ---- node outputs
(O_P, O_T, O_X, O_RHO, O_H, O_M, O_TW, O_GAS) = range(8)
N_NO = 8  # O_GAS: volume fraction of purge gas in a mixed node (0 elsewhere)

# ---- link integer columns
# every link is a quasi-steady flow element (orifices, valves, fittings, regulators, pumps)
L_A, L_B, L_DRAWA, L_DRAWB, L_CTRL = range(5)
N_LI = 5
# what a link end draws from a two-phase node
DRAW_MIX = 0
DRAW_LIQUID = 1
DRAW_VAPOR = 2
# how the opening is set
CTRL_FIXED = 0
CTRL_VALVE = 1  # commanded, with actuator travel time
CTRL_CHECK = 2  # opens on forward pressure difference
CTRL_RELIEF = 3  # opens when upstream exceeds downstream by the set pressure
CTRL_REGULATOR = 4  # throttles to hold the downstream pressure at the setpoint
CTRL_PUMP = 5  # fixed mass flow from its inlet while running; on below LF_P1, off above LF_P2 (LF_CDA: kg/s)
CTRL_AIR = 6  # a pneumatic actuator's draw on its air supply (a); LF_CDA: pilot Cd*A, LF_P1/LF_P2: gauge
# pressures where the valve starts to stroke and is fully stroked

# ---- link float columns
LF_CDA, LF_P1, LF_P2, LF_LOCK, LF_LEAK, LF_SPE, LF_PREF, LF_RESET = range(8)
N_LF = 8
# Regulator behaviors (0 = off): LF_LOCK the rise above setpoint where the seat seals at
# zero flow (lockup), LF_LEAK the share of its flow area that leaks past the closed seat
# (creep), LF_SPE the outlet rise per unit of inlet fall (supply pressure effect) from
# LF_PREF, the inlet pressure when the setpoint was last set. LF_P2 is the droop band;
# with droop off, LF_RESET (1/s) is the rate at which V_BIAS raises the setpoint to win
# the droop back, so the outlet settles at the setpoint at any flow.

# ---- valve actuator columns (runtime)
V_POS, V_CMD, V_RATE_OPEN, V_RATE_CLOSE, V_AIR, V_TGT, V_VALVE, V_PACT, V_VOL, V_NO, V_Q, V_EXCH, V_BIAS = range(13)
N_V = 13
# V_AIR: a pneumatic valve's actuator link (-1 for none); V_TGT: the position its actuator
# pressure drives it to. On an actuator link (CTRL_AIR): V_VALVE its valve link, V_PACT the
# actuator pressure (absolute), V_VOL its volume, V_NO 1 when air strokes the valve closed
# (normally open), V_Q its draw on the supply over the current step, V_EXCH its volume when
# it exchanges air with the supply this step (else 0). On a regulator link, V_BIAS: the
# pressure (Pa) its droop reset adds to the setpoint (0 with droop on).
# V_AIR and V_VALVE hold link indices as integer-valued floats.

# ---- link outputs
LO_MDOT, LO_OPEN = range(2)
N_LO = 2
