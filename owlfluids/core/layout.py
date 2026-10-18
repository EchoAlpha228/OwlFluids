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

# ---- link float columns
LF_CDA, LF_P1, LF_P2 = range(3)
N_LF = 3

# ---- valve actuator columns (runtime)
V_POS, V_CMD, V_RATE_OPEN, V_RATE_CLOSE = range(4)
N_V = 4

# ---- link outputs
LO_MDOT, LO_OPEN = range(2)
N_LO = 2
