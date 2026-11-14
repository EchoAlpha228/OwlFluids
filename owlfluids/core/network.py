"""Turns a System into the packed arrays the kernel steps.

Steps: group connected ports into nets, let each component create its nodes and
links, give every remaining net a small junction volume, spread fluids along the
links, set initial states, then work out the Jacobian sparsity, its colouring and
a band-narrowing order of the unknowns.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from .. import units
from ..components.library import get_type
from ..fluids.props import state as _state
from ..fluids.registry import (
    FLUIDS,
    I_CV,
    I_R,
    L_CV,
    L_K,
    L_PREF,
    L_PSAT,
    L_RHO0,
    L_T0,
    L_U0,
    MODEL_LIQUID,
    MODEL_TABLE,
    P_OFF,
    PURGE_GASES,
    FluidSet,
    canonical,
)
from ..io.system import System
from . import layout as L

LINE_DIAMETER_MIN = 0.0046  # m, the bore of 1/4 in tubing: the smallest line "auto" sizes
GAS_TRACE = 1e-6  # least gas partial pressure (as a share of the total) a liquid node starts its solve from


class BuildError(ValueError):
    """The system cannot be turned into a network; the message says why."""


@dataclass
class _Node:
    name: str
    kind: int
    fluid: str | None
    volume: float
    init: tuple | None
    owner: str
    heat: tuple = (0.0, 293.15)
    wall: tuple | None = None
    partner: int = -1
    rhoref: float = 0.0
    area: float = 0.0
    bp: float = 0.0
    bt: float = 0.0
    primed: bool = False  # initial state copied from the volume it is open to
    gas: str | None = None  # purge gas that can mix into this node's fluid
    sat: bool = False  # a tank liquid that starts saturated, under its own vapor
    iface: tuple = (0.0, 0.0, 0.0)  # tank liquid surface conductances (see layout.NF_IFC)
    gas_fill: bool = False  # a mixed node that starts full of its purge gas
    phase: str = "auto"  # a boundary exactly on its saturation line: liquid, vapor, or auto (vapor)


@dataclass
class _Link:
    name: str
    a: int
    b: int
    owner: str
    cda: float = 0.0
    ctrl: int = L.CTRL_FIXED
    p1: float = 0.0
    p2: float = 1.0
    draw_a: int = L.DRAW_MIX
    draw_b: int = L.DRAW_MIX
    valve: tuple | None = None
    air: tuple | None = None  # pneumatic actuator: (its valve link, volume, normally open)
    reg: tuple = (0.0, 0.0, 0.0, 0.0)  # regulator lockup rise, seat leak, supply effect, droop reset rate
    nets: tuple = (-1, -1)  # the points (ports a tee joins) its ends are on; -1 for an end no port reaches
    area: float = 0.0  # a line's flow area


@dataclass
class Network:
    """Everything the kernel needs, plus the names that map it back to the system."""

    system: System
    fluids: FluidSet
    dt: float
    NI: np.ndarray
    NF: np.ndarray
    LI: np.ndarray
    LF: np.ndarray
    LV: np.ndarray
    x0: np.ndarray
    xscale: np.ndarray
    node_names: list[str]
    link_names: list[str]
    node_owner: list[str]
    link_owner: list[str]
    port_nodes: dict[str, int]
    comp_nodes: dict[str, list[int]]
    comp_links: dict[str, list[int]]
    valves: dict[str, int]
    col_ptr: np.ndarray = field(repr=False, default=None)
    col_rows: np.ndarray = field(repr=False, default=None)
    color_ptr: np.ndarray = field(repr=False, default=None)
    color_cols: np.ndarray = field(repr=False, default=None)
    iperm: np.ndarray = field(repr=False, default=None)
    sowner: np.ndarray = field(repr=False, default=None)
    nl_ptr: np.ndarray = field(repr=False, default=None)
    nl_idx: np.ndarray = field(repr=False, default=None)
    kl: int = 0
    ku: int = 0
    warnings: list[str] = field(default_factory=list)
    mix_init: np.ndarray = field(repr=False, default=None)
    pumps: dict[str, int] = field(default_factory=dict)  # component id -> its link
    actuators: dict[str, int] = field(default_factory=dict)  # pneumatic valve id -> its actuator link
    # "COMP.port" -> [(link, sign), ...]: the flow along the line at that port is the sum of sign * mdot
    port_flows: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
    # (a, b) for each connection -> [(link, sign), ...]: the flow along that line from a to b,
    # or None where a line has no flow of its own (see core/lines.py)
    line_flows: dict[tuple[str, str], list[tuple[int, float]] | None] = field(default_factory=dict)

    @property
    def n_states(self) -> int:
        return len(self.x0)

    @property
    def n_colors(self) -> int:
        return len(self.color_ptr) - 1


class NetworkBuilder:
    def __init__(self, system: System, pressed: dict[str, str] | None = None):
        self.system = system
        # saturated tanks that a separate gas can be pushed into: tank id -> that gas
        self.pressed: dict[str, tuple] = {}
        self._already_pressed = dict(pressed or {})
        self.nodes: list[_Node] = []
        self.links: list[_Link] = []
        self.net_of: dict[tuple[str, str], int] = {}
        self.net_node: dict[int, int] = {}
        self.net_draw: dict[int, int] = {}
        self.net_volume: dict[int, float] = defaultdict(float)
        self.comp_nodes: dict[str, list[int]] = defaultdict(list)
        self.comp_links: dict[str, list[int]] = defaultdict(list)
        self.valves: dict[str, int] = {}
        self.pumps: dict[str, int] = {}
        self.actuators: dict[str, int] = {}
        self.exhausts: dict[int, int] = {}  # air supply node -> where its actuators' air goes
        self.assumed_pneumatic: dict[str, list[str]] = defaultdict(list)  # supply -> valves not set either way
        self.air_stroke: dict[int, float] = {}  # supply node -> lowest gauge pressure that moves one of its valves
        self.warnings: list[str] = list(system.notes)
        self.vented: set[tuple[str, str]] = set()  # unconnected ports that vent to the atmosphere
        self.p_init = units.parse(system.setting("initial_pressure"), "pressure")
        self.t_init = units.parse(system.setting("initial_temperature"), "temperature")
        self.junction_volume = units.parse(system.setting("junction_volume"), "volume")

    # ---- nets
    def _group_nets(self) -> None:
        parent: dict[tuple[str, str], tuple[str, str]] = {}

        def find(p):
            while parent[p] != p:
                parent[p] = parent[parent[p]]
                p = parent[p]
            return p

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        comps = {c.id: c for c in self.system.components}
        for c in self.system.components:
            ct = get_type(c.type)
            for p in ct.ports:
                parent[(c.id, p)] = (c.id, p)
            if ct.merges_ports:
                for p in ct.ports[1:]:
                    union((c.id, ct.ports[0]), (c.id, p))
        # each connection is a line between two points, which it keeps apart (see _add_lines);
        # one of length 0 joins parts bolted straight together into one point
        self.line_ends = []
        self.connected = set()
        for a, b in self.system.connections:
            pa, pb = self._port(a, comps), self._port(b, comps)
            self.connected |= {pa, pb}
            if self._line_length(a, b) == 0.0:
                union(pa, pb)
            else:
                self.line_ends.append((pa, pb, a, b))
        roots: dict[tuple[str, str], int] = {}
        for p in parent:
            r = find(p)
            self.net_of[p] = roots.setdefault(r, len(roots))

    def _line_length(self, a: str, b: str) -> float:
        spec = self.system.line_spec(a, b)
        raw = spec["length"] if spec.get("length") is not None else self.system.setting("line_length")
        length = units.parse(raw, "length")
        if length < 0.0:
            raise BuildError(f"the line {a} -- {b} has a negative length")
        return length

    @staticmethod
    def _port(ref: str, comps) -> tuple[str, str]:
        if "." not in ref:
            raise BuildError(f"connection end {ref!r} must look like COMPONENT.port")
        cid, port = ref.rsplit(".", 1)
        if cid not in comps:
            raise BuildError(f"connection refers to unknown component {cid!r}")
        ct = get_type(comps[cid].type)
        if port not in ct.ports:
            raise BuildError(f"{cid} ({ct.type_name}) has no port {port!r}; ports: {', '.join(ct.ports)}")
        return cid, port

    # ---- API used by component build()
    def new_node(self, comp, suffix, fluid=None, volume=0.0, init=None, heat=(0.0, 293.15), wall=None,
                 kind=L.NODE_VOLUME) -> int:
        name = f"{comp.id}.{suffix}" if suffix else comp.id
        n = _Node(name, kind, canonical(fluid) if fluid else None, float(volume), init, comp.id,
                  (float(heat[0] or 0.0), float(heat[1])), wall)
        self.nodes.append(n)
        idx = len(self.nodes) - 1
        self.comp_nodes[comp.id].append(idx)
        return idx

    def new_boundary(self, comp, fluid, p, t, phase="auto") -> int:
        idx = self.new_node(comp, "", fluid=fluid, kind=L.NODE_BOUNDARY)
        self.nodes[idx].bp = p
        self.nodes[idx].bt = t
        self.nodes[idx].phase = phase
        return idx

    def new_pressurized_tank(self, comp, liquid, pressurant, vol, fill, p, t_l, t_g, area, heat, wall=None,
                             iface=(0.0, 0.0, 0.0), saturated=False):
        """An ullage node over a liquid node. ``saturated``: the ullage is the liquid's own
        vapor, and the liquid starts saturated at t_l (or at the saturation temperature
        for p); its pressure is fixed when the fluids are known (``_resolve_saturated``).

        The tank's wall and ambient heat sit on the liquid node and are shared with the
        ullage by wetted area.
        """
        g = self.new_node(comp, "ullage", fluid=pressurant, volume=vol,
                          init=("satvap", t_l) if saturated else ("pt", p, t_g))
        liq = self.new_node(comp, "liquid", fluid=liquid, volume=vol * fill, init=("tank_liquid", p, t_l, fill),
                            kind=L.NODE_TANK_LIQUID, heat=heat, wall=wall)
        self.nodes[g].partner = liq
        self.nodes[liq].partner = g
        self.nodes[liq].area = area
        self.nodes[liq].sat = saturated
        self.nodes[liq].iface = iface
        self.bind_port(comp, "top", g, L.DRAW_MIX)
        self.bind_port(comp, "bottom", liq, L.DRAW_LIQUID)
        return g, liq

    def bind_port(self, comp, port, node, draw) -> None:
        net = self.net_of[(comp.id, port)]
        if net in self.net_node and self.net_node[net] != node:
            other = self.nodes[self.net_node[net]].owner
            raise BuildError(
                f"{comp.id}.{port} and {other} are joined by a line of length 0, but both are volumes; "
                "give the line a length, or put a valve, orifice or fitting between them")
        self.net_node[net] = node
        self.net_draw[net] = draw

    def port_open(self, comp, port) -> bool:
        """Whether nothing is connected to this port."""
        net = self.net_of[(comp.id, port)]
        return not any(p in self.connected for p, n in self.net_of.items() if n == net)

    def across(self, comp, port) -> int | None:
        """The node at the far end of the (first) line on this port, if it has one yet."""
        net = self.net_of[(comp.id, port)]
        for pa, pb, _, _ in self.line_ends:
            for here, there in ((pa, pb), (pb, pa)):
                if self.net_of[here] == net and self.net_of[there] != net:
                    return self.net_node.get(self.net_of[there])
        return None

    def vent_port(self, comp, port) -> None:
        """Let an unconnected port vent to the atmosphere (the fluid it vents is whatever reaches it)."""
        net = self.net_of[(comp.id, port)]
        n = _Node(f"{comp.id}.{port}.atm", L.NODE_BOUNDARY, None, 0.0, None, f"{comp.id}.{port}")
        n.bp, n.bt = units.ATM, 293.15
        self.nodes.append(n)
        self.net_node[net] = len(self.nodes) - 1
        self.net_draw[net] = L.DRAW_MIX
        self.vented.add((comp.id, port))

    def port_node(self, comp, port) -> int:
        return self._net_node(self.net_of[(comp.id, port)])

    def _net_node(self, net) -> int:
        if net not in self.net_node:
            j = _Node(f"J{net}", L.NODE_VOLUME, None, 0.0, None, "")
            self.nodes.append(j)
            self.net_node[net] = len(self.nodes) - 1
            self.net_draw[net] = L.DRAW_MIX
        return self.net_node[net]

    def set_port_init(self, comp, port, init, fluid=None) -> None:
        """Give a junction the initial state (and fluid) of a line attached to it, if it has none."""
        n = self.nodes[self.port_node(comp, port)]
        if n.owner:
            return
        if n.init is None and init is not None and (init[1] is not None or init[2] is not None):
            n.init = init
        if fluid and n.fluid is None:
            n.fluid = canonical(fluid)

    def port_draw(self, comp, port) -> int:
        return self.net_draw.get(self.net_of[(comp.id, port)], L.DRAW_MIX)

    def add_port_volume(self, comp, port, vol) -> None:
        self.net_volume[self.net_of[(comp.id, port)]] += float(vol)
        self.port_node(comp, port)

    def add_orifice(self, comp, suffix, port_a, port_b, cda, ctrl=L.CTRL_FIXED, p1=0.0, p2=1.0, valve=None,
                    port_volume=None, reg=(0.0, 0.0, 0.0, 0.0)):
        for port in (port_a, port_b):
            if self.port_open(comp, port) and (comp.id, port) not in self.vented:
                self.warnings.append(f"{comp.id}.{port} is not connected to anything, so it is capped")
        a = self.port_node(comp, port_a)
        b = self.port_node(comp, port_b)
        # Valves, fittings and orifices have internal cavities on each side of the
        # restriction. Without them two flow elements in a row meet at a near-zero
        # volume, which a fixed real-time step cannot resolve. Default: a cavity
        # three equivalent diameters long on each side.
        if port_volume is None:
            area = cda / 0.6
            d_eq = (4.0 * area / 3.141592653589793) ** 0.5
            port_volume = area * 3.0 * d_eq
        for port in (port_a, port_b):
            net = self.net_of[(comp.id, port)]
            self.net_volume[net] += port_volume
        if a == b:
            raise BuildError(f"both ports of {comp.id} are connected to the same point")
        lk = _Link(comp.id + (f".{suffix}" if suffix else ""), a, b, comp.id, cda=cda, ctrl=ctrl,
                   p1=p1, p2=p2, draw_a=self.port_draw(comp, port_a), draw_b=self.port_draw(comp, port_b),
                   valve=valve, reg=reg,
                   nets=(self.net_of[(comp.id, port_a)], self.net_of[(comp.id, port_b)]))
        self.links.append(lk)
        idx = len(self.links) - 1
        self.comp_links[comp.id].append(idx)
        if ctrl == L.CTRL_VALVE:
            self.valves[comp.id] = idx

    def add_pump(self, comp, port, fluid, t, mdot, p_on, p_off, spin_up, volume) -> None:
        """A pump drawing ``fluid`` from the atmosphere at ``t`` into ``port``."""
        a = self.new_boundary(comp, fluid, 101325.0, t)
        self.nodes[a].name = f"{comp.id}.inlet"
        b = self.port_node(comp, port)
        self.net_volume[self.net_of[(comp.id, port)]] += volume
        rate = 1.0 / max(spin_up, 1e-6)
        lk = _Link(comp.id, a, b, comp.id, cda=mdot, ctrl=L.CTRL_PUMP, p1=p_on, p2=p_off,
                   valve=(0.0, rate, rate), nets=(-1, self.net_of[(comp.id, port)]))
        self.links.append(lk)
        idx = len(self.links) - 1
        self.comp_links[comp.id].append(idx)
        self.pumps[comp.id] = idx

    def add_actuator(self, comp, supply: int, volume, p_start, p_full, cda, normally_open, assumed=False) -> None:
        """A spring-return pneumatic actuator on valve ``comp``, filled from node ``supply``.

        Air it draws leaves the network through an exhaust boundary of the supply's fluid
        (the actuator itself is not a node: the kernel steps its pressure explicitly).
        """
        if supply not in self.exhausts:
            sn = self.nodes[supply]
            self.nodes.append(_Node(f"{sn.owner}.exhaust", L.NODE_BOUNDARY, None, 0.0, None, sn.owner,
                                    bp=101325.0, bt=293.15))
            self.exhausts[supply] = len(self.nodes) - 1
        lk = _Link(f"{comp.id}.actuator", supply, self.exhausts[supply], comp.id, cda=cda, ctrl=L.CTRL_AIR,
                   p1=p_start, p2=p_full, air=(self.valves[comp.id], float(volume), bool(normally_open)))
        self.links.append(lk)
        idx = len(self.links) - 1
        self.comp_links[comp.id].append(idx)
        self.actuators[comp.id] = idx
        self.air_stroke[supply] = min(self.air_stroke.get(supply, math.inf), p_start)
        if assumed:
            self.assumed_pneumatic[self.nodes[supply].owner].append(comp.id)

    def _bore(self, lk) -> float:
        """The bore of the line a flow element sits in: its own diameter when it has one (a
        fitting's is its line's), else twice the bore of its Cd*A: a valve or orifice
        restricts the line it is in."""
        if lk.owner:
            comp = self.system.component(lk.owner)
            ct = get_type(comp.type)
            if any(p.name == "diameter" for p in ct.params):
                d = ct.param(comp, "diameter")
                if d:
                    return float(d)
        return 2.0 * (4.0 * lk.cda / math.pi) ** 0.5

    def _line_size(self, pa, pb) -> float:
        """A line's inner diameter for line_diameter "auto": plumbing is sized to its valves and
        fittings, so the largest bore of those at either end (through tees), and never under
        1/4 in tubing."""
        ends = {self.net_of[pa], self.net_of[pb]}
        # a tee passes its lines' size on: the elements one line further on count too
        for qa, qb, _, _ in self.line_ends:
            na, nb = self.net_of[qa], self.net_of[qb]
            for here, there in ((na, nb), (nb, na)):
                if here in ends and self.nodes[self.net_node[here]].owner == "" and not any(
                        here in lk.nets for lk in self.links):
                    ends = ends | {there}
        bore = max((self._bore(lk) for lk in self.links if lk.ctrl not in (L.CTRL_LINE, L.CTRL_AIR, L.CTRL_PUMP)
                    and set(lk.nets) & ends), default=0.0)
        return max(bore, LINE_DIAMETER_MIN)

    def _add_lines(self) -> None:
        """A link for every line, carrying its own mass flow: the fluid in a line has inertia.

        The fluid the line holds is put in the junctions at its ends that are not tanks,
        volumes or boundaries (shared when both are): next to a valve, where a valve
        shutting on a moving column of liquid stops it against that volume. With the whole
        line's volume there, the surge is the Joukowsky rho*c*v.
        """
        dia_setting = self.system.setting("line_diameter")
        for pa, pb, ra, rb in self.line_ends:
            na, nb = self.net_of[pa], self.net_of[pb]
            if na == nb:
                continue  # two ports of one tee
            spec = self.system.line_spec(ra, rb)
            length = self._line_length(ra, rb)
            if spec.get("diameter") is not None:
                dia = units.parse(spec["diameter"], "length")
            elif str(dia_setting).strip().lower() == "auto":
                dia = self._line_size(pa, pb)
            else:
                dia = units.parse(dia_setting, "length")
            if not dia > 0.0:
                raise BuildError(f"the line {ra} -- {rb} needs a diameter above zero")
            area = math.pi * dia * dia / 4.0
            a, b = self._net_node(na), self._net_node(nb)
            junctions = [net for net, i in ((na, a), (nb, b)) if self.nodes[i].owner == ""]
            for net in junctions:
                self.net_volume[net] += area * length / len(junctions)
            lk = _Link(f"{ra} -- {rb}", a, b, "", cda=area / length, ctrl=L.CTRL_LINE,
                       draw_a=self.net_draw.get(na, L.DRAW_MIX), draw_b=self.net_draw.get(nb, L.DRAW_MIX),
                       nets=(na, nb), area=area)
            self.links.append(lk)

    # ---- build
    def build(self) -> Network:
        sysm = self.system
        ids = [c.id for c in sysm.components]
        dup = {i for i in ids if ids.count(i) > 1}
        if dup:
            raise BuildError(f"component ids must be unique: {', '.join(sorted(dup))}")
        self._group_nets()
        # volumes first so their ports own their nets, then everything else
        order = sorted(sysm.components, key=lambda c: 0 if get_type(c.type).category == "Volumes" else 1)
        for c in order:
            ct = get_type(c.type)
            try:
                ct.build(self, c)
            except BuildError:
                raise
            except (ValueError, KeyError, TypeError) as e:
                raise BuildError(f"{c.id}: {e}") from None
        self._add_lines()
        for supply, valves in self.assumed_pneumatic.items():
            self.warnings.append(f"{supply} runs {', '.join(valves)} as pneumatic valves; set actuator: solenoid "
                                 "on any that are direct-acting")
        # every net that is used by a link but owns no volume is a junction
        for net, node in self.net_node.items():
            n = self.nodes[node]
            if n.owner == "":
                n.volume = max(self.net_volume.get(net, 0.0), self.junction_volume)
        self._assign_fluids()
        fs = self._fluid_set()
        self._resolve_saturated(fs)
        self._find_pressed(fs, *self._spread)
        if self.pressed:
            return self._rebuild_pressed(fs)
        for n in self.nodes:
            if n.sat and fs.models[fs.names.index(n.fluid)] != "table":
                raise BuildError(f"{n.owner}: {n.fluid} is modelled with no vapor of its own, so a fill needs a gas "
                                 "above it: set a pressurant (such as air or nitrogen) and a pressure")
        self._resolve_tank_liquids(fs)
        if self.system.setting("prime_lines"):
            self._prime_lines(fs)
        for i, p_start in self.air_stroke.items():
            n = self.nodes[i]
            if n.kind == L.NODE_BOUNDARY:
                p0 = n.bp
            else:
                p0 = n.init[1] if n.init and n.init[0] == "pt" and n.init[1] is not None else self.p_init
            if p0 - units.ATM < p_start:
                self.warnings.append(
                    f"{n.owner} starts at {units.to_psi(p0 - units.ATM):.0f} psig, below the "
                    f"{units.to_psi(p_start):.0f} psig its valves need to move: none of them will stroke until "
                    "it is charged (give it a pressure, or feed it from a compressor or regulator)")
        return self._compile(fs)

    def _rebuild_pressed(self, fs: FluidSet) -> Network:
        """Build again with each pressed or filled single-fluid tank given an ullage.

        A tank without a pressurant holds one fluid, so another fluid pushed into it
        would be counted as that fluid. A tank of liquid (no fill) that a gas can press
        gets the gas as its pressurant over a 1 % ullage, at its pressure and
        temperature. An "empty" tank of air or nitrogen that a liquid can be filled
        into becomes that liquid, none of it yet, under the gas.
        """
        sysm = copy.deepcopy(self.system)
        notes = []
        for c in sysm.components:
            if c.id not in self.pressed:
                continue
            other, kind, t_liq = self.pressed[c.id]
            ct = get_type(c.type)
            own = canonical(ct.param(c, "fluid"))
            if kind == "empty":
                t_g = ct.param(c, "temperature")
                c.params.update(fluid=other, pressurant=own, fill=0.0, gas_temperature=t_g)
                c.params.pop("temperature", None)
                if t_liq is not None and not math.isnan(t_liq):
                    c.params["temperature"] = float(t_liq)
                notes.append(f"{c.id}: {other} can be filled into its {own}, so it is modelled as {other} collecting "
                             f"under the {own}, which it compresses unless the tank is vented")
                continue
            if kind == "fill":
                c.params.update(pressurant=other, pressure=float(t_liq))
                notes.append(f"{c.id}: {other} can be pushed into it, so its {own} is under a {other} ullage "
                             f"(set 'pressurant: {other}' to choose its starting state)")
                continue
            f = fs.names.index(own)
            t, p = ct.param(c, "temperature"), ct.param(c, "pressure")
            t_g = t
            if other == own:
                t_g = max(t, fs.saturation_temperature(f, p)) + 3.0  # its own vapor, just above boiling
            c.params.update(pressurant=other, pressure=float(p), temperature=float(t), gas_temperature=float(t_g),
                            fill=0.99)
            notes.append(f"{c.id}: {other} can be pushed into its {own}, so it is modelled with a {other} ullage "
                         f"over the liquid, from a 1 % ullage (set 'pressurant: {other}' and a fill to choose)")
        b = NetworkBuilder(sysm, pressed={**self._already_pressed, **self.pressed})
        b.warnings[:0] = notes
        return b.build()

    def _fluid_set(self) -> FluidSet:
        sysm = self.system
        models_cfg = {canonical(k): v for k, v in (sysm.setting("fluid_models") or {}).items()}
        used = sorted({n.fluid for n in self.nodes} | {n.gas for n in self.nodes if n.gas})
        models = []
        for f in used:
            m = models_cfg.get(f, FLUIDS[f][1])
            if m not in FLUIDS[f][2]:
                raise BuildError(f"{f} cannot use the {m} model; allowed: {', '.join(FLUIDS[f][2])}")
            models.append(m)
        return FluidSet(used, models, backend=sysm.setting("backend"))

    def _resolve_saturated(self, fs: FluidSet) -> None:
        """Each saturated tank's starting temperature and pressure, on its saturation line."""
        for n in self.nodes:
            if not n.sat:
                continue
            _, p, t, fill = n.init
            f = fs.names.index(n.fluid)
            if fs.models[f] != "table":
                continue  # no vapor of its own: it needs a gas to press it (``_find_pressed``)
            if t is None:
                if p is None:
                    raise BuildError(f"{n.owner} needs temperature or pressure with its fill")
                t = fs.saturation_temperature(f, p)
            tc = fs.critical_temperature(f)
            if t >= tc - 0.5:
                raise BuildError(f"{n.owner}: {n.fluid} has no liquid above {tc:.1f} K; lower its temperature")
            n.init = ("tank_liquid", fs.saturation_pressure(f, t), t, fill)
            self.nodes[n.partner].init = ("satvap", t)

    def _resolve_tank_liquids(self, fs: FluidSet) -> None:
        """Fix each tank liquid's starting temperature, and check it is a liquid there.

        Without a ``temperature`` the liquid starts at the system's initial
        temperature, or at its boiling point at the tank pressure if that is colder
        (a cryogen just loaded and pressurized).
        """
        for n in self.nodes:
            if not n.init or n.init[0] != "tank_liquid" or n.sat:
                continue
            _, p, t, fill = n.init
            f = fs.names.index(n.fluid)
            if fs.models[f] != "table":
                n.init = ("tank_liquid", p, self.t_init if t is None else t, fill)
                continue
            tc, t_boil = fs.critical_temperature(f), None
            if p < fs.critical_pressure(f):
                t_boil = fs.saturation_temperature(f, p)
            t_max = min(tc - 1.0, t_boil - 0.5 if t_boil is not None else tc)
            if t is None:
                t = self.t_init
                if t > t_max:
                    t = t_max
                    self.warnings.append(f"{n.owner}: {n.fluid} starts at {t:.1f} K, just below its boiling point "
                                         "at the tank pressure; set its temperature to choose")
            elif t_max < t < tc and (t_boil is None or t <= t_boil):
                self.warnings.append(f"{n.owner}: {n.fluid} at {t:.1f} K is too close to boiling; "
                                     f"starting it at {t_max:.1f} K")
                t = t_max
            if t > t_max:
                limit = f"boils at {t_boil:.1f} K at this pressure" if t_boil is not None else \
                    f"has no liquid above {tc:.1f} K"
                raise BuildError(f"{n.owner}: {n.fluid} at {t:.1f} K is not a liquid ({n.fluid} {limit}); "
                                 "lower its temperature or raise the pressure")
            n.init = ("tank_liquid", p, t, fill)

    def _assign_fluids(self) -> None:
        """Spread fluids from the components that set one along the links.

        Flow passes check valves, relief valves, regulators and pumps forwards only.
        A node reached by one fluid carries it. A node reached by a purge gas (see
        ``PURGE_GASES``) and one other fluid carries both: the other fluid is its own and
        the purge gas mixes into it, so a purge can push propellant out of a line. Any
        other meeting of fluids is an error. Volumes that set their fluid hold only it;
        another fluid flowing into one is counted as the volume's fluid (with a warning).
        """
        adj = defaultdict(list)
        for lk in self.links:
            adj[lk.a].append(lk.b)
            adj[lk.b].append(lk.a)
        src: dict[int, str] = {}
        for i, n in enumerate(self.nodes):
            if n.fluid:
                src[i] = n.fluid
        fwd = defaultdict(list)
        one_way = set()
        for k, lk in enumerate(self.links):
            fwd[lk.a].append(lk.b)
            if lk.ctrl in (L.CTRL_CHECK, L.CTRL_RELIEF, L.CTRL_REGULATOR, L.CTRL_PUMP, L.CTRL_AIR):
                one_way.add(k)
            else:
                fwd[lk.b].append(lk.a)

        def spread(graph):
            # node -> fluid -> (links away, the node that sets it)
            out: dict[int, dict[str, tuple[int, int]]] = defaultdict(dict)
            for start, fl in src.items():
                q = deque([(start, 0)])
                seen = {start}
                while q:
                    i, d = q.popleft()
                    for j in graph[i]:
                        if j in seen or j in src:
                            continue
                        seen.add(j)
                        if fl not in out[j] or out[j][fl][0] > d + 1:
                            out[j][fl] = (d + 1, start)
                        q.append((j, d + 1))
            return out

        reached = spread(adj)
        flows = spread(fwd)
        purge_list = ", ".join(PURGE_GASES)
        for i, n in enumerate(self.nodes):
            if n.fluid is not None:
                continue
            # what can flow there; a dead end only flow leaves still takes its neighbours' fluid
            fls = flows.get(i) or reached.get(i)
            if not fls:
                raise BuildError(f"{n.name} is not connected to anything that sets its fluid")
            if len(fls) == 1:
                n.fluid = next(iter(fls))
                continue
            gas = next((g for g in PURGE_GASES if g in fls), None)
            if len(fls) > 2 or gas is None:
                what = " and ".join(f"{f} from {self.nodes[src_i].owner}" for f, (_d, src_i) in sorted(fls.items()))
                raise BuildError(
                    f"{n.name} is reached by {what}; a line may carry one fluid plus one purge gas "
                    f"({purge_list}). Set 'fluid' on a component or close off the lines to separate them")
            n.fluid = next(f for f in fls if f != gas)
            n.gas = gas
            # it starts holding whichever can flow there from nearer (the gas on a tie)
            n.gas_fill = fls[gas][0] <= fls[n.fluid][0]
        self._spread = (src, fwd, flows)
        warned = set()
        for k, lk in enumerate(self.links):
            na, nb = self.nodes[lk.a], self.nodes[lk.b]
            sa = {na.fluid} | ({na.gas} if na.gas else set())
            sb = {nb.fluid} | ({nb.gas} if nb.gas else set())
            others = (sa | sb) - set(PURGE_GASES)
            if len(others) > 1:
                what = lk.owner or f"the line {lk.name}"
                raise BuildError(f"{what} connects {na.fluid} to {nb.fluid}; lines may carry one fluid "
                                 f"plus one purge gas ({purge_list})")
            for n, s_self, s_other in ((na, sa, sb), (nb, sb, sa)):
                if n is na and k in one_way:
                    continue  # nothing flows back into its inlet
                extra = s_other - s_self
                if n.gas is None and extra and n.owner and (n.owner, n.fluid) not in warned:
                    warned.add((n.owner, n.fluid))
                    what = " or ".join(sorted(extra))
                    if n.kind == L.NODE_TANK_LIQUID:
                        gas = self.nodes[n.partner].fluid
                        self.warnings.append(f"{n.owner} holds only {n.fluid} and {gas}: any {what} flowing into "
                                             f"it rises into its ullage and is counted as {gas}")
                    else:
                        self.warnings.append(f"{n.owner} holds only {n.fluid}: any {what} flowing into it "
                                             f"is counted as {n.fluid}")

    def _find_pressed(self, fs: FluidSet, src, fwd, flows) -> None:
        """Tanks without a pressurant that a gas can be pushed into from a higher pressure.

        A tank of liquid (saturated, or liquid at its pressure and temperature) is
        rebuilt with that gas as its pressurant (see ``_rebuild_pressed``). A tank of
        gas that a different gas can reach holds both, mixed, as a purged line does.

        The gas must come from a gas source at a higher pressure than the tank starts
        at: shop air purging a fill line cannot get into a nitrous bottle at 750 psi,
        air that could back into a LOX tank through its vent is no reason to give it
        an ullage, and liquid from a dewar filling a run tank is not a press.
        """
        types = {c.id: c.type for c in self.system.components}

        def liquid_at(f, p, t):
            if fs.models[f] == "liquid":
                return True
            if fs.models[f] != "table" or t >= fs.critical_temperature(f):
                return False
            return p > fs.saturation_pressure(f, t)

        def gas_source_p(j, g):
            n = self.nodes[j]
            if n.kind == L.NODE_BOUNDARY:
                p, t = n.bp, n.bt
            elif n.init and n.init[0] == "pt":
                p = self.p_init if n.init[1] is None else n.init[1]
                t = self.t_init if n.init[2] is None else n.init[2]
            else:
                return 0.0  # a saturated source, such as an LN2 dewar: what it sends may be liquid
            return 0.0 if liquid_at(fs.names.index(g), p, t) else p

        def liquid_source_t(j, g):
            """The temperature of the liquid node j can send (nan if not known), or None if it sends none."""
            n = self.nodes[j]
            f = fs.names.index(g)
            if n.kind == L.NODE_BOUNDARY:
                return n.bt if liquid_at(f, n.bp, n.bt) else None
            if not n.init:
                return None
            if n.init[0] == "tank_liquid":
                return math.nan if n.init[2] is None else n.init[2]
            if n.init[0] == "saturated":
                t = n.init[1]
                return fs.saturation_temperature(f, n.init[2]) if t is None else t
            if n.init[0] == "pt":
                p = self.p_init if n.init[1] is None else n.init[1]
                t = self.t_init if n.init[2] is None else n.init[2]
                return t if liquid_at(f, p, t) else None
            return None

        into = defaultdict(list)
        for j, outs in fwd.items():
            for i in outs:
                into[i].append(j)

        def sources(i):
            """What can flow into node i, and from where: (fluid, source node) for every node
            that sets its fluid and reaches i along the way flow can go (not through i)."""
            out, seen, todo = [], {i}, list(into[i])
            while todo:
                j = todo.pop()
                if j in seen:
                    continue
                seen.add(j)
                if j in src:
                    out.append((src[j], j))
                else:
                    todo.extend(into[j])
            return out
        for i, n in enumerate(self.nodes):
            if (n.kind != L.NODE_VOLUME or not n.init or n.init[0] not in ("satvap", "pt") or n.gas
                    or types.get(n.owner) != "tank"):
                continue
            if n.owner in self._already_pressed and n.partner < 0:
                continue
            f = fs.names.index(n.fluid)
            if n.init[0] == "satvap" and fs.models[f] != "table":
                # a fill of water or ethanol with no gas over it: a gas that can press it becomes its pressurant
                p = self.nodes[n.partner].init[1]
                p = units.ATM if p is None else p
                kind = "fill"
            elif n.init[0] == "satvap":  # a saturated tank's vapor
                p = fs.saturation_pressure(f, n.init[1])
                kind = "gas"
            else:
                p = self.p_init if n.init[1] is None else n.init[1]
                t = self.t_init if n.init[2] is None else n.init[2]
                kind = "liquid" if liquid_at(f, p, t) else "gas"
            came = sources(i)
            gases = {g for g, s in came if g in PURGE_GASES and (g != n.fluid or kind != "gas")
                     and gas_source_p(s, g) > p + 0.5 * units.ATM}
            if kind == "gas" and n.partner < 0 and n.fluid in PURGE_GASES:
                # an "empty" tank of air or nitrogen that a liquid can be filled into: the
                # liquid collects at the bottom under the gas, which it compresses
                liquids = {}
                for g, s in came:
                    t_l = liquid_source_t(s, g)
                    if g not in PURGE_GASES and t_l is not None:
                        liquids[g] = t_l
                if liquids:
                    liq = sorted(liquids)[0]
                    self.pressed[n.owner] = (liq, "empty", liquids[liq])
                    continue
            if not gases:
                continue
            gas = sorted(gases, key=PURGE_GASES.index)[0]
            if kind == "fill":
                self.pressed[n.owner] = (gas, kind, p)
            elif kind != "gas":
                if n.partner < 0:
                    self.pressed[n.owner] = (gas, kind, None)
            elif gas != n.fluid:
                # a gas tank, or the ullage of one with a pressurant, that another gas
                # reaches holds both, as a purged line does
                n.gas = gas
                where = "its ullage holds" if n.partner >= 0 else "it holds"
                self.warnings = [w for w in self.warnings if not w.startswith(f"{n.owner} holds only")]
                self.warnings.append(f"{n.owner}: {gas} can be pushed into its {n.fluid}, so {where} a mix of both")

    def _prime_lines(self, fs: FluidSet) -> None:
        """Start lines in equilibrium with the tank or volume they are open to.

        Every node without its own initial state takes the state of the nearest
        volume reachable through fittings, orifices and open valves; a line
        on a tank's liquid port starts full of that liquid, as it is after loading.
        Lines behind closed valves, check valves and regulators keep the system's
        initial pressure and temperature.
        """
        def explicit(n):
            if n.kind == L.NODE_BOUNDARY:
                return True
            i = n.init
            return i is not None and (i[0] != "pt" or i[1] is not None or i[2] is not None)

        def source_state(i, draw):
            n = self.nodes[i]
            if n.kind == L.NODE_BOUNDARY:
                f = fs.names.index(n.fluid)
                if n.phase in ("liquid", "vapor") and fs.models[f] == "table" and n.bt < fs.critical_temperature(f):
                    psat = fs.saturation_pressure(f, n.bt)
                    if abs(n.bp - psat) <= 1e-6 * psat:  # on its saturation line: the phase it supplies
                        return ("saturated", n.bt, None, 1.0 if n.phase == "liquid" else 0.0)
                return ("pt", n.bp, n.bt)
            if n.init[0] == "tank_liquid":
                if n.sat:  # saturated liquid: a line on it starts full of that liquid
                    return ("saturated", n.init[2], None, 1.0)
                return ("pt", n.init[1], n.init[2])
            if n.init[0] == "saturated" and draw != L.DRAW_MIX:
                return ("saturated", n.init[1], n.init[2], 1.0 if draw == L.DRAW_LIQUID else 0.0)
            return n.init

        adj = defaultdict(list)
        for lk in self.links:
            if lk.ctrl == L.CTRL_REGULATOR:
                # a regulated line sits locked up at its setpoint (gas sources only)
                adj[lk.a].append((lk.b, lk.draw_a, lk.p1))
                continue
            if lk.ctrl not in (L.CTRL_FIXED, L.CTRL_LINE):
                if lk.ctrl != L.CTRL_VALVE or not lk.valve or lk.valve[0] <= 0.0:
                    continue
            adj[lk.a].append((lk.b, lk.draw_a, None))
            adj[lk.b].append((lk.a, lk.draw_b, None))
        def content(n):
            return n.gas if n.gas and n.gas_fill else n.fluid

        state: dict[int, tuple] = {}
        origin: dict[int, str] = {}
        q = deque()
        for i, n in enumerate(self.nodes):
            if explicit(n):
                q.append((i, None, content(n)))
        while q:
            i, st, fl = q.popleft()
            for j, draw, setpoint in adj[i]:
                if j in state or explicit(self.nodes[j]):
                    continue
                new = st if st is not None else source_state(i, draw)
                if setpoint is not None:
                    if new[0] != "pt" or new[1] is None:
                        continue
                    new = ("pt", min(new[1], setpoint), new[2])
                state[j] = new
                origin[j] = fl
                q.append((j, new, fl))
        for j, st in state.items():
            n = self.nodes[j]
            n.init = st
            n.primed = True
            if n.gas:
                # a mixed line starts full of whatever it was primed from
                n.gas_fill = origin[j] == n.gas

    def _compile(self, fs: FluidSet) -> Network:
        sysm = self.system
        used = fs.names

        nn, nl = len(self.nodes), len(self.links)
        NI = np.zeros((nn, L.N_NI), dtype=np.int64)
        NF = np.zeros((nn, L.N_NF))
        x0: list[float] = []
        xscale: list[float] = []
        # mixed nodes: the state their first solves start from (see kernel._mixed_one)
        mix_init = np.zeros((nn, 4))
        rho0 = np.zeros(nn)  # starting density of each node, to scale the lines' flows by
        for i, n in enumerate(self.nodes):
            f = used.index(n.fluid)
            NI[i, L.N_KIND] = n.kind
            NI[i, L.N_FLUID] = f
            NI[i, L.N_PARTNER] = n.partner
            NI[i, L.N_GAS] = used.index(n.gas) if n.gas else -1
            NI[i, L.N_GX] = -1
            NF[i, L.NF_VOL] = n.volume
            NF[i, L.NF_HA] = n.heat[0]
            NF[i, L.NF_TAMB] = n.heat[1]
            if n.kind == L.NODE_BOUNDARY:
                NI[i, L.N_SX] = -1
                rho, u = self._boundary_state(fs, f, n)
                NF[i, L.NF_BP], NF[i, L.NF_BT], NF[i, L.NF_BRHO], NF[i, L.NF_BU] = n.bp, n.bt, rho, u
                rho0[i] = rho
                continue
            if n.gas and n.gas_fill:
                # starts full of purge gas, with a trace of the line's own fluid at the same state
                init = n.init if n.init and n.init[0] == "pt" else ("pt", None, None)
                p0 = self.p_init if init[1] is None else init[1]
                t0 = self.t_init if init[2] is None else init[2]
                rho, u = fs.rho_u_from_pt(f, p0, t0)
            else:
                rho, u, t0 = self._initial_state(fs, f, n)
                if n.gas:
                    p0 = float(_state(f, rho, u, *fs.arrays()[:7])[0])
            if n.kind == L.NODE_TANK_LIQUID:
                rhoref = fs.liquid_density(f, t0)
                NF[i, L.NF_RHOREF] = rhoref
                NF[i, L.NF_AREA] = n.area
                NF[i, L.NF_IFC], NF[i, L.NF_IFB], NF[i, L.NF_IFS] = n.iface
                tank_vol = self.nodes[n.partner].volume
                NF[i, L.NF_MREF] = rhoref * 0.002 * tank_vol
                m = rhoref * n.volume
                rho = rhoref
            else:
                vol = n.volume
                if n.partner >= 0:  # ullage shares the tank with the liquid
                    vol = n.volume - self.nodes[n.partner].volume
                    if fs.model[f] == MODEL_TABLE and rho > fs.liquid_density(f, t0) * 0.5:
                        raise BuildError(
                            f"{n.owner}: the {n.fluid} pressurant would be liquid at its initial pressure and "
                            f"{t0:.1f} K; set a warmer gas_temperature")
                m = rho * vol
                # below this mass (about 1 kPa of gas) outflows are throttled, so a
                # volume pumped down to vacuum cannot be driven to negative mass
                NF[i, L.NF_MREF] = max(vol, 1e-9) * 0.01
            m_full = m
            rho0[i] = rho
            e = m * u
            if n.gas:
                g = used.index(n.gas)
                m, e, m_g, mix_init[i] = _mixed_init(fs, f, g, vol, p0, t0, rho, u, n.gas_fill)
                gas_scale = max(max(p0, 1e5) * n.volume / (fs.par[g, I_R] * t0), 1e-9)
            NI[i, L.N_SX] = len(x0)
            x0 += [m, e]
            # mixed nodes scale by the mass that would fill the node, so a trace is not
            # held to a relative accuracy
            xscale += [max(m_full, 1e-9), max(abs(m_full * u), m_full * 1e5, 1e-3)]
            if n.wall:
                mc, ha_in, ha_out, t_amb, at_fluid, *curve = n.wall
                NI[i, L.N_WALL] = curve[0] if curve else L.WALL_CONSTANT
                NF[i, L.NF_WALLMC] = mc
                NF[i, L.NF_HA] = ha_in
                NF[i, L.NF_HAAMB] = ha_out
                NF[i, L.NF_TAMB] = t_amb
                # a volume primed from a tank has been sitting full, and a tank shell holds its
                # contents: their walls are at the fluid's temperature
                x0.append(t0 if n.primed or at_fluid else t_amb)
                xscale.append(100.0)
            if n.gas:
                NI[i, L.N_GX] = len(x0)
                x0.append(m_g)
                xscale.append(gas_scale)
        # ullage nodes need the liquid reference density of their partner
        LI = np.zeros((nl, L.N_LI), dtype=np.int64)
        LF = np.zeros((nl, L.N_LF))
        LV = np.zeros((nl, L.N_V))
        for k, lk in enumerate(self.links):
            LI[k] = (lk.a, lk.b, lk.draw_a, lk.draw_b, lk.ctrl, -1)
            if lk.ctrl == L.CTRL_LINE:
                # its flow starts at rest; scaled by the flow at 1 m/s
                LI[k, L.L_SX] = len(x0)
                x0.append(0.0)
                xscale.append(max(0.5 * (rho0[lk.a] + rho0[lk.b]) * lk.area, 1e-9))
            LF[k, :L.LF_PREF] = (lk.cda, lk.p1, lk.p2, *lk.reg[:3])
            LF[k, L.LF_RESET] = lk.reg[3]
            LV[k, L.V_AIR] = -1.0
            if lk.valve:
                start, r_open, r_close = lk.valve
                LV[k, :4] = (start, start, r_open, r_close)
                LV[k, L.V_TGT] = start
            else:
                LV[k, :4] = (1.0, 1.0, 0.0, 0.0)
        for k, lk in enumerate(self.links):
            if lk.air:
                m, vol, no = lk.air
                LV[k, L.V_VALVE], LV[k, L.V_PACT], LV[k, L.V_VOL], LV[k, L.V_NO] = m, 101325.0, vol, float(no)
                LV[m, L.V_AIR] = k

        port_nodes = {f"{cid}.{p}": self.net_node[net] for (cid, p), net in self.net_of.items()
                      if net in self.net_node}
        net = Network(sysm, fs, units.parse(sysm.setting("dt"), "time"), NI, NF, LI, LF, LV,
                      np.array(x0, dtype=float), np.array(xscale, dtype=float),
                      [n.name for n in self.nodes], [lk.name for lk in self.links],
                      [n.owner for n in self.nodes], [lk.owner for lk in self.links],
                      port_nodes, dict(self.comp_nodes), dict(self.comp_links), dict(self.valves),
                      warnings=self.warnings, mix_init=mix_init, pumps=dict(self.pumps),
                      actuators=dict(self.actuators), port_flows=self._port_flows(),
                      line_flows=self._line_flows())
        _structure(net)
        return net

    def _boundary_state(self, fs: FluidSet, f: int, n: _Node) -> tuple[float, float]:
        """A boundary's density and energy; on its saturation line, the phase it asks for."""
        if fs.model[f] == MODEL_TABLE and n.bt < fs.critical_temperature(f):
            psat = fs.saturation_pressure(f, n.bt)
            if abs(n.bp - psat) <= 1e-6 * psat:
                if n.phase == "auto":
                    self.warnings.append(
                        f"{n.owner} sits exactly at {n.fluid}'s vapor pressure at {n.bt:.2f} K, so it supplies "
                        "saturated vapor; set phase: liquid for saturated liquid")
                return fs.rho_u_saturated(f, n.bt, 1.0 if n.phase == "liquid" else 0.0)
        if n.phase in ("liquid", "vapor") and fs.model[f] == MODEL_TABLE and n.bt < fs.critical_temperature(f):
            psat = fs.saturation_pressure(f, n.bt)
            if (n.phase == "liquid") != (n.bp > psat):
                raise BuildError(f"{n.owner}: {n.fluid} at {n.bp / 6894.757:.1f} psia and {n.bt:.2f} K is "
                                 f"{'vapor' if n.bp < psat else 'liquid'}, not {n.phase}")
        return fs.rho_u_from_pt(f, n.bp, n.bt)

    def _lines_at(self, net) -> list[tuple[int, float]]:
        """The lines on a point, each with the sign that makes its flow positive out of that point."""
        return [(k, 1.0 if lk.nets[0] == net else -1.0) for k, lk in enumerate(self.links)
                if lk.ctrl == L.CTRL_LINE and net in lk.nets]

    def _port_flows(self) -> dict[str, list[tuple[int, float]]]:
        """How to read the flow along the line at each port, for mass flow sensors on lines.

        At a flow element's port (valve, orifice, regulator, fitting, pump) it is the element's
        own flow, positive in to out. At a tank's or volume's port it is the flow leaving the
        part there: the lines on that port, outward positive. A tee's ports are all one point,
        so no flow belongs to one of them.
        """
        comps = {c.id: get_type(c.type) for c in self.system.components}
        on_net: dict[int, list[int]] = defaultdict(list)  # flow elements joined straight on (length 0)
        for k, lk in enumerate(self.links):
            if lk.ctrl != L.CTRL_LINE:
                for net in set(lk.nets) - {-1}:
                    on_net[net].append(k)
        out = {}
        for (cid, port), net in self.net_of.items():
            own = [k for k in self.comp_links.get(cid, []) if net in self.links[k].nets]
            if own:
                out[f"{cid}.{port}"] = [(own[0], 1.0)]
            elif not comps[cid].merges_ports and self._lines_at(net):
                out[f"{cid}.{port}"] = self._lines_at(net)
            elif not comps[cid].merges_ports and on_net.get(net):
                out[f"{cid}.{port}"] = [(k, 1.0 if self.links[k].nets[0] == net else -1.0) for k in on_net[net]]
        return out

    def _line_flows(self) -> dict[tuple[str, str], list[tuple[int, float]] | None]:
        """How to read the flow along each line from a to b: its own flow (see core/lines.py)."""
        own = {lk.name: k for k, lk in enumerate(self.links) if lk.ctrl == L.CTRL_LINE}
        ports = self._port_flows()
        out = {}
        for a, b in self.system.connections:
            k = own.get(f"{a} -- {b}")
            if k is not None:
                out[(a, b)] = [(k, 1.0)]
                continue
            # a line of length 0 is no line of its own: what the flow element at one end passes
            out[(a, b)] = None
            for ref, sign in ((a, -1.0), (b, 1.0)):
                cid, port = ref.rsplit(".", 1)
                terms = ports.get(ref)
                if terms and any(self.links[t].owner == cid for t, _ in terms):
                    first = port == get_type(self.system.component(cid).type).ports[0]
                    # positive from a to b: into the element's inlet at b, out of its outlet at a
                    out[(a, b)] = [(t, g * (sign if first else -sign)) for t, g in terms]
                    break
        return out

    def _initial_state(self, fs: FluidSet, f: int, n: _Node) -> tuple[float, float, float]:
        init = n.init or ("pt", None, None)
        if init[0] == "saturated":
            _, t, p, fill = init
            if t is None:
                if p is None:
                    raise BuildError(f"{n.owner} needs temperature or pressure with its fill")
                t = fs.saturation_temperature(f, p)
            if fs.models[f] != "table":
                raise BuildError(f"{n.owner}: {n.fluid} is modelled with no vapor of its own, so a fill needs a gas "
                                 "above it: set a pressurant (such as air or nitrogen) and a pressure")
            rho, u = fs.rho_u_saturated(f, t, fill)
            return rho, u, t
        if init[0] == "satvap":
            rho, u = fs.rho_u_saturated(f, init[1], 0.0)
            return rho, u, init[1]
        if init[0] == "tank_liquid":
            _, p, t, _fill = init
            rho, u = fs.rho_u_from_pt(f, p, t)
            if fs.models[f] == "table":
                # the tank liquid is held at the saturated-liquid density for its temperature,
                # so its energy must be the saturated-liquid energy at that temperature too;
                # otherwise (rho, u) lands inside the dome at a lower temperature
                rho, u = fs.rho_u_saturated(f, t, 1.0)
            return rho, u, t
        _, p, t = init
        p = self.p_init if p is None else p
        t = self.t_init if t is None else t
        rho, u = fs.rho_u_from_pt(f, p, t)
        return rho, u, t


def _mixed_init(fs: FluidSet, f: int, g: int, vol: float, p0: float, t0: float, rho: float, u: float,
                gas_fill: bool) -> tuple[float, float, float, tuple]:
    """Starting masses and energy of a node holding fluid f and purge gas g at p0, t0.

    It starts full of whichever it was primed with and holds none of the other, in
    the kernel's own terms (``kernel._mixed_one``), so it starts at p0.
    Returns (fluid mass, energy, gas mass, (fluid energy, log gas partial pressure,
    temperature, 0)) the kernel's first solve starts from.
    """
    par = fs.par
    r_g, u_g = par[g, I_R], par[g, I_CV] * t0 + par[g, P_OFF]
    frac = 0.0 if gas_fill else 1.0
    if fs.model[f] != MODEL_LIQUID:
        # the fluid spread over the node, the gas's partial pressure on top
        m_f = rho * vol * frac
        m_g = p0 * (1.0 - frac) * vol / (r_g * t0)
        return m_f, m_f * u + m_g * u_g, m_g, (u, 0.0, t0, 0.0)
    # a liquid compressed by the total pressure, the gas in the space it leaves
    y = vol * (1.0 - frac)
    p_g = max(p0 - par[f, L_PSAT], GAS_TRACE * p0)
    rho_liq = par[f, L_RHO0] * (1.0 + (par[f, L_PSAT] + p_g - par[f, L_PREF]) / par[f, L_K])
    m_f = rho_liq * (vol - y)
    m_g = p_g * y / (r_g * t0)
    u_f = par[f, L_U0] + par[f, L_CV] * (t0 - par[f, L_T0])
    return m_f, m_f * u_f + m_g * u_g, m_g, (u_f, math.log(p_g), t0, 0.0)


def _structure(net: Network) -> None:
    """Jacobian sparsity (CSC), column colouring and a band-narrowing order."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    NI, LI = net.NI, net.LI
    n = net.n_states

    def node_states(i):
        if NI[i, L.N_KIND] == L.NODE_BOUNDARY:
            return []
        sx = NI[i, L.N_SX]
        out = list(range(sx, sx + (3 if NI[i, L.N_WALL] else 2)))
        if NI[i, L.N_GX] >= 0:
            out.append(int(NI[i, L.N_GX]))
        return out

    def group(i):
        g = [i]
        if NI[i, L.N_PARTNER] >= 0:
            g.append(int(NI[i, L.N_PARTNER]))
        return g

    entries = set()

    def mark(rows, cols):
        for r in rows:
            for c in cols:
                entries.add((r, c))

    for i in range(NI.shape[0]):
        s = [st for g in group(i) for st in node_states(g)]
        mark(s, s)
    for k in range(LI.shape[0]):
        a, b = int(LI[k, L.L_A]), int(LI[k, L.L_B])
        s = [st for g in group(a) + group(b) for st in node_states(g)]
        if LI[k, L.L_SX] >= 0:  # a line's flow moves its ends, and their pressures drive it
            s.append(int(LI[k, L.L_SX]))
        mark(s, s)
    for i in range(n):
        entries.add((i, i))
    rows = np.array([e[0] for e in entries])
    cols = np.array([e[1] for e in entries])
    pat = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsc()
    pat.sort_indices()
    net.col_ptr = pat.indptr.astype(np.int64)
    net.col_rows = pat.indices.astype(np.int64)

    # greedy colouring: columns sharing no row can be perturbed together
    row_sets = [set(pat.indices[pat.indptr[j]:pat.indptr[j + 1]]) for j in range(n)]
    colour_rows: list[set] = []
    colour_cols: list[list[int]] = []
    for j in range(n):
        for c, rs in enumerate(colour_rows):
            if not (rs & row_sets[j]):
                rs |= row_sets[j]
                colour_cols[c].append(j)
                break
        else:
            colour_rows.append(set(row_sets[j]))
            colour_cols.append([j])
    net.color_ptr = np.cumsum([0] + [len(c) for c in colour_cols]).astype(np.int64)
    net.color_cols = np.array([j for c in colour_cols for j in c], dtype=np.int64)

    sym = (pat + pat.T).tocsr()
    # (a system of boundaries and links only has no states: nothing to order)
    perm = reverse_cuthill_mckee(sym, symmetric_mode=True).astype(np.int64) if n else np.zeros(0, np.int64)
    iperm = np.empty(n, dtype=np.int64)
    iperm[perm] = np.arange(n)
    net.iperm = iperm
    bw = int(np.max(np.abs(iperm[rows] - iperm[cols]))) if len(rows) else 0
    net.kl = net.ku = bw

    # owner node of each state and the links at each node, for the incremental Jacobian
    # (a line's flow state belongs to the line: -1 - its link)
    sowner = np.zeros(n, dtype=np.int64)
    for i in range(NI.shape[0]):
        for st in node_states(i):
            sowner[st] = i
    for k in range(LI.shape[0]):
        if LI[k, L.L_SX] >= 0:
            sowner[LI[k, L.L_SX]] = -1 - k
    adj: list[list[int]] = [[] for _ in range(NI.shape[0])]
    for k in range(LI.shape[0]):
        adj[int(LI[k, L.L_A])].append(k)
        adj[int(LI[k, L.L_B])].append(k)
    net.sowner = sowner
    net.nl_ptr = np.cumsum([0] + [len(a) for a in adj]).astype(np.int64)
    net.nl_idx = np.array([k for a in adj for k in a], dtype=np.int64)


def build_network(system: System) -> Network:
    return NetworkBuilder(system).build()
