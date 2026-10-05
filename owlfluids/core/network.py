"""Turns a System into the packed arrays the kernel steps.

Steps: group connected ports into nets, let each component create its nodes and
links, give every remaining net a small junction volume, spread fluids along the
links, set initial states, then work out the Jacobian sparsity, its colouring and
a band-narrowing order of the unknowns.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from .. import units
from ..components.library import get_type
from ..fluids.registry import FLUIDS, FluidSet, canonical
from ..io.system import System
from . import layout as L


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


@dataclass
class _Link:
    name: str
    kind: int
    a: int
    b: int
    owner: str
    cda: float = 0.0
    area: float = 0.0
    length: float = 0.0
    diameter: float = 0.0
    roughness: float = 0.0
    k_minor: float = 0.0
    ctrl: int = L.CTRL_FIXED
    p1: float = 0.0
    p2: float = 1.0
    draw_a: int = L.DRAW_MIX
    draw_b: int = L.DRAW_MIX
    valve: tuple | None = None


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
    kl: int = 0
    ku: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def n_states(self) -> int:
        return len(self.x0)

    @property
    def n_colors(self) -> int:
        return len(self.color_ptr) - 1


class NetworkBuilder:
    def __init__(self, system: System):
        self.system = system
        self.nodes: list[_Node] = []
        self.links: list[_Link] = []
        self.net_of: dict[tuple[str, str], int] = {}
        self.net_node: dict[int, int] = {}
        self.net_draw: dict[int, int] = {}
        self.net_volume: dict[int, float] = defaultdict(float)
        self.comp_nodes: dict[str, list[int]] = defaultdict(list)
        self.comp_links: dict[str, list[int]] = defaultdict(list)
        self.valves: dict[str, int] = {}
        self.warnings: list[str] = []
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
        for a, b in self.system.connections:
            pa, pb = self._port(a, comps), self._port(b, comps)
            union(pa, pb)
        roots: dict[tuple[str, str], int] = {}
        for p in parent:
            r = find(p)
            self.net_of[p] = roots.setdefault(r, len(roots))

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

    def new_boundary(self, comp, fluid, p, t) -> int:
        idx = self.new_node(comp, "", fluid=fluid, kind=L.NODE_BOUNDARY)
        self.nodes[idx].bp = p
        self.nodes[idx].bt = t
        return idx

    def new_pressurized_tank(self, comp, liquid, pressurant, vol, fill, p, t_l, t_g, area, heat):
        g = self.new_node(comp, "ullage", fluid=pressurant, volume=vol, init=("pt", p, t_g), heat=heat)
        liq = self.new_node(comp, "liquid", fluid=liquid, volume=vol * fill, init=("tank_liquid", p, t_l, fill),
                            kind=L.NODE_TANK_LIQUID)
        self.nodes[g].partner = liq
        self.nodes[liq].partner = g
        self.nodes[liq].area = area
        self.bind_port(comp, "gas", g, L.DRAW_MIX)
        self.bind_port(comp, "liquid", liq, L.DRAW_LIQUID)
        return g, liq

    def bind_port(self, comp, port, node, draw) -> None:
        net = self.net_of[(comp.id, port)]
        if net in self.net_node and self.net_node[net] != node:
            other = self.nodes[self.net_node[net]].owner
            raise BuildError(
                f"{comp.id}.{port} and {other} are connected directly, but both are volumes; "
                "put a pipe, valve or orifice between them")
        self.net_node[net] = node
        self.net_draw[net] = draw

    def port_node(self, comp, port) -> int:
        net = self.net_of[(comp.id, port)]
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

    def add_pipe(self, comp, suffix, a, b, area, diameter, length, roughness, k_minor, draw_a=0, draw_b=0):
        lk = _Link(f"{comp.id}.{suffix}", L.LINK_PIPE, a, b, comp.id, area=area, length=length,
                   diameter=diameter, roughness=roughness, k_minor=k_minor, draw_a=draw_a, draw_b=draw_b)
        self.links.append(lk)
        self.comp_links[comp.id].append(len(self.links) - 1)

    def add_orifice(self, comp, suffix, port_a, port_b, cda, ctrl=L.CTRL_FIXED, p1=0.0, p2=1.0, valve=None,
                    port_volume=None):
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
        lk = _Link(comp.id + (f".{suffix}" if suffix else ""), L.LINK_ORIFICE, a, b, comp.id, cda=cda, ctrl=ctrl,
                   p1=p1, p2=p2, draw_a=self.port_draw(comp, port_a), draw_b=self.port_draw(comp, port_b),
                   valve=valve)
        self.links.append(lk)
        idx = len(self.links) - 1
        self.comp_links[comp.id].append(idx)
        if ctrl == L.CTRL_VALVE:
            self.valves[comp.id] = idx

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
        # every net that is used by a link but owns no volume is a junction
        for net, node in self.net_node.items():
            n = self.nodes[node]
            if n.owner == "":
                n.volume = max(self.net_volume.get(net, 0.0), self.junction_volume)
        self._assign_fluids()
        return self._compile()

    def _assign_fluids(self) -> None:
        adj = defaultdict(list)
        for lk in self.links:
            adj[lk.a].append(lk.b)
            adj[lk.b].append(lk.a)
        src: dict[int, str] = {}
        for i, n in enumerate(self.nodes):
            if n.fluid:
                src[i] = n.fluid
        reached: dict[int, tuple[str, int]] = {}
        for start, fl in src.items():
            q = deque([start])
            seen = {start}
            while q:
                i = q.popleft()
                for j in adj[i]:
                    if j in seen or j in src:
                        continue
                    seen.add(j)
                    if j in reached and reached[j][0] != fl:
                        a = self.nodes[reached[j][1]].owner
                        b = self.nodes[start].owner
                        raise BuildError(
                            f"{self.nodes[j].name} is reached by {reached[j][0]} from {a} and {fl} from {b}; "
                            "lines may carry one fluid only (set 'fluid' on a component to separate them)")
                    reached[j] = (fl, start)
                    q.append(j)
        for i, n in enumerate(self.nodes):
            if n.fluid is None:
                if i not in reached:
                    raise BuildError(f"{n.name} is not connected to anything that sets its fluid")
                n.fluid = reached[i][0]
        for lk in self.links:
            fa, fb = self.nodes[lk.a].fluid, self.nodes[lk.b].fluid
            if fa != fb:
                raise BuildError(f"{lk.owner} connects {fa} to {fb}; lines may carry one fluid only")

    def _compile(self) -> Network:
        sysm = self.system
        models_cfg = {canonical(k): v for k, v in (sysm.setting("fluid_models") or {}).items()}
        used = sorted({n.fluid for n in self.nodes})
        models = []
        for f in used:
            m = models_cfg.get(f, FLUIDS[f][1])
            if m not in FLUIDS[f][2]:
                raise BuildError(f"{f} cannot use the {m} model; allowed: {', '.join(FLUIDS[f][2])}")
            models.append(m)
        fs = FluidSet(used, models, backend=sysm.setting("backend"))

        nn, nl = len(self.nodes), len(self.links)
        NI = np.zeros((nn, L.N_NI), dtype=np.int64)
        NF = np.zeros((nn, L.N_NF))
        x0: list[float] = []
        xscale: list[float] = []
        for i, n in enumerate(self.nodes):
            f = used.index(n.fluid)
            NI[i, L.N_KIND] = n.kind
            NI[i, L.N_FLUID] = f
            NI[i, L.N_PARTNER] = n.partner
            NF[i, L.NF_VOL] = n.volume
            NF[i, L.NF_HA] = n.heat[0]
            NF[i, L.NF_TAMB] = n.heat[1]
            if n.kind == L.NODE_BOUNDARY:
                NI[i, L.N_SX] = -1
                rho, u = fs.rho_u_from_pt(f, n.bp, n.bt)
                NF[i, L.NF_BP], NF[i, L.NF_BT], NF[i, L.NF_BRHO], NF[i, L.NF_BU] = n.bp, n.bt, rho, u
                continue
            rho, u, t0 = self._initial_state(fs, f, n)
            if n.kind == L.NODE_TANK_LIQUID:
                rhoref = fs.liquid_density(f, t0)
                NF[i, L.NF_RHOREF] = rhoref
                NF[i, L.NF_AREA] = n.area
                tank_vol = self.nodes[n.partner].volume
                NF[i, L.NF_MREF] = rhoref * 0.002 * tank_vol
                m = rhoref * n.volume
                rho = rhoref
            else:
                vol = n.volume
                if n.partner >= 0:  # ullage shares the tank with the liquid
                    vol = n.volume - self.nodes[n.partner].volume
                    if fs.model[f] != 1 and rho > fs.liquid_density(f, t0) * 0.5:
                        raise BuildError(
                            f"{n.owner}: the {n.fluid} pressurant would be liquid at its initial pressure and "
                            f"{t0:.1f} K; set a warmer gas_temperature")
                m = rho * vol
                # below this mass (about 1 kPa of gas) outflows are throttled, so a
                # volume pumped down to vacuum cannot be driven to negative mass
                NF[i, L.NF_MREF] = max(vol, 1e-9) * 0.01
            NI[i, L.N_SX] = len(x0)
            x0 += [m, m * u]
            xscale += [max(m, 1e-9), max(abs(m * u), m * 1e5, 1e-3)]
            if n.wall:
                mc, ha_in, ha_out, t_amb = n.wall
                NI[i, L.N_WALL] = 1
                NF[i, L.NF_WALLMC] = mc
                NF[i, L.NF_HA] = ha_in
                NF[i, L.NF_HAAMB] = ha_out
                NF[i, L.NF_TAMB] = t_amb
                x0.append(t_amb)
                xscale.append(100.0)
        # ullage nodes need the liquid reference density of their partner
        LI = np.zeros((nl, L.N_LI), dtype=np.int64)
        LF = np.zeros((nl, L.N_LF))
        LV = np.zeros((nl, L.N_V))
        for k, lk in enumerate(self.links):
            LI[k] = (lk.kind, lk.a, lk.b, lk.draw_a, lk.draw_b, lk.ctrl, -1)
            LF[k, L.LF_CDA] = lk.cda
            LF[k, L.LF_AREA] = lk.area
            LF[k, L.LF_LEN] = lk.length
            LF[k, L.LF_DIAM] = lk.diameter
            LF[k, L.LF_ROUGH] = lk.roughness
            LF[k, L.LF_K] = lk.k_minor
            LF[k, L.LF_P1] = lk.p1
            LF[k, L.LF_P2] = lk.p2
            if lk.kind == L.LINK_PIPE:
                LI[k, L.L_SX] = len(x0)
                x0.append(0.0)
                xscale.append(1e-2)
            if lk.valve:
                start, r_open, r_close = lk.valve
                LV[k] = (start, start, r_open, r_close)
            else:
                LV[k] = (1.0, 1.0, 0.0, 0.0)

        port_nodes = {f"{cid}.{p}": self.net_node[net] for (cid, p), net in self.net_of.items()
                      if net in self.net_node}
        net = Network(sysm, fs, units.parse(sysm.setting("dt"), "time"), NI, NF, LI, LF, LV,
                      np.array(x0, dtype=float), np.array(xscale, dtype=float),
                      [n.name for n in self.nodes], [lk.name for lk in self.links],
                      [n.owner for n in self.nodes], [lk.owner for lk in self.links],
                      port_nodes, dict(self.comp_nodes), dict(self.comp_links), dict(self.valves),
                      warnings=self.warnings)
        _structure(net)
        return net

    def _initial_state(self, fs: FluidSet, f: int, n: _Node) -> tuple[float, float, float]:
        init = n.init or ("pt", None, None)
        if init[0] == "saturated":
            _, t, p, fill = init
            if t is None:
                if p is None:
                    raise BuildError(f"{n.owner} needs temperature or pressure with its fill")
                t = fs.saturation_temperature(f, p)
            rho, u = fs.rho_u_saturated(f, t, fill)
            return rho, u, t
        if init[0] == "tank_liquid":
            _, p, t, _fill = init
            rho, u = fs.rho_u_from_pt(f, p, t)
            if fs.models[f] == "table":
                # take the liquid's energy at its own temperature on the liquid side
                import CoolProp.CoolProp as CP
                st = fs._backend_state(FLUIDS[fs.names[f]][0])
                tc = st.T_critical()
                t_use = min(t, tc - 1.0)
                try:
                    st.update(CP.PT_INPUTS, max(p, 1.0), t_use)
                    if st.Q() >= 0 or st.rhomass() < st.rhomass_critical():
                        raise ValueError
                except ValueError:
                    st.update(CP.QT_INPUTS, 0.0, t_use)
                rho, u = st.rhomass(), st.umass()
            return rho, u, t
        _, p, t = init
        p = self.p_init if p is None else p
        t = self.t_init if t is None else t
        rho, u = fs.rho_u_from_pt(f, p, t)
        return rho, u, t


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
        return list(range(sx, sx + (3 if NI[i, L.N_WALL] else 2)))

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
        if LI[k, L.L_SX] >= 0:
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
    perm = reverse_cuthill_mckee(sym, symmetric_mode=True).astype(np.int64)
    iperm = np.empty(n, dtype=np.int64)
    iperm[perm] = np.arange(n)
    net.iperm = iperm
    bw = int(np.max(np.abs(iperm[rows] - iperm[cols]))) if len(rows) else 0
    net.kl = net.ku = bw


def build_network(system: System) -> Network:
    return NetworkBuilder(system).build()
