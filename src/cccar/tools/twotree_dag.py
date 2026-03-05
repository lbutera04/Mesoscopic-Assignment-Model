import numpy as np
import networkx as nx
import osmnx as ox
from collections import deque

place = "Bernalillo County, NM"
network_type = "drive"
y0, x0 = 35.089004161208564, -106.59095403952803   # start (lat,lon)
y1, x1 = 35.13055817766804, -106.52443543435083     
slack = 1.50                 # budget = slack * ssp
eps_rel = 1e-9               # float hygiene


# ============================================================
# 0) small helpers
# ============================================================
def best_edge_uv_by_tt(Gm: nx.MultiDiGraph, u, v, weight="travel_time"):
    ed = Gm.get_edge_data(u, v)
    if not ed:
        return None, None
    best_k, best_w, best_d = None, np.inf, None
    for k, d in ed.items():
        w = float(d.get(weight, np.inf))
        if np.isfinite(w) and w < best_w:
            best_w, best_k, best_d = w, k, d
    return best_k, best_d

def reachability_prune(H: nx.MultiDiGraph, o, d) -> nx.MultiDiGraph:
    if o not in H or d not in H:
        return nx.MultiDiGraph()
    fwd = set(nx.descendants(H, o)) | {o}
    bwd = set(nx.descendants(H.reverse(copy=False), d)) | {d}
    core = fwd & bwd
    if o not in core or d not in core:
        return nx.MultiDiGraph()
    return H.subgraph(core).copy()

def is_dag_multidigraph(H: nx.MultiDiGraph) -> bool:
    D = nx.DiGraph()
    D.add_nodes_from(H.nodes)
    D.add_edges_from((u, v) for (u, v) in H.edges())
    return nx.is_directed_acyclic_graph(D)

def find_one_cycle(H: nx.MultiDiGraph):
    D = nx.DiGraph()
    D.add_nodes_from(H.nodes)
    D.add_edges_from((u, v) for (u, v) in H.edges())
    try:
        cyc = nx.find_cycle(D, orientation="original")
        return [(u, v) for (u, v, _) in cyc]
    except Exception:
        return None

def topo_index(H: nx.MultiDiGraph) -> dict:
    D = nx.DiGraph()
    D.add_nodes_from(H.nodes)
    D.add_edges_from((u, v) for (u, v) in H.edges())
    order = list(nx.topological_sort(D))
    return {n: i for i, n in enumerate(order)}

# ============================================================
# 1) build gated graph
# ============================================================
def build_gate_graph(
    G: nx.MultiDiGraph,
    orig,
    dest,
    *,
    slack: float,
    weight: str = "travel_time",
    eps_rel: float = 1e-9,
):
    dist_o = nx.single_source_dijkstra_path_length(G, orig, weight=weight)
    dist_to_d = nx.single_source_dijkstra_path_length(G.reverse(copy=False), dest, weight=weight)

    ssp_path = nx.shortest_path(G, orig, dest, weight=weight)
    ssp = 0.0
    for u, v in zip(ssp_path[:-1], ssp_path[1:]):
        k, d = best_edge_uv_by_tt(G, u, v, weight=weight)
        ssp += float(d.get(weight, np.inf))
    if not np.isfinite(ssp) or ssp <= 0:
        raise RuntimeError("SSP not finite/positive; choose different OD points.")

    budget = float(slack * ssp)
    eps = float(eps_rel * ssp)

    cand = []
    for u in G.nodes:
        du = dist_o.get(u, np.inf)
        hv = dist_to_d.get(u, np.inf)
        if np.isfinite(du) and np.isfinite(hv) and (du + hv <= budget):
            cand.append(u)
    cand_set = set(cand)

    H_gate = nx.MultiDiGraph()
    H_gate.graph.update(G.graph)
    H_gate.add_nodes_from((n, G.nodes[n]) for n in cand)

    for u in cand:
        du = dist_o.get(u, np.inf)
        if not np.isfinite(du):
            continue
        for v, edict in G[u].items():
            if v not in cand_set:
                continue
            hv = dist_to_d.get(v, np.inf)
            if not np.isfinite(hv):
                continue
            for k, data in edict.items():
                w = float(data.get(weight, np.inf))
                if not np.isfinite(w):
                    continue
                if du + w + hv <= budget + 1e-9:
                    H_gate.add_edge(u, v, key=k, **data)

    return H_gate, dist_o, dist_to_d, ssp, budget, eps, cand_set

# ============================================================
# 2) YOUR two-tree meet-middle (claiming) core
# ============================================================
def build_two_tree_meet_middle_core(
    H_gate: nx.MultiDiGraph,
    orig,
    dest,
    *,
    weight: str = "travel_time",
    tie_break: str = "min_id",
):
    """
    Meet-middle claiming with stopping:
      - forward expansion from orig on children_fwd
      - reverse expansion from dest on children_rev
      - do NOT "push through" nodes already claimed by the other side
      - add only edges used in claiming

    This matches the 'stop when you hit the other side' behavior (not naive union).
    """
    if orig not in H_gate or dest not in H_gate:
        return nx.MultiDiGraph()

    pred_fwd, dist_fwd = nx.dijkstra_predecessor_and_distance(H_gate, orig, weight=weight)
    pred_rev, dist_rev = nx.dijkstra_predecessor_and_distance(H_gate.reverse(copy=False), dest, weight=weight)

    # Build children lists (multi-parent) but we will only CLAIM a node once.
    children_fwd = {}
    for v, ps in pred_fwd.items():
        if v == orig:
            continue
        for p in ps:
            children_fwd.setdefault(p, []).append(v)

    children_rev = {}
    # pred_rev built on reversed graph: predecessor p for x => original edge x -> p
    for x, ps in pred_rev.items():
        if x == dest:
            continue
        for p in ps:
            children_rev.setdefault(p, []).append(x)

    # deterministic parent choice when multiple predecessors exist (only used when adding edge)
    def pick_best_parent_forward(v):
        ps = pred_fwd.get(v) or []
        if not ps:
            return None
        if tie_break == "min_id":
            return min(ps)
        return min(ps, key=lambda p: (dist_fwd.get(p, np.inf), p))

    def pick_best_parent_reverse(x):
        ps = pred_rev.get(x) or []
        if not ps:
            return None
        if tie_break == "min_id":
            return min(ps)
        return min(ps, key=lambda p: (dist_rev.get(p, np.inf), p))

    H = nx.MultiDiGraph()
    H.graph.update(H_gate.graph)
    H.add_nodes_from((n, H_gate.nodes[n]) for n in H_gate.nodes)

    F_claimed = {orig}
    R_claimed = {dest}
    qF = deque([orig])
    qR = deque([dest])

    while qF or qR:
        # forward step
        if qF:
            u = qF.popleft()
            for v in children_fwd.get(u, []):
                if v in F_claimed:
                    continue
                if v in R_claimed:
                    # meet: add u->v but do not claim/expand through v
                    k, d = best_edge_uv_by_tt(H_gate, u, v, weight=weight)
                    if d is not None:
                        H.add_edge(u, v, key=k, **d)
                    continue

                # claim v with its chosen parent edge (which will be some predecessor in pred_fwd)
                p = pick_best_parent_forward(v)
                if p is None:
                    continue
                k, d = best_edge_uv_by_tt(H_gate, p, v, weight=weight)
                if d is not None:
                    H.add_edge(p, v, key=k, **d)
                F_claimed.add(v)
                qF.append(v)

        # reverse step
        if qR:
            p = qR.popleft()
            for x in children_rev.get(p, []):
                if x in R_claimed:
                    continue
                if x in F_claimed:
                    # meet: add x->p but do not claim/expand through x
                    k, d = best_edge_uv_by_tt(H_gate, x, p, weight=weight)
                    if d is not None:
                        H.add_edge(x, p, key=k, **d)
                    continue

                # claim x with its chosen successor-toward-dest edge (x->parent_in_original)
                parent = pick_best_parent_reverse(x)
                if parent is None:
                    continue
                # original edge is x -> parent
                k, d = best_edge_uv_by_tt(H_gate, x, parent, weight=weight)
                if d is not None:
                    H.add_edge(x, parent, key=k, **d)
                R_claimed.add(x)
                qR.append(x)

    # prune to corridor
    H = reachability_prune(H, orig, dest)

    # sanity check DAG; if not, show cycle (so you can see what violated the intended stopping)
    if not is_dag_multidigraph(H):
        cyc = find_one_cycle(H)
        print("WARNING: two-tree meet-middle produced a cycle (showing one directed cycle):")
        print(cyc[:20] if cyc else cyc)

    return H

# ============================================================
# 3) targeted intersection-seed webbing (bounded Dijkstra)
# ============================================================
TARGET_HW = {
    "motorway","motorway_link","trunk","trunk_link",
    "primary","primary_link","secondary","secondary_link",
    "tertiary","tertiary_link","unclassified",
}

def hw_tag(d):
    hw = d.get("highway")
    if isinstance(hw, (list, tuple)) and hw:
        return str(hw[0])
    return "" if hw is None else str(hw)

def base_hw(hw: str) -> str:
    return hw[:-5] if hw.endswith("_link") else hw

def select_intersection_seeds(G_full: nx.MultiDiGraph, H_core: nx.MultiDiGraph,
                              *, min_incident=4, min_distinct_bases=2, require_link_present=False):
    core = set(H_core.nodes)
    seeds = []
    for n in core:
        cnt = 0
        bases = set()
        has_link = False

        # out edges
        if n in G_full:
            for v, edict in G_full[n].items():
                for _, d in edict.items():
                    hw = hw_tag(d)
                    if hw in TARGET_HW:
                        cnt += 1
                        bases.add(base_hw(hw))
                        if hw.endswith("_link"):
                            has_link = True

        # in edges
        for _, _, _, d in G_full.in_edges(n, keys=True, data=True):
            hw = hw_tag(d)
            if hw in TARGET_HW:
                cnt += 1
                bases.add(base_hw(hw))
                if hw.endswith("_link"):
                    has_link = True

        if cnt >= min_incident and len(bases) >= min_distinct_bases and (has_link or not require_link_present):
            seeds.append(n)
    return seeds

from collections import deque

def topo_init_from_dag(H: nx.MultiDiGraph):
    """Initialize a topo order and pos map from an existing DAG MultiDiGraph."""
    D = nx.DiGraph()
    D.add_nodes_from(H.nodes)
    D.add_edges_from((u, v) for (u, v) in H.edges())
    order = list(nx.topological_sort(D))
    pos = {n: i for i, n in enumerate(order)}
    return order, pos

def _bfs_within_interval(adj, start, lo, hi, pos):
    """Forward BFS restricted to nodes whose pos is in [lo, hi]."""
    seen = set()
    q = deque([start])
    while q:
        x = q.popleft()
        if x in seen:
            continue
        px = pos.get(x, None)
        if px is None or px < lo or px > hi:
            continue
        seen.add(x)
        for y in adj.get(x, ()):
            py = pos.get(y, None)
            if py is None or py < lo or py > hi:
                continue
            if y not in seen:
                q.append(y)
    return seen

def topo_add_edge_online(order, pos, adj, radj, u, v):
    """
    Try to add constraint u->v to an existing topological order.
    If possible, updates 'order' and 'pos' in place and returns True.
    If it would create a cycle, returns False (caller should reject edge).
    """
    if u == v:
        return False

    pu = pos.get(u, None)
    pv = pos.get(v, None)

    # New nodes: append at end with stable position
    if pu is None:
        pos[u] = len(order); order.append(u); pu = pos[u]
    if pv is None:
        pos[v] = len(order); order.append(v); pv = pos[v]

    if pu < pv:
        return True  # already consistent

    # We need to reorder nodes in the interval [pv, pu]
    lo, hi = pv, pu

    # F = nodes reachable forward from v within interval
    F = _bfs_within_interval(adj, v, lo, hi, pos)

    # B = nodes that can reach u (i.e. backward reachable from u) within interval
    B = _bfs_within_interval(radj, u, lo, hi, pos)

    # If there's overlap, adding u->v creates a cycle
    # (because v ->* x and x ->* u already exists with x in overlap, plus u->v closes cycle)
    if F & B:
        return False

    # Otherwise, we can reorder by moving F after B within [lo, hi] while preserving internal order.
    interval_nodes = order[lo:hi+1]

    # keep relative order as currently in 'order'
    F_list = [x for x in interval_nodes if x in F]
    B_list = [x for x in interval_nodes if x in B]
    M_list = [x for x in interval_nodes if (x not in F and x not in B)]  # middle untouched

    # New interval: (interval - F) followed by F
    # but (interval - F) must preserve B before others as it already does; we keep current order via B_list + M_list
    new_interval = B_list + M_list + F_list

    # Write back
    order[lo:hi+1] = new_interval
    for i in range(lo, hi+1):
        pos[order[i]] = i

    return True

import heapq

def web_out_from_seeds_halo_dynamic_topo(
    H_core: nx.MultiDiGraph,
    H_gate: nx.MultiDiGraph,
    seeds: list,
    *,
    G_full: nx.MultiDiGraph,
    dist_o: dict,
    dist_to_d: dict,
    weight="travel_time",
    cutoff_sec=180.0,
    min_forward_jump=10,
    max_targets_per_seed=15,
    max_edges_added_total=8000,
):
    """
    Halo webbing with TRUE dynamic topo maintenance:
      - Search on full H_gate
      - STOP whenever you hit any node currently in H_core (except seed)
      - Choose targets that are downstream by rank among ORIGINAL core nodes
      - Reconstruct path and add edges one-by-one
      - Each edge is validated against and merged into a GLOBAL topo order via online update
      - If an edge would create a cycle, reject that target path (do not partially add)
    """
    if H_core.number_of_nodes() == 0:
        return H_core

    # Freeze original core for ranking / downstream test only
    original_core = set(H_core.nodes)

    # Rank among ORIGINAL nodes (your proxy for downstream)
    orig_sorted = sorted(
        original_core,
        key=lambda n: (float(dist_o.get(n, np.inf)), -float(dist_to_d.get(n, np.inf)), int(n))
    )
    rank = {n: i for i, n in enumerate(orig_sorted)}

    # Initialize dynamic topo state from current DAG
    if not is_dag_multidigraph(H_core):
        raise RuntimeError("H_core must start as a DAG for dynamic topo webbing.")
    order, pos = topo_init_from_dag(H_core)

    # Maintain adjacency for online topo updates on the evolving core graph (simple DiGraph view)
    adj = {n: set() for n in H_core.nodes}
    radj = {n: set() for n in H_core.nodes}
    for u, v in H_core.edges():
        adj.setdefault(u, set()).add(v)
        radj.setdefault(v, set()).add(u)

    added = 0

    def ensure_node(n):
        if n not in H_core:
            H_core.add_node(n, **G_full.nodes[n])
        adj.setdefault(n, set())
        radj.setdefault(n, set())
        if n not in pos:
            pos[n] = len(order)
            order.append(n)

    def try_add_edge_safe(u, v):
        """
        Attempt to add u->v into H_core while keeping DAG by online topo update.
        Returns True if committed, False if it would create a cycle.
        """
        nonlocal added
        if added >= max_edges_added_total:
            return False

        ensure_node(u)
        ensure_node(v)

        # First, check if adding the constraint can be accommodated
        ok = topo_add_edge_online(order, pos, adj, radj, u, v)
        if not ok:
            return False  # would introduce a cycle

        # Now commit into adjacency view
        adj[u].add(v)
        radj[v].add(u)

        # Commit the MultiDiGraph edge using your "best edge"
        k, d = best_edge_uv_by_tt(H_gate, u, v, weight=weight)
        if d is None:
            # If there is no actual edge data in H_gate, roll back adjacency constraint
            # (rare if your pred reconstruction is consistent)
            adj[u].remove(v)
            radj[v].remove(u)
            # We *could* also roll back topo order, but simplest is: don't add constraints that can't be realized.
            # In practice, pred edges come from H_gate so d should exist.
            return False

        if not H_core.has_edge(u, v, key=k):
            H_core.add_edge(u, v, key=k, **d)
            added += 1
        return True

    # Webbing loop
    for s in seeds:
        if added >= max_edges_added_total:
            break
        if s not in H_gate:
            continue

        rs = rank.get(s, None)
        if rs is None:
            continue

        dist = {s: 0.0}
        pred = {s: None}
        pq = [(0.0, s)]
        targets = []

        while pq:
            d_u, u = heapq.heappop(pq)
            if d_u > cutoff_sec:
                continue

            for v in H_gate.successors(u):
                if v in dist:
                    continue

                k, d = best_edge_uv_by_tt(H_gate, u, v, weight=weight)
                if d is None:
                    continue
                w = float(d.get(weight, np.inf))
                if not np.isfinite(w):
                    continue

                nd = d_u + w
                if nd > cutoff_sec:
                    continue

                dist[v] = nd
                pred[v] = u

                # stop at any CURRENT DAG node
                if v in H_core and v != s:
                    # accept as target only if it's an ORIGINAL core node and "downstream enough"
                    if v in original_core:
                        rt = rank.get(v)
                        if rt is not None and rt >= rs + min_forward_jump:
                            targets.append((nd, v))
                    continue

                heapq.heappush(pq, (nd, v))

        targets.sort()
        targets = targets[:max_targets_per_seed]

        for _, t in targets:
            if added >= max_edges_added_total:
                break

            # reconstruct path t -> s
            cur = t
            path = [cur]
            while cur != s:
                p = pred.get(cur)
                if p is None:
                    break
                cur = p
                path.append(cur)
                if len(path) > 10000:
                    break

            if path[-1] != s:
                continue
            path.reverse()

            # IMPORTANT: add path edges transactionally
            # If any edge would create a cycle, reject the whole path.
            committed = []
            ok = True
            for u, v in zip(path[:-1], path[1:]):
                if not try_add_edge_safe(u, v):
                    ok = False
                    break
                committed.append((u, v))

            if not ok:
                # Roll back edges we added for this target (both graph edges and adjacency sets).
                # NOTE: Rolling back topo order itself is nontrivial; instead we do a safe strategy:
                #   - remove committed edges from H_core/adj/radj
                #   - then REBUILD topo state from scratch occasionally
                # Given sizes (~thousands), rebuild is fine and keeps correctness.
                for u, v in committed:
                    # remove from adjacency sets
                    if v in adj.get(u, ()):
                        adj[u].remove(v)
                    if u in radj.get(v, ()):
                        radj[v].remove(u)
                    # remove all multiedges u->v we might have added (conservative)
                    if H_core.has_edge(u, v):
                        # remove only edges that exist; if multiple, remove all (fine for “reject path”)
                        keys = list(H_core[u][v].keys())
                        for kk in keys:
                            H_core.remove_edge(u, v, key=kk)

                # Rebuild topo state from current H_core (still a DAG)
                if not is_dag_multidigraph(H_core):
                    # Shouldn't happen because we reject cycle-creating edges; but if it does, expose immediately.
                    cyc = find_one_cycle(H_core)
                    raise RuntimeError(f"Cycle detected despite rejection. Example cycle: {cyc}")

                order, pos = topo_init_from_dag(H_core)
                adj = {n: set() for n in H_core.nodes}
                radj = {n: set() for n in H_core.nodes}
                for uu, vv in H_core.edges():
                    adj[uu].add(vv)
                    radj[vv].add(uu)

    return H_core

# ============================================================
# RUN (uses your already-defined: place, network_type, x0,y0,x1,y1, slack, eps_rel)
# ============================================================
G = ox.graph_from_place(place, network_type=network_type, simplify=True)
sample_edge = next(iter(G.edges(data=True)))[-1]
if "travel_time" not in sample_edge:
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

orig = ox.nearest_nodes(G, X=x0, Y=y0)
dest = ox.nearest_nodes(G, X=x1, Y=y1)
print("Origin node:", orig, "Dest node:", dest)

H_gate, dist_o, dist_to_d, ssp, budget, eps, cand_set = build_gate_graph(
    G, orig, dest, slack=slack, weight="travel_time", eps_rel=eps_rel
)
print(f"SSP: {ssp:.1f}s   budget(slack={slack:.3f}): {budget:.1f}s")
print("H_gate:", H_gate.number_of_nodes(), "nodes,", H_gate.number_of_edges(), "edges")

H = build_two_tree_meet_middle_core(H_gate, orig, dest, weight="travel_time", tie_break="min_id")
print("Two-tree core (meet-middle):", H.number_of_nodes(), "nodes,", H.number_of_edges(), "edges",
      "| is_dag:", is_dag_multidigraph(H))

seeds = select_intersection_seeds(G, H, min_incident=4, min_distinct_bases=2, require_link_present=False)
print("Seed intersections in core:", len(seeds))

H_keep_old = H.copy()

H_web = web_out_from_seeds_halo_dynamic_topo(
    H, H_gate, seeds,
    G_full=G,
    dist_o=dist_o,
    dist_to_d=dist_to_d,
    cutoff_sec=360,
    min_forward_jump=1,
    max_targets_per_seed=400,
    max_edges_added_total=15000,
)
H_web = reachability_prune(H_web, orig, dest)
print("Webbed core:", H_web.number_of_nodes(), "nodes,", H_web.number_of_edges(), "edges",
      "| is_dag:", is_dag_multidigraph(H_web))