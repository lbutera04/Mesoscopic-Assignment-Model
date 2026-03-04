from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra as cs_dijkstra
from tqdm import tqdm
from numba import njit

import networkx as nx
import sumolib

from cccar.config import Config
from cccar.osm.attributes import load_bad_edges, build_edge_attributes, _sumo_roadclass, _DECAY
from cccar.osm.graph_build import build_connection_graph_no_internals, build_csr_from_graph
from cccar.osm.geo import build_edges_gdf, attach_block_groups
from cccar.demand.replica import load_replica, map_replica_to_edges
from cccar.demand.spawns import build_spawn_tables
from cccar.demand.centroids import build_bg_centroid_edge_map
# from cccar.corridor.core import _build_od_core_csr_numba_fast


@dataclass
class DagMetrics:
    o_eid: str
    d_eid: str
    trips: int
    ssp: float
    nloc: int
    mloc: int
    edge_cut: Optional[int]
    dom_chain_len: Optional[int]


def _dominators_chain_len(idom: Dict[int, int], s: int, t: int) -> Optional[int]:
    """
    Returns number of strict dominators on the s->t dominator chain (excluding s and t).
    If t not reachable / dominators fail, returns None.
    """
    if t not in idom or s not in idom:
        return None
    if t == s:
        return 0

    # Walk idom pointers from t back to s
    chain = []
    cur = t
    seen = set()
    while True:
        if cur in seen:
            # cycle shouldn't happen in dominator tree, but guard anyway
            return None
        seen.add(cur)
        if cur == s:
            break
        if cur not in idom:
            return None
        cur = idom[cur]
        chain.append(cur)

    # chain currently includes s at the end and includes t's parent etc.
    # strict dominators excluding endpoints:
    strict = [x for x in chain if x != s and x != t]
    return len(strict)


def _csr_to_nx_digraph(indptr: np.ndarray, indices: np.ndarray, nloc: int) -> nx.DiGraph:
    """
    Build a NetworkX DiGraph from CSR. Nodes are 0..nloc-1.
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(nloc))
    for u in range(nloc):
        a = int(indptr[u])
        b = int(indptr[u + 1])
        if a == b:
            continue
        vs = indices[a:b]
        # indices are already OD-local node ids
        for v in vs.tolist():
            G.add_edge(u, int(v))
    return G

def _core_to_physical_nx_digraph(
    od_indptr: np.ndarray,
    od_indices: np.ndarray,
    od_pos: np.ndarray,
    core_nodes: np.ndarray,
    nloc: int,
    start_loc: int,
    end_loc: int,
) -> tuple[nx.DiGraph, int, int]:
    """
    Project the OD-local *core CSR* (which may be state-expanded) down to a
    *physical* DiGraph:

      - physical nodes = core_nodes[u_loc] (global node id; may repeat in core)
      - physical edges = unique od_pos slots (global CSR position), ignoring
        promotion edges where od_pos == -1.

    By deduping on od_pos, this prevents state-layer duplication from inflating
    connectivity metrics.
    """
    G = nx.DiGraph()

    s_phys = int(core_nodes[int(start_loc)])
    t_phys = int(core_nodes[int(end_loc)])
    G.add_node(s_phys)
    G.add_node(t_phys)

    seen_pos: set[int] = set()

    for u in range(nloc):
        u_phys = int(core_nodes[u])
        a = int(od_indptr[u])
        b = int(od_indptr[u + 1])
        for tt in range(a, b):
            p = int(od_pos[tt])
            if p < 0:
                # promotion edge (not a real network arc)
                continue
            if p in seen_pos:
                # same physical arc appears multiple times across state layers
                continue
            seen_pos.add(p)

            v = int(od_indices[tt])
            v_phys = int(core_nodes[v])
            G.add_edge(u_phys, v_phys, pos=p)

    return G, s_phys, t_phys


@njit(cache=True)
def _build_od_core_csr_numba_layered_viols(
    origin_i: int,
    dest_i: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    w: np.ndarray,
    dist_o: np.ndarray,
    dist_to_d: np.ndarray,
    reach_o: np.ndarray,
    slack: float,
    decay_node: np.ndarray,
    stamp: np.ndarray,
    g2l: np.ndarray,
    cur_stamp: int,
    viol_budget: int,          # <-- set to 3 at callsite
):
    """
    Like your current builder, but replaces strict per-edge monotone potential
    with a layered "violation budget" state graph.

    State = (u_loc, b) where b = number of non-monotone edges used so far.
      - If F[v] > F[u] (monotone), transition: (u,b) -> (v,b)
      - Else (violation), transition: (u,b) -> (v,b+1) if b < viol_budget

    Acyclicity: every transition strictly increases lexicographic key (b, F),
    so the state graph is a DAG.

    IMPORTANT: To keep a *single* sink (end_loc), we unify all dest states
    via "promotion" edges: (dest,b) -> (dest,b+1) for b < viol_budget.
    Then end_loc is (dest, viol_budget).

    Returns a CSR over *state-nodes* (so duplicates of the same physical node
    across different b are expected).
    """
    n = indptr.shape[0] - 1
    ssp = dist_o[dest_i]
    if not np.isfinite(ssp) or ssp <= 0.0:
        return (np.zeros(1, np.int64),
                np.zeros(0, np.int64),
                np.zeros(0, np.int64),
                np.zeros(0, np.float64),
                np.zeros(0, np.int64),
                -1, -1)

    budget = slack * ssp
    cand_stamp = cur_stamp

    # ------------------------------------------------------------------
    # 1) Candidate nodes among reach_o, using slack lower-bound filter.
    # ------------------------------------------------------------------
    k = 0
    for ii in range(reach_o.shape[0]):
        u = int(reach_o[ii])
        hu = dist_to_d[u]
        if not np.isfinite(hu):
            continue
        du = dist_o[u]
        if (du + hu) <= budget:
            k += 1

    if k == 0:
        return (np.zeros(1, np.int64),
                np.zeros(0, np.int64),
                np.zeros(0, np.int64),
                np.zeros(0, np.float64),
                np.zeros(0, np.int64),
                -1, -1)

    cand_nodes = np.empty(k, np.int64)
    jj = 0
    for ii in range(reach_o.shape[0]):
        u = int(reach_o[ii])
        hu = dist_to_d[u]
        if not np.isfinite(hu):
            continue
        du = dist_o[u]
        if (du + hu) <= budget:
            cand_nodes[jj] = u
            stamp[u] = cand_stamp
            g2l[u] = jj
            jj += 1

    if stamp[origin_i] != cand_stamp or stamp[dest_i] != cand_stamp:
        return (np.zeros(1, np.int64),
                np.zeros(0, np.int64),
                np.zeros(0, np.int64),
                np.zeros(0, np.float64),
                np.zeros(0, np.int64),
                -1, -1)

    o_loc = int(g2l[origin_i])
    d_loc = int(g2l[dest_i])

    # ------------------------------------------------------------------
    # 2) Build the candidate CSR (NO monotonicity filter here).
    #     We keep only edges that meet slack & stay inside cand set.
    #
    #     The layered monotonicity is enforced later during reachability
    #     and core CSR construction.
    # ------------------------------------------------------------------
    outdeg = np.zeros(k, np.int64)

    # Precompute mixed-potential F on candidate nodes
    F = np.empty(k, np.float64)
    for i in range(k):
        u = cand_nodes[i]
        F[i] = dist_o[u] - dist_to_d[u]

    for i_u in range(k):
        u = cand_nodes[i_u]
        gu = dist_o[u]
        s = indptr[u]
        e = indptr[u + 1]
        cnt = 0
        for p in range(s, e):
            v = int(indices[p])
            if stamp[v] != cand_stamp:
                continue
            if gu + w[p] + dist_to_d[v] > budget:
                continue
            cnt += 1
        outdeg[i_u] = cnt

    m2 = 0
    for i in range(k):
        m2 += outdeg[i]
    if m2 == 0:
        return (np.zeros(1, np.int64),
                np.zeros(0, np.int64),
                np.zeros(0, np.int64),
                np.zeros(0, np.float64),
                np.zeros(0, np.int64),
                -1, -1)

    cand_indptr = np.zeros(k + 1, np.int64)
    for i in range(k):
        cand_indptr[i + 1] = cand_indptr[i] + outdeg[i]

    cand_indices = np.empty(m2, np.int64)
    cand_pos = np.empty(m2, np.int64)

    cursor = np.empty(k, np.int64)
    for i in range(k):
        cursor[i] = cand_indptr[i]

    for i_u in range(k):
        u = cand_nodes[i_u]
        gu = dist_o[u]
        s = indptr[u]
        e = indptr[u + 1]
        for p in range(s, e):
            v = int(indices[p])
            if stamp[v] != cand_stamp:
                continue
            if gu + w[p] + dist_to_d[v] > budget:
                continue
            j_v = int(g2l[v])
            t = cursor[i_u]
            cand_indices[t] = j_v
            cand_pos[t] = p
            cursor[i_u] = t + 1

    # ------------------------------------------------------------------
    # 3) Build reverse CSR for the candidate graph (for backward reachability)
    # ------------------------------------------------------------------
    indeg = np.zeros(k, np.int64)
    for u in range(k):
        s = cand_indptr[u]
        e = cand_indptr[u + 1]
        for t in range(s, e):
            indeg[int(cand_indices[t])] += 1

    rev_indptr = np.zeros(k + 1, np.int64)
    for i in range(k):
        rev_indptr[i + 1] = rev_indptr[i] + indeg[i]

    rev_src = np.empty(m2, np.int64)
    rev_pos = np.empty(m2, np.int64)

    cur2 = np.empty(k, np.int64)
    for i in range(k):
        cur2[i] = rev_indptr[i]

    for u in range(k):
        s = cand_indptr[u]
        e = cand_indptr[u + 1]
        for t in range(s, e):
            v = int(cand_indices[t])
            tt = cur2[v]
            rev_src[tt] = u
            rev_pos[tt] = cand_pos[t]
            cur2[v] = tt + 1

    # ------------------------------------------------------------------
    # 4) Layered reachability prune on the implicit state graph
    #    State index: idx = b*k + u_loc, where b in [0..viol_budget]
    #
    #    Also add promotion edges at dest:
    #       (d_loc,b) -> (d_loc,b+1) for b < viol_budget
    #    so we can have a single sink end_loc = (d_loc, viol_budget).
    # ------------------------------------------------------------------
    B = viol_budget
    if B < 0:
        B = 0
    nstate = (B + 1) * k

    # Forward reachability from (o_loc, 0)
    fwd = np.zeros(nstate, np.uint8)
    stack = np.empty(nstate, np.int64)
    top = 0

    start_state = 0 * k + o_loc
    stack[top] = start_state
    top += 1
    fwd[start_state] = 1

    while top > 0:
        top -= 1
        idx = int(stack[top])
        b = idx // k
        u = idx - b * k

        # promotion at dest: (d,b)->(d,b+1)
        if u == d_loc and b < B:
            idx2 = (b + 1) * k + u
            if fwd[idx2] == 0:
                fwd[idx2] = 1
                stack[top] = idx2
                top += 1

        Fu = F[u]
        s = cand_indptr[u]
        e = cand_indptr[u + 1]
        for t in range(s, e):
            v = int(cand_indices[t])

            # classify monotone vs violation by potential
            # monotone if F[v] > F[u]
            if F[v] > Fu:
                b2 = b
            else:
                if b >= B:
                    continue
                b2 = b + 1

                # OPTIONAL GUARD against "budget-consuming loops":
                # Uncomment to prevent strong backward moves in dist_o.
                # (Useful if you see paths bouncing u<->v consuming violations.)
                # if dist_o[cand_nodes[v]] + 0.0 < dist_o[cand_nodes[u]]:
                #     continue

            idx2 = b2 * k + v
            if fwd[idx2] == 0:
                fwd[idx2] = 1
                stack[top] = idx2
                top += 1

    # Backward reachability to the unified sink (d_loc, B)
    bwd = np.zeros(nstate, np.uint8)
    stack2 = np.empty(nstate, np.int64)
    top2 = 0

    end_state = B * k + d_loc
    stack2[top2] = end_state
    top2 += 1
    bwd[end_state] = 1

    while top2 > 0:
        top2 -= 1
        idx = int(stack2[top2])
        b = idx // k
        v = idx - b * k

        # reverse of promotion edges: (d,b)->(d,b+1)
        if v == d_loc and b > 0:
            idx2 = (b - 1) * k + v
            if bwd[idx2] == 0:
                bwd[idx2] = 1
                stack2[top2] = idx2
                top2 += 1

        Fv = F[v]
        s = rev_indptr[v]
        e = rev_indptr[v + 1]
        for t in range(s, e):
            u = int(rev_src[t])

            # forward edge u->v is monotone if F[v] > F[u]
            if Fv > F[u]:
                b_prev = b
            else:
                # forward would have consumed one budget: (u,b-1)->(v,b)
                if b == 0:
                    continue
                b_prev = b - 1

            idx2 = b_prev * k + u
            if bwd[idx2] == 0:
                bwd[idx2] = 1
                stack2[top2] = idx2
                top2 += 1

    # Core mask over states
    core_mask = np.zeros(nstate, np.uint8)
    core_count = 0
    for i in range(nstate):
        if fwd[i] and bwd[i]:
            core_mask[i] = 1
            core_count += 1

    # Must include start_state and end_state
    if core_count == 0 or core_mask[start_state] == 0 or core_mask[end_state] == 0:
        return (np.zeros(1, np.int64),
                np.zeros(0, np.int64),
                np.zeros(0, np.int64),
                np.zeros(0, np.float64),
                np.zeros(0, np.int64),
                -1, -1)

    # ------------------------------------------------------------------
    # 5) Compress core states to CSR, compute per-slot decay.
    #    Nodes in this CSR are *state nodes*; core_nodes_global stores the
    #    underlying global node id (cand_nodes[u_loc]) for each state.
    # ------------------------------------------------------------------
    candstate2core = np.full(nstate, -1, np.int64)
    core_nodes_global = np.empty(core_count, np.int64)
    ci = 0
    for idx in range(nstate):
        if core_mask[idx]:
            candstate2core[idx] = ci
            b = idx // k
            u = idx - b * k
            core_nodes_global[ci] = cand_nodes[u]  # physical node id (may repeat)
            ci += 1

    # Count outdegrees in compressed core CSR (state transitions)
    outdeg2 = np.zeros(core_count, np.int64)
    for idx in range(nstate):
        if core_mask[idx] == 0:
            continue
        b = idx // k
        u = idx - b * k
        u2 = int(candstate2core[idx])

        # promotion at dest
        if u == d_loc and b < B:
            idx_to = (b + 1) * k + u
            if core_mask[idx_to]:
                outdeg2[u2] += 1

        Fu = F[u]
        s = cand_indptr[u]
        e = cand_indptr[u + 1]
        for t in range(s, e):
            v = int(cand_indices[t])
            if F[v] > Fu:
                b2 = b
            else:
                if b >= B:
                    continue
                b2 = b + 1
                # Optional guard (same as above)
                # if dist_o[cand_nodes[v]] + 0.0 < dist_o[cand_nodes[u]]:
                #     continue

            idx_to = b2 * k + v
            if core_mask[idx_to]:
                outdeg2[u2] += 1

    m3 = 0
    for i in range(core_count):
        m3 += outdeg2[i]

    od_indptr = np.zeros(core_count + 1, np.int64)
    for i in range(core_count):
        od_indptr[i + 1] = od_indptr[i] + outdeg2[i]

    od_indices = np.empty(m3, np.int64)
    od_pos = np.empty(m3, np.int64)
    slot_decay = np.empty(m3, np.float64)

    cur3 = np.empty(core_count, np.int64)
    for i in range(core_count):
        cur3[i] = od_indptr[i]

    # Fill compressed CSR
    for idx in range(nstate):
        if core_mask[idx] == 0:
            continue
        b = idx // k
        u = idx - b * k
        u2 = int(candstate2core[idx])

        # promotion at dest
        if u == d_loc and b < B:
            idx_to = (b + 1) * k + u
            if core_mask[idx_to]:
                v2 = int(candstate2core[idx_to])
                tt = cur3[u2]
                od_indices[tt] = v2
                od_pos[tt] = -1            # sentinel: not a real road-edge
                v_gl = int(core_nodes_global[v2])
                slot_decay[tt] = decay_node[v_gl]
                cur3[u2] = tt + 1

        Fu = F[u]
        s = cand_indptr[u]
        e = cand_indptr[u + 1]
        for t in range(s, e):
            v = int(cand_indices[t])
            if F[v] > Fu:
                b2 = b
            else:
                if b >= B:
                    continue
                b2 = b + 1
                # Optional guard (same as above)
                # if dist_o[cand_nodes[v]] + 0.0 < dist_o[cand_nodes[u]]:
                #     continue

            idx_to = b2 * k + v
            if core_mask[idx_to] == 0:
                continue
            v2 = int(candstate2core[idx_to])
            tt = cur3[u2]
            od_indices[tt] = v2
            p_global = int(cand_pos[t])
            od_pos[tt] = p_global
            v_gl = int(core_nodes_global[v2])
            slot_decay[tt] = decay_node[v_gl]
            cur3[u2] = tt + 1

    start_loc = int(candstate2core[start_state])
    end_loc = int(candstate2core[end_state])
    return od_indptr, od_indices, od_pos, slot_decay, core_nodes_global, start_loc, end_loc


# Convenience wrapper for your requested budget = 3
@njit(cache=True)
def _build_od_core_csr_numba_fast(
    origin_i: int,
    dest_i: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    w: np.ndarray,
    dist_o: np.ndarray,
    dist_to_d: np.ndarray,
    reach_o: np.ndarray,
    slack: float,
    decay_node: np.ndarray,
    stamp: np.ndarray,
    g2l: np.ndarray,
    cur_stamp: int,
):
    return _build_od_core_csr_numba_layered_viols(
        origin_i, dest_i,
        indptr, indices, w,
        dist_o, dist_to_d,
        reach_o, slack,
        decay_node, stamp, g2l,
        cur_stamp,
        3,
    )

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CCCAR DAG structural diagnostics (min s-t cut, dominators) without sampling."
    )
    parser.add_argument("--max-ods", type=int, default=1000, help="Max number of OD pairs to analyze.")
    parser.add_argument(
        "--min-trips", type=int, default=1, help="Skip OD pairs with fewer than this many trips."
    )
    parser.add_argument(
        "--cut-limit-nodes",
        type=int,
        default=4000,
        help="Only run min s-t edge cut if nloc <= this (cuts can be expensive).",
    )
    parser.add_argument(
        "--cut-limit-edges",
        type=int,
        default=25000,
        help="Only run min s-t edge cut if mloc <= this (cuts can be expensive).",
    )
    parser.add_argument(
        "--report-top",
        type=int,
        default=15,
        help="How many worst/most-interesting ODs to print at end.",
    )
    args = parser.parse_args()

    cfg = Config()
    rng = np.random.default_rng(cfg.rng_seed)

    print("\n==================== CCCAR DAG DIAGNOSTICS ====================")

    # ---- Step 1: load net, build graph (same as cli.py up to CSR) ----
    print("\nStep 1: Load SUMO net and build edge-connection graph")
    net = sumolib.net.readNet(cfg.net_path)

    bad_edges = load_bad_edges(cfg.bad_edges_path)
    edge_attrs = build_edge_attributes(net, cfg.allowed_vtypes, bad_edges, cfg.bad_edge_penalty)
    Gfull = build_connection_graph_no_internals(net, edge_attrs, cfg.allowed_vtypes)

    print(f"  Drivable edges (nodes): {len(Gfull.nodes()):,}")
    print(f"  Connection arcs:        {Gfull.number_of_edges():,}")

    # ---- Step 2: geodata + BG ----
    print("\nStep 2: Build edges GeoDataFrame and attach block groups")
    edges_gdf = build_edges_gdf(net, edge_attrs)
    edges_gdf = attach_block_groups(edges_gdf, cfg.bg_path)

    # ---- Step 3: load Replica + spawn/despawn ----
    print("\nStep 3: Load Replica trips and map to spawn/despawn edges")
    replica_df = load_replica(cfg.replica_trips_path)
    for c in ["origin_bgrp_fips_2020", "destination_bgrp_fips_2020"]:
        if c in replica_df.columns:
            replica_df[c] = (
                replica_df[c]
                .astype("string")
                .str.replace(r"\.0$", "", regex=True)
                .str.zfill(12)
            )

    origin_bg_counts = replica_df["origin_bgrp_fips_2020"].value_counts().rename("trip_count")
    spawn_tables = build_spawn_tables(edges_gdf, origin_bg_counts)

    mapped = map_replica_to_edges(
        replica_df,
        spawn_tables,
        rng=rng,
        dep_bin_str=cfg.dep_bin_str,
        spread=True,
        weighted_rounds=False,
    )

    mapped = mapped[mapped["origin_bgrp_fips_2020"] != mapped["destination_bgrp_fips_2020"]].copy()

    if cfg.demand_scale < 1.0:
        before = len(mapped)
        mapped = mapped.sample(frac=cfg.demand_scale, random_state=cfg.rng_seed).reset_index(drop=True)
        print(f"  Demand scaling: kept {len(mapped):,} / {before:,} inter-BG trips ({100*cfg.demand_scale:.1f}%)")

    # ---- Step 4: centroid edges + OD pairs ----
    print("\nStep 4: Assign BG centroid edges and build time-independent OD demand")
    bg_centroid_edge = build_bg_centroid_edge_map(edges_gdf, cfg.bg_path, prefer_trunk=False)
    bg_centroid_edge = {str(k).zfill(12): v for k, v in bg_centroid_edge.items()}

    mapped["origin_centroid_edge"] = mapped["origin_bgrp_fips_2020"].map(bg_centroid_edge)
    mapped["dest_centroid_edge"] = mapped["destination_bgrp_fips_2020"].map(bg_centroid_edge)

    mapped = mapped.dropna(subset=["origin_centroid_edge", "dest_centroid_edge", "spawn_edge", "despawn_edge"]).copy()

    if "vtype_key" not in mapped.columns:
        if "primary_mode" in mapped.columns:
            mapped["vtype_key"] = (
                mapped["primary_mode"]
                .astype(str)
                .str.lower()
                .apply(lambda x: "truck" if ("truck" in x or "commercial" in x) else "car")
            )
        else:
            mapped["vtype_key"] = "car"

    od_pairs = (
        mapped
        .groupby(["origin_centroid_edge", "dest_centroid_edge", "vtype_key"], observed=True)
        .size()
        .reset_index(name="trip_count")
    )
    od_pairs = od_pairs[od_pairs["trip_count"] >= int(args.min_trips)].copy()

    print(f"  OD pairs (time-independent): {len(od_pairs):,}")

    # ---- CSR build ----
    print("\nPrep: Build CSR adjacency (travel-time)")
    nodes = list(Gfull.nodes())
    idx_map = {n: i for i, n in enumerate(nodes)}
    A = build_csr_from_graph(Gfull, nodes)
    indptr = A.indptr.astype(np.int64)
    indices = A.indices.astype(np.int64)
    base_w = A.data.astype(np.float64)

    # ---- Dijkstra precompute (same spirit as api.py; needed for corridor build) ----
    n = len(nodes)
    Asp = sp.csr_matrix((base_w, indices, indptr), shape=(n, n))
    AT = Asp.transpose().tocsr()

    o_eids = pd.Series(od_pairs["origin_centroid_edge"].astype(str).unique())
    d_eids = pd.Series(od_pairs["dest_centroid_edge"].astype(str).unique())

    o_idx = [idx_map[e] for e in o_eids if e in idx_map]
    d_idx = [idx_map[e] for e in d_eids if e in idx_map]

    # Dedup preserve order
    seen = set()
    o_idx = [i for i in o_idx if (i not in seen and not seen.add(i))]
    seen = set()
    d_idx = [i for i in d_idx if (i not in seen and not seen.add(i))]

    print(f"\nPrecompute Dijkstra: {len(o_idx):,} unique origins, {len(d_idx):,} unique destinations")

    dist_cache_o: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    reach_cache_o: Dict[int, np.ndarray] = {}
    dist_cache_to_d: Dict[int, np.ndarray] = {}

    for oi in tqdm(o_idx, desc="Precompute origin Dijkstra", leave=False):
        dist, pred = cs_dijkstra(Asp, directed=True, indices=[oi], return_predecessors=True)
        dist0 = dist[0].astype(np.float64)
        dist_cache_o[int(oi)] = (dist0, pred[0].astype(np.int64))
        reach_cache_o[int(oi)] = np.flatnonzero(np.isfinite(dist0)).astype(np.int64)

    for di in tqdm(d_idx, desc="Precompute dest Dijkstra (reverse)", leave=False):
        dist_to = cs_dijkstra(AT, directed=True, indices=[di], return_predecessors=False)
        dist_cache_to_d[int(di)] = dist_to[0].astype(np.float64)

    # ---- per-node decay (required by _build_od_core_csr_numba_fast) ----
    decay_node = np.empty(n, dtype=np.float64)
    for gi in range(n):
        rc = _sumo_roadclass(edge_attrs, nodes[gi])
        decay_node[gi] = float(_DECAY.get(rc, 0.05))

    # ---- reusable buffers for OD core builder ----
    stamp = np.zeros(n, dtype=np.int32)
    g2l = np.zeros(n, dtype=np.int64)
    cur_stamp = np.int32(1)

    # ---- Analyze ODs ----
    metrics: List[DagMetrics] = []

    # deterministic subset: highest-trip ODs first
    od_pairs_sorted = od_pairs.sort_values("trip_count", ascending=False).reset_index(drop=True)
    od_pairs_sorted = od_pairs_sorted.head(int(args.max_ods))

    print(f"\nAnalyzing {len(od_pairs_sorted):,} ODs (max_ods={args.max_ods})")

    for row in tqdm(od_pairs_sorted.itertuples(index=False), total=len(od_pairs_sorted), desc="DAG diagnostics"):
        o_eid = str(row.origin_centroid_edge)
        d_eid = str(row.dest_centroid_edge)
        trips = int(row.trip_count)

        if o_eid not in idx_map or d_eid not in idx_map:
            continue
        oi = int(idx_map[o_eid])
        di = int(idx_map[d_eid])

        if oi not in dist_cache_o or di not in dist_cache_to_d:
            continue

        dist_o, _pred_o = dist_cache_o[oi]
        dist_to_d = dist_cache_to_d[di]

        ssp = float(dist_o[di])
        if not np.isfinite(ssp):
            continue

        od_indptr, od_indices, od_pos, slot_decay, core_nodes, start_loc, end_loc = _build_od_core_csr_numba_fast(
            oi, di,
            indptr, indices, base_w,
            dist_o, dist_to_d,
            reach_cache_o[int(oi)],
            float(cfg.dag_slack),
            decay_node,
            stamp, g2l, int(cur_stamp),
        )
        cur_stamp = np.int32(cur_stamp + 1)
        if cur_stamp == 0:
            stamp[:] = 0
            cur_stamp = np.int32(1)

        nloc = int(len(od_indptr) - 1)
        mloc = int(len(od_indices))

        # Empty / failed corridor
        if mloc == 0 or start_loc < 0 or end_loc < 0 or nloc <= 0:
            metrics.append(DagMetrics(o_eid, d_eid, trips, ssp, nloc, mloc, None, None))
            continue

        # Build NX graph for diagnostics
        G = _csr_to_nx_digraph(od_indptr, od_indices, nloc)
        s = int(start_loc)
        t = int(end_loc)

        # Dominators chain length (cheap-ish)
        dom_chain_len: Optional[int] = None
        try:
            idom = nx.immediate_dominators(G, s)
            dom_chain_len = _dominators_chain_len(idom, s, t)
        except Exception:
            dom_chain_len = None

        # ---- Min s-t edge cut on PHYSICAL arcs (provably meaningful under state expansion) ----
        edge_cut = None  # this is what DagMetrics stores today

        if nloc <= int(args.cut_limit_nodes) and mloc <= int(args.cut_limit_edges):
            try:
                Gp, sop, tp = _core_to_physical_nx_digraph(
                    od_indptr, od_indices, od_pos, core_nodes,
                    nloc, start_loc, end_loc
                )

                cut_edges = nx.minimum_edge_cut(Gp, sop, tp)
                edge_cut = len(cut_edges)  # RAW: always report size (even if adjacent to s/t)

                # Optional: if you still want the "interior-only" view, compute it here
                # but DO NOT use it to overwrite edge_cut unless you *intend* to.
                # edge_cut_interior = edge_cut
                # if edge_cut == 1:
                #     u, v = next(iter(cut_edges))
                #     if u == sp or v == tp:
                #         edge_cut_interior = None

            except Exception:
                edge_cut = None

        metrics.append(DagMetrics(o_eid, d_eid, trips, ssp, nloc, mloc, edge_cut, dom_chain_len))
        continue

    if not metrics:
        print("\nNo diagnostics collected (check that centroid edges exist in idx_map and ODs are feasible).")
        return

    df = pd.DataFrame([m.__dict__ for m in metrics])

    # Summary
    print("\n==================== SUMMARY ====================")
    print(f"ODs analyzed: {len(df):,}")
    print("Core size (nloc nodes):",
          f"median={df['nloc'].median():.0f}, p90={df['nloc'].quantile(0.90):.0f}, max={df['nloc'].max():.0f}")
    print("Core size (mloc arcs):",
          f"median={df['mloc'].median():.0f}, p90={df['mloc'].quantile(0.90):.0f}, max={df['mloc'].max():.0f}")

    # Cut stats (only where computed)
    cut_df = df.dropna(subset=["edge_cut"]).copy()
    if len(cut_df) > 0:
        print("\nMin s-t edge cut (computed on gated subset only):")
        print(f"  computed for {len(cut_df):,} / {len(df):,} ODs")
        print(f"  cut==1: {(cut_df['edge_cut'] == 1).mean()*100:.2f}%")
        print(f"  cut<=2: {(cut_df['edge_cut'] <= 2).mean()*100:.2f}%")
        print(f"  median={cut_df['edge_cut'].median():.0f}, p90={cut_df['edge_cut'].quantile(0.90):.0f}, max={cut_df['edge_cut'].max():.0f}")
    else:
        print("\nMin s-t edge cut: not computed for any ODs (increase --cut-limit-nodes/edges).")

    # Dominator chain stats
    dom_df = df.dropna(subset=["dom_chain_len"]).copy()
    if len(dom_df) > 0:
        print("\nDominator chain length (strict dominators on s→t chain):")
        print(f"  computed for {len(dom_df):,} / {len(df):,} ODs")
        print(f"  dom_chain_len>=1: {(dom_df['dom_chain_len'] >= 1).mean()*100:.2f}%")
        print(f"  median={dom_df['dom_chain_len'].median():.0f}, p90={dom_df['dom_chain_len'].quantile(0.90):.0f}, max={dom_df['dom_chain_len'].max():.0f}")
    else:
        print("\nDominators: not computed for any ODs (unexpected; check NX version / graph).")

    # “Most structurally constrained” views
    print("\n==================== TOP CASES ====================")

    if len(cut_df) > 0:
        worst_cut = cut_df.sort_values(["edge_cut", "trips"], ascending=[True, False]).head(int(args.report_top))
        print(f"\nLowest min-cut ODs (top {args.report_top}):")
        print(worst_cut[["o_eid", "d_eid", "trips", "nloc", "mloc", "edge_cut", "dom_chain_len"]].to_string(index=False))

    if len(dom_df) > 0:
        worst_dom = dom_df.sort_values(["dom_chain_len", "trips"], ascending=[False, False]).head(int(args.report_top))
        print(f"\nLargest dominator-chain ODs (top {args.report_top}):")
        print(worst_dom[["o_eid", "d_eid", "trips", "nloc", "mloc", "edge_cut", "dom_chain_len"]].to_string(index=False))

    # Optional: write CSV for offline inspection
    out_csv = getattr(cfg, "out_dag_diag_csv", None)
    if isinstance(out_csv, str) and out_csv:
        df.to_csv(out_csv, index=False)
        print(f"\nWrote diagnostics CSV: {out_csv}")
    else:
        print("\nTip: add Config.out_dag_diag_csv to auto-write results.")

    print("\nDONE.\n")


if __name__ == "__main__":
    main()