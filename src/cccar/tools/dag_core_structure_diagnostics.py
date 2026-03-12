#!/usr/bin/env python3
from __future__ import annotations

"""
Core-structure diagnostics for OD DAG builders.

This script mirrors the file-loading and precompute harness in tools/dag_benchmarks.py,
then *builds the actual OD DAGs* for each OD using:

  (A) forward-progress (monotone-potential) core builder
  (B) two-tree halo core + shortcut webbing

and computes a set of structural metrics to explain heavy-tail edge-volume behavior.

Run (from repo root, with cccar on PYTHONPATH):

  python tools/dag_core_structure_diagnostics.py --n-ods 200 --min-trips 5

You can also restrict to a single model:

  python tools/dag_core_structure_diagnostics.py --model twotree
  python tools/dag_core_structure_diagnostics.py --model forward

Notes
-----
- The two-tree/web core builder is imported from cccar.corridor.twotree_web.
- The forward-progress core builder below is copied from tools/dag_benchmarks.py
  (Numba kernel that constructs a CSR core using layered forward-progress/violation logic).

The goal is to answer:
- Are we forced through tiny cuts (bridges/dominators)?
- Or do we have lots of alternatives but the sampler collapses anyway?
"""

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra as cs_dijkstra
from tqdm import tqdm
from numba import njit

import sumolib

# Repo imports (same spirit as dag_benchmarks.py)
from ..config import Config
from ..osm.attributes import load_bad_edges, build_edge_attributes, _sumo_roadclass, _DECAY
from ..osm.graph_build import build_connection_graph_no_internals, build_csr_from_graph
from ..osm.geo import build_edges_gdf, attach_block_groups
from ..demand.replica import load_replica, map_replica_to_edges
from ..demand.spawns import build_spawn_tables
from ..demand.centroids import build_bg_centroid_edge_map

from ..corridor.twotree_web import (
    # build_od_twotree_web_csr_numba,
    # build_seed_bfs_tree_cache_hops_numba,
    compute_arterial_seed_nodes_from_edgeattrs,
    # compute_touched_seeds_for_od,
)

from ..corridor.DAW import (
    build_od_twotree_web_csr_numba,
    build_seed_bfs_tree_cache_hops_numba,
    # compute_arterial_seed_nodes_from_edgeattrs,
    compute_touched_seeds_for_od,
)

# =============================================================================
# Forward-progress core builder (copied from tools/dag_benchmarks.py)
# =============================================================================

@njit
def _build_od_core_csr_numba_layered_viols(
    indptr: np.ndarray,
    indices: np.ndarray,
    w: np.ndarray,
    dist_o: np.ndarray,
    dist_to_d: np.ndarray,
    reach_o: np.ndarray,
    start: int,
    end: int,
    slack: float,
    stamp: np.ndarray,
    g2l: np.ndarray,
    node_decay: np.ndarray,
):
    """
    Build an OD-local CSR "forward-progress" core.

    This is a faithful copy of the kernel used in tools/dag_benchmarks.py.
    It gates nodes by budget (dist_o + dist_to_d <= slack * ssp) and then
    keeps arcs using a layered progress/violation rule to enforce acyclicity.

    Returns:
      od_indptr, od_indices, od_pos, slot_decay, core_nodes, start_loc, end_loc

    Where:
      - core_nodes: global node ids (CSR indices) included in this OD core
      - od_* arrays define adjacency over local indices [0..nloc)
      - od_pos maps each local slot to a global arc index position in the full CSR
      - slot_decay is per-outgoing-slot decay (uses node_decay of head)
    """
    ssp = float(dist_o[int(end)])
    if (not math.isfinite(ssp)) or ssp <= 0.0:
        return (
            np.zeros(1, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int64),
            np.int64(-1),
            np.int64(-1),
        )

    budget = float(slack) * ssp
    eps_rel = 1e-9
    tol = float(eps_rel) * max(1.0, budget)

    # Stamp candidates + build g2l
    cand_stamp = stamp[0] + 1
    stamp[0] = cand_stamp

    nloc = 0
    for i in range(reach_o.shape[0]):
        gu = int(reach_o[i])
        if stamp[gu] == cand_stamp:
            continue
        if float(dist_o[gu]) + float(dist_to_d[gu]) <= budget + tol:
            stamp[gu] = cand_stamp
            g2l[gu] = nloc
            nloc += 1

    if nloc == 0 or stamp[int(start)] != cand_stamp or stamp[int(end)] != cand_stamp:
        return (
            np.zeros(1, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int64),
            np.int64(-1),
            np.int64(-1),
        )

    # Collect core nodes (inverse map l2g)
    core_nodes = np.empty(nloc, dtype=np.int64)
    for i in range(reach_o.shape[0]):
        gu = int(reach_o[i])
        if stamp[gu] == cand_stamp:
            core_nodes[int(g2l[gu])] = gu

    start_loc = int(g2l[int(start)])
    end_loc = int(g2l[int(end)])

    # Build local adjacency (first pass: counts)
    row_counts = np.zeros(nloc, dtype=np.int64)

    # Potential / progress: use dist_o as an increasing surrogate
    for lu in range(nloc):
        gu = int(core_nodes[lu])
        du = float(dist_o[gu])

        for p in range(int(indptr[gu]), int(indptr[gu + 1])):
            gv = int(indices[p])
            if stamp[gv] != cand_stamp:
                continue

            # budget-feasible edge check
            if du + float(w[p]) + float(dist_to_d[gv]) > budget + tol:
                continue

            # strict forward progress (acyclic)
            if float(dist_o[gv]) <= du + 1e-12:
                continue

            row_counts[lu] += 1

    # Prefix sum to indptr
    od_indptr = np.empty(nloc + 1, dtype=np.int64)
    od_indptr[0] = 0
    for i in range(nloc):
        od_indptr[i + 1] = od_indptr[i] + row_counts[i]
    mloc = int(od_indptr[nloc])

    od_indices = np.empty(mloc, dtype=np.int64)
    od_pos = np.empty(mloc, dtype=np.int64)
    slot_decay = np.empty(mloc, dtype=np.float64)

    # Second pass: fill
    write_ptr = od_indptr.copy()
    for lu in range(nloc):
        gu = int(core_nodes[lu])
        du = float(dist_o[gu])
        decay_u = float(node_decay[gu])

        for p in range(int(indptr[gu]), int(indptr[gu + 1])):
            gv = int(indices[p])
            if stamp[gv] != cand_stamp:
                continue
            if du + float(w[p]) + float(dist_to_d[gv]) > budget + tol:
                continue
            if float(dist_o[gv]) <= du + 1e-12:
                continue

            lv = int(g2l[gv])
            k = int(write_ptr[lu])
            od_indices[k] = lv
            od_pos[k] = p
            slot_decay[k] = decay_u
            write_ptr[lu] += 1

    return od_indptr, od_indices, od_pos, slot_decay, core_nodes, np.int64(start_loc), np.int64(end_loc)


# =============================================================================
# Metrics
# =============================================================================

@dataclass
class CoreMetrics:
    nloc: int
    mloc: int
    avg_outdeg: float
    max_outdeg: int
    frac_outdeg1: float
    undirected_bridges: int
    undirected_articulations: int
    dom_nodes_on_all_paths: int
    dom_edges_on_all_paths: int
    min_edge_cut: int  # -1 if skipped
    layer_width_entropy: float

def _layer_width_entropy(dist_o_loc: np.ndarray, ssp: float, nbins: int = 20) -> float:
    if not np.isfinite(ssp) or ssp <= 0:
        return float("nan")
    x = dist_o_loc / ssp
    x = np.clip(x, 0.0, 1.0)
    bins = np.minimum((x * nbins).astype(np.int32), nbins - 1)
    counts = np.bincount(bins, minlength=nbins).astype(np.float64)
    p = counts / max(1.0, counts.sum())
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def _csr_to_digraph(od_indptr: np.ndarray, od_indices: np.ndarray) -> nx.DiGraph:
    nloc = od_indptr.shape[0] - 1
    G = nx.DiGraph()
    G.add_nodes_from(range(nloc))
    for u in range(nloc):
        a = int(od_indptr[u]); b = int(od_indptr[u+1])
        if a == b:
            continue
        vs = od_indices[a:b].tolist()
        G.add_edges_from((u, v) for v in vs)
    return G

def compute_core_metrics(
    od_indptr: np.ndarray,
    od_indices: np.ndarray,
    core_nodes: np.ndarray,
    start_loc: int,
    end_loc: int,
    dist_o_global: np.ndarray,
    *,
    compute_mincut: bool = False,
) -> CoreMetrics:
    nloc = int(core_nodes.shape[0])
    mloc = int(od_indices.shape[0])

    outdeg = (od_indptr[1:] - od_indptr[:-1]).astype(np.int64)
    avg_outdeg = float(outdeg.mean()) if nloc > 0 else 0.0
    max_outdeg = int(outdeg.max()) if nloc > 0 else 0
    frac_outdeg1 = float((outdeg == 1).mean()) if nloc > 0 else 0.0

    DG = _csr_to_digraph(od_indptr, od_indices)

    # Undirected bottlenecks
    UG = DG.to_undirected(as_view=False)
    try:
        n_br = sum(1 for _ in nx.bridges(UG))
    except Exception:
        n_br = 0
    try:
        n_art = sum(1 for _ in nx.articulation_points(UG))
    except Exception:
        n_art = 0

    # Dominators (nodes/edges on all s->t paths)
    dom_nodes = 0
    dom_edges = 0
    try:
        idom = nx.immediate_dominators(DG, start_loc)
        # Nodes that dominate t are those on the idom chain from t to s (inclusive)
        chain = set()
        cur = end_loc
        while True:
            chain.add(cur)
            if cur == start_loc:
                break
            cur = idom.get(cur, start_loc)
            # guard against weirdness
            if cur in chain:
                break
        dom_nodes = max(0, len(chain) - 2)  # exclude s,t
        # Dominator edges along the chain (idom[v] -> v) excluding edges incident to s/t if desired
        dom_edges = max(0, len(chain) - 1)
    except Exception:
        dom_nodes = 0
        dom_edges = 0

    # Min edge cut (can be expensive)
    mec = -1
    if compute_mincut:
        try:
            cutset = nx.minimum_edge_cut(DG, start_loc, end_loc)
            mec = int(len(cutset))
        except Exception:
            mec = -1

    # Layer width entropy
    ssp = float(dist_o_global[int(core_nodes[int(end_loc)])]) if nloc > 0 else float("nan")
    dist_o_loc = dist_o_global[core_nodes.astype(np.int64)]
    ent = _layer_width_entropy(dist_o_loc, ssp, nbins=20)

    return CoreMetrics(
        nloc=nloc,
        mloc=mloc,
        avg_outdeg=avg_outdeg,
        max_outdeg=max_outdeg,
        frac_outdeg1=frac_outdeg1,
        undirected_bridges=int(n_br),
        undirected_articulations=int(n_art),
        dom_nodes_on_all_paths=int(dom_nodes),
        dom_edges_on_all_paths=int(dom_edges),
        min_edge_cut=int(mec),
        layer_width_entropy=float(ent),
    )


# =============================================================================
# Harness
# =============================================================================

def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-ods", type=int, default=200)
    parser.add_argument("--min-trips", type=int, default=1)
    parser.add_argument("--slack", type=float, default=None)
    parser.add_argument("--model", type=str, default="both", choices=["both", "forward", "twotree"])
    parser.add_argument("--max-web-edges", type=int, default=None)
    parser.add_argument("--web-max-hops", type=int, default=50)
    parser.add_argument("--compute-mincut", action="store_true")
    parser.add_argument("--seed-min-incident", type=int, default=None)
    parser.add_argument("--seed-min-distinct-bases", type=int, default=None)
    parser.add_argument("--seed-require-link", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    rng = np.random.default_rng(cfg.rng_seed)

    slack = float(args.slack) if args.slack is not None else float(cfg.dag_slack)
    max_web_edges = int(args.max_web_edges) if args.max_web_edges is not None else int(cfg.twotree_max_web_edges)

    print("\n==================== CORE STRUCTURE DIAGNOSTICS ====================")
    print(f"slack={slack:.3f}  model={args.model}  n_ods={args.n_ods}  min_trips={args.min_trips}")
    print(f"twotree: max_web_edges={max_web_edges}  seed_max_hops={args.web_max_hops}")

    # Load net and build edge-connection graph
    net = sumolib.net.readNet(cfg.net_path)
    bad_edges = load_bad_edges(cfg.bad_edges_path)
    edge_attrs = build_edge_attributes(net, cfg.allowed_vtypes, bad_edges, cfg.bad_edge_penalty)
    Gfull = build_connection_graph_no_internals(net, edge_attrs, cfg.allowed_vtypes)

    print(f"  Drivable edges (nodes): {len(Gfull.nodes()):,}")
    print(f"  Connection arcs:        {Gfull.number_of_edges():,}")

    # Geodata + BG
    edges_gdf = build_edges_gdf(net, edge_attrs)
    edges_gdf = attach_block_groups(edges_gdf, cfg.bg_path)

    # Load Replica + spawn/despawn
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
        mapped = mapped.sample(frac=cfg.demand_scale, random_state=cfg.rng_seed).reset_index(drop=True)

    # Centroid edges + OD pairs
    bg_centroid_edge = build_bg_centroid_edge_map(edges_gdf, cfg.bg_path, prefer_trunk=False)
    bg_centroid_edge = {str(k).zfill(12): v for k, v in bg_centroid_edge.items()}

    mapped["origin_centroid_edge"] = mapped["origin_bgrp_fips_2020"].map(bg_centroid_edge)
    mapped["dest_centroid_edge"] = mapped["destination_bgrp_fips_2020"].map(bg_centroid_edge)
    mapped = mapped.dropna(subset=["origin_centroid_edge", "dest_centroid_edge", "spawn_edge", "despawn_edge"]).copy()

    if "vtype_key" not in mapped.columns:
        mapped["vtype_key"] = "car"

    od_pairs = (
        mapped.groupby(["origin_centroid_edge", "dest_centroid_edge", "vtype_key"], observed=True)
        .size()
        .reset_index(name="trip_count")
    )
    od_pairs = od_pairs[od_pairs["trip_count"] >= int(args.min_trips)].copy()
    od_pairs = od_pairs.sort_values("trip_count", ascending=False).head(int(args.n_ods)).reset_index(drop=True)

    # CSR build
    nodes = list(Gfull.nodes())
    idx_map = {n: i for i, n in enumerate(nodes)}
    A = build_csr_from_graph(Gfull, nodes)
    indptr = A.indptr.astype(np.int64)
    indices = A.indices.astype(np.int64)
    base_w = A.data.astype(np.float64)

    n = len(nodes)
    Asp = sp.csr_matrix((base_w, indices, indptr), shape=(n, n))
    AT = Asp.transpose().tocsr()

    # Dijkstra precompute for all unique origins/dests
    o_eids = pd.Series(od_pairs["origin_centroid_edge"].astype(str).unique())
    d_eids = pd.Series(od_pairs["dest_centroid_edge"].astype(str).unique())
    o_idx = [idx_map[e] for e in o_eids if e in idx_map]
    d_idx = [idx_map[e] for e in d_eids if e in idx_map]
    o_seen = set()
    o_idx = [i for i in o_idx if (i not in o_seen and not o_seen.add(i))]
    d_seen = set()
    d_idx = [i for i in d_idx if (i not in d_seen and not d_seen.add(i))]

    dist_cache_o: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    dist_cache_to_d: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    print(f"  Precompute Dijkstra: {len(o_idx):,} origins, {len(d_idx):,} dests")
    for oi in tqdm(o_idx, desc="Origin Dijkstra", leave=False):
        dist, pred = cs_dijkstra(Asp, directed=True, indices=[oi], return_predecessors=True)
        dist0 = dist[0].astype(np.float64)
        pred0 = pred[0].astype(np.int64)
        reach0 = np.flatnonzero(np.isfinite(dist0)).astype(np.int64)
        dist_cache_o[int(oi)] = (dist0, pred0, reach0)

    for di in tqdm(d_idx, desc="Dest Dijkstra (rev)", leave=False):
        dist_to, pred_to = cs_dijkstra(AT, directed=True, indices=[di], return_predecessors=True)
        dist_cache_to_d[int(di)] = (dist_to[0].astype(np.float64), pred_to[0].astype(np.int64))

    # per-node decay
    decay_node = np.empty(n, dtype=np.float64)
    for gi in range(n):
        rc = _sumo_roadclass(edge_attrs, nodes[gi])
        decay_node[gi] = float(_DECAY.get(rc, 0.05))

    # reusable buffers
    stamp = np.zeros(n, dtype=np.int32)
    g2l = np.zeros(n, dtype=np.int64)

    # arc-level scratch for webbing bookkeeping (NEW)
    arc_stamp   = np.zeros(indices.shape[0], dtype=np.int32)   # length = #arcs in full CSR
    arc_counter = np.zeros(indices.shape[0], dtype=np.int32)   # length = #arcs in full CSR

    # Seed selection + BFS cache (for twotree)
    seed_min_inc = int(args.seed_min_incident) if args.seed_min_incident is not None else int(cfg.twotree_seed_min_incident)
    seed_min_bases = int(args.seed_min_distinct_bases) if args.seed_min_distinct_bases is not None else int(cfg.twotree_seed_min_distinct_bases)
    seed_req_link = bool(args.seed_require_link) if args.seed_require_link else bool(cfg.twotree_seed_require_link)

    seed_eids = compute_arterial_seed_nodes_from_edgeattrs(
        Gfull, edge_attrs,
        min_incident=seed_min_inc,
        min_distinct_bases=seed_min_bases,
        require_link_present=seed_req_link,
    )
    seeds_idx = np.asarray([idx_map[e] for e in seed_eids if e in idx_map], dtype=np.int64)

    seed_ptr, seed_u, seed_v = build_seed_bfs_tree_cache_hops_numba(
        indptr, indices,
        seeds_idx,
        max_hops=int(args.web_max_hops),
    )

    node_to_seedid = -np.ones(n, dtype=np.int64)
    for sid, sidx in enumerate(seeds_idx):
        node_to_seedid[int(sidx)] = int(sid)

    # Benchmark loop
    rows: List[dict] = []
    for r in tqdm(od_pairs.itertuples(index=False), total=len(od_pairs), desc="OD metrics"):
        o_eid = str(r.origin_centroid_edge)
        d_eid = str(r.dest_centroid_edge)
        trip_count = int(r.trip_count)
        oi = idx_map.get(o_eid, None)
        di = idx_map.get(d_eid, None)
        if oi is None or di is None:
            continue

        dist_o, pred_o, reach_o = dist_cache_o[int(oi)]
        dist_to_d, pred_to = dist_cache_to_d[int(di)]

        # Skip if unreachable
        ssp = float(dist_o[int(di)])
        if not np.isfinite(ssp) or ssp <= 0:
            continue

        # forward-progress
        if args.model in ("both", "forward"):
            od_indptr, od_indices, od_pos, slot_decay, core_nodes, start_loc, end_loc = _build_od_core_csr_numba_layered_viols(
                indptr, indices, base_w,
                dist_o, dist_to_d, reach_o,
                int(oi), int(di), float(slack),
                stamp, g2l, decay_node
            )
            if core_nodes.shape[0] > 0 and int(start_loc) >= 0 and int(end_loc) >= 0:
                met = compute_core_metrics(
                    od_indptr, od_indices, core_nodes, int(start_loc), int(end_loc),
                    dist_o,
                    compute_mincut=bool(args.compute_mincut),
                )
                rows.append({
                    "model": "forward",
                    "o": o_eid, "d": d_eid, "trip_count": trip_count,
                    **met.__dict__,
                })

        # two-tree/webbing
        if args.model in ("both", "twotree"):
            touched = compute_touched_seeds_for_od(
                reach_o, dist_o, dist_to_d, int(di), float(slack), node_to_seedid
            )
            out = build_od_twotree_web_csr_numba(
                int(oi), int(di),
                indptr, indices, base_w,
                dist_o, dist_to_d, reach_o,
                float(slack),
                decay_node,
                stamp, g2l,
                pred_o.astype(np.int64),
                pred_to.astype(np.int64),
                seed_ptr, seed_u, seed_v,
                touched.astype(np.int64),
                arc_stamp, arc_counter,
                int(max_web_edges),
                1e-9,
            )
            (od_indptr2, od_indices2, od_pos2, slot_decay2,
            core_nodes2, start_loc2, end_loc2,
            web_edges_unique, web_edges_attempted,
            ) = out
            if core_nodes2.shape[0] == 0 or int(start_loc2) < 0 or int(end_loc2) < 0:
                print(f"core_nodes2 shape: {core_nodes2.shape}, start_loc2: {int(start_loc2)}, end_loc2: {int(end_loc2)}")
                print(
                    "DAW probe:",
                    f"code={int(arc_stamp[0])}",
                    f"a={int(arc_stamp[1]) if arc_stamp.shape[0] > 1 else 0}",
                    f"b={int(arc_stamp[2]) if arc_stamp.shape[0] > 2 else 0}",
                    f"c={int(arc_stamp[3]) if arc_stamp.shape[0] > 3 else 0}",
                )
                return
            if core_nodes2.shape[0] > 0 and int(start_loc2) >= 0 and int(end_loc2) >= 0:
                met2 = compute_core_metrics(
                    od_indptr2, od_indices2, core_nodes2, int(start_loc2), int(end_loc2),
                    dist_o,
                    compute_mincut=bool(args.compute_mincut),
                )
                rows.append({
                    "model": "twotree",
                    "o": o_eid, "d": d_eid, "trip_count": trip_count,
                    "touched_seeds": int(touched.shape[0]),
                    "web_edges_unique": int(web_edges_unique),
                    "web_edges_attempted": int(web_edges_attempted),
                    **met2.__dict__,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No metrics computed (no reachable ODs?)")
        return

    out_csv = "core_structure_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv} with {len(df):,} rows")

    def summarize(sub: pd.DataFrame, label: str) -> None:
        w = sub["trip_count"].astype(np.float64)
        def wavg(col: str) -> float:
            x = sub[col].astype(np.float64)
            return float((x * w).sum() / max(1.0, w.sum()))
        cols = [
            "nloc", "mloc", "avg_outdeg", "frac_outdeg1",
            "undirected_bridges", "undirected_articulations",
            "dom_nodes_on_all_paths", "dom_edges_on_all_paths",
            "min_edge_cut", "layer_width_entropy",
            "web_edges_unique", "web_edges_attempted",
        ]
        print(f"\n--- {label} (weighted by trip_count) ---")
        for c in cols:
            if c not in sub.columns:
                continue
            if c == "min_edge_cut" and not args.compute_mincut:
                continue
            print(f"{c:26s}  wavg={wavg(c):9.3f}  median={float(sub[c].median()):9.3f}  p90={float(sub[c].quantile(0.90)):9.3f}")

    if args.model in ("both", "forward"):
        summarize(df[df["model"] == "forward"], "forward-progress")
    if args.model in ("both", "twotree"):
        summarize(df[df["model"] == "twotree"], "two-tree/webbing")

    # Quick comparisons
    if args.model == "both":
        merged = df.pivot_table(index=["o","d"], columns="model", values=["nloc","mloc","undirected_bridges","dom_nodes_on_all_paths","layer_width_entropy"], aggfunc="first")
        merged.columns = ["_".join(c) for c in merged.columns.to_flat_index()]
        merged = merged.dropna()
        if not merged.empty:
            print("\n--- Pairwise deltas (twotree - forward), across ODs where both exist ---")
            for c in ["nloc", "mloc", "undirected_bridges", "dom_nodes_on_all_paths", "layer_width_entropy"]:
                a = merged[f"{c}_twotree"] - merged[f"{c}_forward"]
                print(f"delta {c:20s}  mean={float(a.mean()):9.3f}  median={float(a.median()):9.3f}  p90={float(a.quantile(0.90)):9.3f}")

if __name__ == "__main__":
    run()
