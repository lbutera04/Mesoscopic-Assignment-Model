#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra as cs_dijkstra
from tqdm import tqdm
import sumolib

from ..config import Config
from ..osm.attributes import load_bad_edges, build_edge_attributes, _sumo_roadclass, _DECAY
from ..osm.graph_build import build_connection_graph_no_internals, build_csr_from_graph
from ..osm.geo import build_edges_gdf, attach_block_groups
from ..demand.replica import load_replica, map_replica_to_edges
from ..demand.spawns import build_spawn_tables
from ..demand.centroids import build_bg_centroid_edge_map

from ..corridor.core import _build_od_core_csr_numba_fast
from ..corridor.twotree_web import compute_arterial_seed_nodes_from_edgeattrs
from ..corridor.DAW import (
    build_od_twotree_web_csr_numba,
    build_seed_bfs_tree_cache_hops_numba,
    compute_touched_seeds_for_od,
)


def _build_repo_inputs(cfg: Config, rng: np.random.Generator):
    net = sumolib.net.readNet(cfg.net_path)
    bad_edges = load_bad_edges(cfg.bad_edges_path)
    edge_attrs = build_edge_attributes(net, cfg.allowed_vtypes, bad_edges, cfg.bad_edge_penalty)
    Gfull = build_connection_graph_no_internals(net, edge_attrs, cfg.allowed_vtypes)

    print(f"  Drivable edges (nodes): {len(Gfull.nodes()):,}")
    print(f"  Connection arcs:        {Gfull.number_of_edges():,}")

    edges_gdf = build_edges_gdf(net, edge_attrs)
    edges_gdf = attach_block_groups(edges_gdf, cfg.bg_path)

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

    mapped = mapped[
        mapped["origin_bgrp_fips_2020"] != mapped["destination_bgrp_fips_2020"]
    ].copy()

    if cfg.demand_scale < 1.0:
        mapped = mapped.sample(frac=cfg.demand_scale, random_state=cfg.rng_seed).reset_index(drop=True)

    bg_centroid_edge = build_bg_centroid_edge_map(edges_gdf, cfg.bg_path, prefer_trunk=False)
    bg_centroid_edge = {str(k).zfill(12): v for k, v in bg_centroid_edge.items()}

    mapped["origin_centroid_edge"] = mapped["origin_bgrp_fips_2020"].map(bg_centroid_edge)
    mapped["dest_centroid_edge"] = mapped["destination_bgrp_fips_2020"].map(bg_centroid_edge)
    mapped = mapped.dropna(
        subset=["origin_centroid_edge", "dest_centroid_edge", "spawn_edge", "despawn_edge"]
    ).copy()

    if "vtype_key" not in mapped.columns:
        mapped["vtype_key"] = "car"

    nodes = list(Gfull.nodes())
    idx_map = {n: i for i, n in enumerate(nodes)}

    A = build_csr_from_graph(Gfull, nodes)
    indptr = A.indptr.astype(np.int64)
    indices = A.indices.astype(np.int64)
    base_w = A.data.astype(np.float64)

    n = len(nodes)
    Asp = sp.csr_matrix((base_w, indices, indptr), shape=(n, n))
    AT = Asp.transpose().tocsr()

    decay_node = np.empty(n, dtype=np.float64)
    for gi in range(n):
        rc = _sumo_roadclass(edge_attrs, nodes[gi])
        decay_node[gi] = float(_DECAY.get(rc, 0.05))

    return (
        Gfull,
        edge_attrs,
        mapped,
        nodes,
        idx_map,
        indptr,
        indices,
        base_w,
        Asp,
        AT,
        decay_node,
    )


def _build_od_pairs(mapped: pd.DataFrame, min_trips: int, n_ods: int) -> pd.DataFrame:
    od_pairs = (
        mapped.groupby(
            ["origin_centroid_edge", "dest_centroid_edge", "vtype_key"],
            observed=True,
        )
        .size()
        .reset_index(name="trip_count")
    )
    od_pairs = od_pairs[od_pairs["trip_count"] >= int(min_trips)].copy()
    od_pairs = od_pairs.sort_values("trip_count", ascending=False).head(int(n_ods)).reset_index(drop=True)
    return od_pairs


def _precompute_dijkstra_caches(
    od_pairs: pd.DataFrame,
    idx_map: Dict[str, int],
    Asp: sp.csr_matrix,
    AT: sp.csr_matrix,
):
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

    return dist_cache_o, dist_cache_to_d


def _build_benchmark_od_list(
    od_pairs: pd.DataFrame,
    idx_map: Dict[str, int],
    dist_cache_o: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    dist_cache_to_d: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> Tuple[List[Tuple[int, int]], List[int]]:
    bench_pairs: List[Tuple[int, int]] = []
    bench_trips: List[int] = []

    for row in od_pairs.itertuples(index=False):
        o_eid = str(row.origin_centroid_edge)
        d_eid = str(row.dest_centroid_edge)

        if o_eid not in idx_map or d_eid not in idx_map:
            continue

        oi = int(idx_map[o_eid])
        di = int(idx_map[d_eid])

        if oi not in dist_cache_o or di not in dist_cache_to_d:
            continue

        dist_o, _, _ = dist_cache_o[oi]
        if not np.isfinite(dist_o[di]):
            continue

        bench_pairs.append((oi, di))
        bench_trips.append(int(row.trip_count))

    if not bench_pairs:
        raise RuntimeError("No feasible benchmark ODs after filtering.")

    return bench_pairs, bench_trips


def _warmup_forward(
    bench_pairs: List[Tuple[int, int]],
    dist_cache_o: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    dist_cache_to_d: Dict[int, Tuple[np.ndarray, np.ndarray]],
    indptr: np.ndarray,
    indices: np.ndarray,
    base_w: np.ndarray,
    decay_node: np.ndarray,
    slack: float,
    stamp: np.ndarray,
    g2l: np.ndarray,
):
    cur_stamp = np.int32(1)

    for oi, di in bench_pairs:
        dist_o, _, reach_o = dist_cache_o[oi]
        dist_to_d, _ = dist_cache_to_d[di]
        if np.isfinite(dist_o[di]):
            _build_od_core_csr_numba_fast(
                int(oi),
                int(di),
                indptr,
                indices,
                base_w,
                dist_o,
                dist_to_d,
                reach_o,
                float(slack),
                decay_node,
                stamp,
                g2l,
                int(cur_stamp),
            )
            return

    raise RuntimeError("Could not find feasible warm-up OD for forward model.")


def _warmup_twotree(
    bench_pairs: List[Tuple[int, int]],
    dist_cache_o: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    dist_cache_to_d: Dict[int, Tuple[np.ndarray, np.ndarray]],
    indptr: np.ndarray,
    indices: np.ndarray,
    base_w: np.ndarray,
    decay_node: np.ndarray,
    slack: float,
    stamp: np.ndarray,
    g2l: np.ndarray,
    seed_ptr: np.ndarray,
    seed_u: np.ndarray,
    seed_v: np.ndarray,
    arc_stamp: np.ndarray,
    arc_counter: np.ndarray,
    max_web_edges: int,
):
    touched0 = np.zeros(0, dtype=np.int64)

    for oi, di in bench_pairs:
        dist_o, pred_o, reach_o = dist_cache_o[oi]
        dist_to_d, pred_to = dist_cache_to_d[di]
        if np.isfinite(dist_o[di]):
            build_od_twotree_web_csr_numba(
                int(oi),
                int(di),
                indptr,
                indices,
                base_w,
                dist_o,
                dist_to_d,
                reach_o,
                float(slack),
                decay_node,
                stamp,
                g2l,
                pred_o,
                pred_to,
                seed_ptr,
                seed_u,
                seed_v,
                touched0,
                arc_stamp,
                arc_counter,
                int(max_web_edges),
            )
            return

    raise RuntimeError("Could not find feasible warm-up OD for twotree model.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark production forward OD-core builder vs production DAW/twotree builder."
    )
    parser.add_argument("--n_ods", type=int, default=1000, help="How many OD pairs to benchmark.")
    parser.add_argument("--bench_reps", type=int, default=3, help="Timing repetitions.")
    parser.add_argument("--min_trips", type=int, default=1, help="Minimum OD trip count.")
    parser.add_argument("--slack", type=float, default=None, help="Override cfg.dag_slack.")
    parser.add_argument("--max_web_edges", type=int, default=None, help="Override cfg.twotree_max_web_edges.")
    parser.add_argument("--web_max_hops", type=int, default=50, help="Hop radius for cached seed BFS trees.")
    parser.add_argument("--seed_min_incident", type=int, default=None)
    parser.add_argument("--seed_min_distinct_bases", type=int, default=None)
    parser.add_argument("--seed_require_link", action="store_true")
    parser.add_argument("--write_csv", type=str, default=None, help="Optional path for per-OD benchmark rows.")
    args = parser.parse_args()

    cfg = Config()
    rng = np.random.default_rng(cfg.rng_seed)

    slack = float(args.slack) if args.slack is not None else float(cfg.dag_slack)
    max_web_edges = (
        int(args.max_web_edges)
        if args.max_web_edges is not None
        else int(cfg.twotree_max_web_edges)
    )

    print("\n==================== REAL MODEL DAG BENCHMARK ====================")
    print(
        f"n_ods={args.n_ods}  reps={args.bench_reps}  "
        f"min_trips={args.min_trips}  slack={slack:.3f}"
    )
    print(
        f"twotree: max_web_edges={max_web_edges}  "
        f"web_max_hops={args.web_max_hops}"
    )

    (
        Gfull,
        edge_attrs,
        mapped,
        nodes,
        idx_map,
        indptr,
        indices,
        base_w,
        Asp,
        AT,
        decay_node,
    ) = _build_repo_inputs(cfg, rng)

    od_pairs = _build_od_pairs(mapped, args.min_trips, args.n_ods)
    print(f"  Candidate OD rows after filtering: {len(od_pairs):,}")

    dist_cache_o, dist_cache_to_d = _precompute_dijkstra_caches(
        od_pairs, idx_map, Asp, AT
    )

    bench_pairs, bench_trips = _build_benchmark_od_list(
        od_pairs, idx_map, dist_cache_o, dist_cache_to_d
    )
    print(f"  Feasible benchmark ODs used:      {len(bench_pairs):,}")

    n = len(nodes)
    m = len(indices)

    # Forward builder scratch
    stamp_forward = np.zeros(n, dtype=np.int32)
    g2l_forward = np.zeros(n, dtype=np.int64)

    # Twotree builder scratch
    stamp_twotree = np.zeros(n, dtype=np.int32)
    g2l_twotree = np.zeros(n, dtype=np.int64)
    arc_stamp = np.zeros(m, dtype=np.int32)
    arc_counter = np.zeros(m, dtype=np.int32)

    # Seed selection + BFS cache
    seed_min_inc = (
        int(args.seed_min_incident)
        if args.seed_min_incident is not None
        else int(cfg.twotree_seed_min_incident)
    )
    seed_min_bases = (
        int(args.seed_min_distinct_bases)
        if args.seed_min_distinct_bases is not None
        else int(cfg.twotree_seed_min_distinct_bases)
    )
    seed_req_link = bool(args.seed_require_link) if args.seed_require_link else bool(cfg.twotree_seed_require_link)

    print("  Building arterial seed cache")
    seed_eids = compute_arterial_seed_nodes_from_edgeattrs(
        Gfull,
        edge_attrs,
        min_incident=seed_min_inc,
        min_distinct_bases=seed_min_bases,
        require_link_present=seed_req_link,
    )
    seeds_idx = np.asarray([idx_map[e] for e in seed_eids if e in idx_map], dtype=np.int64)

    seed_ptr, seed_u, seed_v = build_seed_bfs_tree_cache_hops_numba(
        indptr,
        indices,
        seeds_idx,
        max_hops=int(args.web_max_hops),
    )

    node_to_seedid = -np.ones(n, dtype=np.int64)
    for sid, sidx in enumerate(seeds_idx):
        node_to_seedid[int(sidx)] = int(sid)

    print(f"  Seeds selected:                   {len(seeds_idx):,}")
    print(f"  Cached seed tree arcs:            {int(seed_ptr[-1]):,}")

    print("  Precomputing touched seed sets per OD")
    bench_touched: List[np.ndarray] = []
    for oi, di in tqdm(bench_pairs, desc="Touched seeds", leave=False):
        dist_o, _, reach_o = dist_cache_o[oi]
        dist_to_d, _ = dist_cache_to_d[di]
        touched = compute_touched_seeds_for_od(
            reach_o,
            dist_o,
            dist_to_d,
            int(di),
            float(slack),
            node_to_seedid,
        )
        bench_touched.append(np.ascontiguousarray(touched, dtype=np.int64))

    print("  Warming kernels")
    _warmup_forward(
        bench_pairs,
        dist_cache_o,
        dist_cache_to_d,
        indptr,
        indices,
        base_w,
        decay_node,
        slack,
        stamp_forward,
        g2l_forward,
    )
    _warmup_twotree(
        bench_pairs,
        dist_cache_o,
        dist_cache_to_d,
        indptr,
        indices,
        base_w,
        decay_node,
        slack,
        stamp_twotree,
        g2l_twotree,
        seed_ptr,
        seed_u,
        seed_v,
        arc_stamp,
        arc_counter,
        max_web_edges,
    )

    per_od_rows: List[dict] = []

    print("\n==================== BENCHMARK ====================")
    for rep in range(int(args.bench_reps)):
        # ------------------------
        # forward (production)
        # ------------------------
        t0 = time.perf_counter()
        acc_forward = 0
        cur_stamp_forward = np.int32(1)

        for j, (oi, di) in enumerate(bench_pairs):
            dist_o, _, reach_o = dist_cache_o[oi]
            dist_to_d, _ = dist_cache_to_d[di]

            out = _build_od_core_csr_numba_fast(
                int(oi),
                int(di),
                indptr,
                indices,
                base_w,
                dist_o,
                dist_to_d,
                reach_o,
                float(slack),
                decay_node,
                stamp_forward,
                g2l_forward,
                int(cur_stamp_forward),
            )
            (
                od_indptr,
                od_indices,
                od_pos,
                slot_decay,
                core_nodes,
                start_loc,
                end_loc,
            ) = out

            cur_stamp_forward = np.int32(cur_stamp_forward + 1)
            if cur_stamp_forward == 0:
                stamp_forward[:] = 0
                cur_stamp_forward = np.int32(1)

            nloc = int(core_nodes.shape[0])
            mloc = int(od_indices.shape[0])
            acc_forward += nloc + mloc

            per_od_rows.append(
                {
                    "rep": rep + 1,
                    "model": "forward",
                    "origin_idx": int(oi),
                    "dest_idx": int(di),
                    "trip_count": int(bench_trips[j]),
                    "nloc": nloc,
                    "mloc": mloc,
                    "web_edges_unique": 0,
                    "web_edges_attempted": 0,
                    "valid": int(nloc > 0 and int(start_loc) >= 0 and int(end_loc) >= 0),
                }
            )

        t1 = time.perf_counter()
        dt_forward = t1 - t0
        print(
            f"[rep {rep + 1}] forward   "
            f"{len(bench_pairs) / dt_forward:10.2f} iters/sec   "
            f"time={dt_forward:8.3f}s   acc={acc_forward}"
        )

        # ------------------------
        # twotree / DAW
        # ------------------------
        t0 = time.perf_counter()
        acc_twotree = 0

        for j, (oi, di) in enumerate(bench_pairs):
            dist_o, pred_o, reach_o = dist_cache_o[oi]
            dist_to_d, pred_to = dist_cache_to_d[di]
            touched_seeds = bench_touched[j]

            out = build_od_twotree_web_csr_numba(
                int(oi),
                int(di),
                indptr,
                indices,
                base_w,
                dist_o,
                dist_to_d,
                reach_o,
                float(slack),
                decay_node,
                stamp_twotree,
                g2l_twotree,
                pred_o,
                pred_to,
                seed_ptr,
                seed_u,
                seed_v,
                touched_seeds,
                arc_stamp,
                arc_counter,
                int(max_web_edges),
            )
            (
                od_indptr,
                od_indices,
                od_pos,
                slot_decay,
                core_nodes,
                start_loc,
                end_loc,
                web_edges_unique,
                web_edges_attempted,
            ) = out

            nloc = int(core_nodes.shape[0])
            mloc = int(od_indices.shape[0])
            acc_twotree += nloc + mloc

            per_od_rows.append(
                {
                    "rep": rep + 1,
                    "model": "twotree",
                    "origin_idx": int(oi),
                    "dest_idx": int(di),
                    "trip_count": int(bench_trips[j]),
                    "nloc": nloc,
                    "mloc": mloc,
                    "web_edges_unique": int(web_edges_unique),
                    "web_edges_attempted": int(web_edges_attempted),
                    "valid": int(nloc > 0 and int(start_loc) >= 0 and int(end_loc) >= 0),
                }
            )

        t1 = time.perf_counter()
        dt_twotree = t1 - t0
        print(
            f"[rep {rep + 1}] twotree   "
            f"{len(bench_pairs) / dt_twotree:10.2f} iters/sec   "
            f"time={dt_twotree:8.3f}s   acc={acc_twotree}"
        )

    df = pd.DataFrame(per_od_rows)

    print("\n==================== SUMMARY BY MODEL ====================")
    summary = (
        df.groupby("model", observed=True)
        .agg(
            reps=("rep", "nunique"),
            ods=("origin_idx", "count"),
            mean_nloc=("nloc", "mean"),
            median_nloc=("nloc", "median"),
            mean_mloc=("mloc", "mean"),
            median_mloc=("mloc", "median"),
            mean_web_edges_unique=("web_edges_unique", "mean"),
            mean_web_edges_attempted=("web_edges_attempted", "mean"),
            valid_share=("valid", "mean"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))

    print("\n==================== SUMMARY BY REP ====================")
    rep_summary = (
        df.groupby(["rep", "model"], observed=True)
        .agg(
            ods=("origin_idx", "count"),
            total_nloc=("nloc", "sum"),
            total_mloc=("mloc", "sum"),
            mean_nloc=("nloc", "mean"),
            mean_mloc=("mloc", "mean"),
            mean_web_edges_unique=("web_edges_unique", "mean"),
            mean_web_edges_attempted=("web_edges_attempted", "mean"),
        )
        .reset_index()
    )
    print(rep_summary.to_string(index=False))

    if args.write_csv:
        df.to_csv(args.write_csv, index=False)
        print(f"\nWrote per-OD benchmark rows to: {args.write_csv}")

    print("\nDONE.\n")


if __name__ == "__main__":
    main()