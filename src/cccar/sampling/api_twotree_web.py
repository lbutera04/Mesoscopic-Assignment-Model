from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra as cs_dijkstra
from tqdm import tqdm

from ..config import Config
from ..corridor.twotree_web import (
    build_od_twotree_web_csr_numba,
    build_seed_bfs_tree_cache_hops_numba,
    compute_arterial_seed_nodes_from_edgeattrs,
    compute_touched_seeds_for_od,
)
from ..osm.attributes import _DECAY, _sumo_roadclass
from ..routes.build import reconstruct_from_predecessors
from .numba_kernels import _compute_end_pos, _compute_out_sums
from .python_impl import sample_path_uniform_dag


def dag_sample_centroid_od_paths_twotree_web(
    od_pairs: pd.DataFrame,
    nodes: list[str],
    idx: dict[str, int],
    G,
    indptr: np.ndarray,
    indices: np.ndarray,
    base_w: np.ndarray,
    edge_attrs: dict[str, dict],
    cfg: Config,
    rng: np.random.Generator,
) -> dict[tuple[str, str], list[tuple[list[str], int]]]:
    """
    Two-tree HALO + webbing model integrated from tools/dag_benchmarks.py.
    """
    n = len(nodes)
    A = sp.csr_matrix((base_w, indices, indptr), shape=(n, n))
    AT = A.transpose().tocsr()

    o_eids = pd.Series(od_pairs["origin_centroid_edge"].astype(str).unique())
    d_eids = pd.Series(od_pairs["dest_centroid_edge"].astype(str).unique())

    o_idx = [idx[e] for e in o_eids if e in idx]
    d_idx = [idx[e] for e in d_eids if e in idx]

    o_seen = set()
    o_idx = [i for i in o_idx if (i not in o_seen and not o_seen.add(i))]
    d_seen = set()
    d_idx = [i for i in d_idx if (i not in d_seen and not d_seen.add(i))]

    dist_cache_o: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    dist_cache_to_d: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    print(f"  Precompute Dijkstra: {len(o_idx):,} unique origins, {len(d_idx):,} unique destinations")

    for oi in tqdm(o_idx, desc="Precompute origin Dijkstra", leave=False):
        dist, pred = cs_dijkstra(A, directed=True, indices=[oi], return_predecessors=True)
        dist0 = dist[0].astype(np.float64)
        pred0 = pred[0].astype(np.int64)
        reach0 = np.flatnonzero(np.isfinite(dist0)).astype(np.int64)
        dist_cache_o[int(oi)] = (dist0, pred0, reach0)

    for di in tqdm(d_idx, desc="Precompute dest Dijkstra (reverse)", leave=False):
        dist_to, pred_to = cs_dijkstra(AT, directed=True, indices=[di], return_predecessors=True)
        dist_cache_to_d[int(di)] = (dist_to[0].astype(np.float64), pred_to[0].astype(np.int64))

    decay_node = np.empty(n, dtype=np.float64)
    for gi in range(n):
        eid = nodes[gi]
        rc = _sumo_roadclass(edge_attrs, eid)
        decay_node[gi] = float(_DECAY.get(rc, 0.05))

    print("  Building arterial seed cache for webbing")
    seed_eids = compute_arterial_seed_nodes_from_edgeattrs(
        G,
        edge_attrs,
        min_incident=int(cfg.twotree_seed_min_incident),
        min_distinct_bases=int(cfg.twotree_seed_min_distinct_bases),
        require_link_present=bool(cfg.twotree_seed_require_link),
    )
    seed_idx = np.asarray([idx[e] for e in seed_eids if e in idx], dtype=np.int64)

    node_to_seedid = -np.ones(n, dtype=np.int64)
    for sid, gi in enumerate(seed_idx):
        node_to_seedid[int(gi)] = sid

    if seed_idx.size > 0:
        seed_ptr, seed_u, seed_v = build_seed_bfs_tree_cache_hops_numba(
            indptr,
            indices,
            seed_idx,
            max_hops=int(cfg.twotree_seed_bfs_max_hops),
        )
    else:
        seed_ptr = np.zeros(1, dtype=np.int64)
        seed_u = np.zeros(0, dtype=np.int64)
        seed_v = np.zeros(0, dtype=np.int64)

    print(
        "  Two-tree seed cache: "
        f"{seed_idx.size:,} seeds, {seed_u.size:,} cached tree arcs "
        f"(max_hops={cfg.twotree_seed_bfs_max_hops})"
    )

    global_toll = np.ones(len(base_w), dtype=np.float64)

    stamp = np.zeros(n, dtype=np.int32)
    g2l = np.zeros(n, dtype=np.int64)

    arc_stamp = np.zeros(len(indices), dtype=np.int32)
    arc_counter = np.zeros(1, dtype=np.int32)

    od_paths: dict[tuple[str, str], list[tuple[list[str], int]]] = {}

    for row in tqdm(od_pairs.itertuples(index=False), total=len(od_pairs), desc="Centroid OD DAG sampling (two-tree/web)"):
        o_eid = str(row.origin_centroid_edge)
        d_eid = str(row.dest_centroid_edge)
        trips = int(row.trip_count)
        key = (o_eid, d_eid)

        if o_eid not in idx or d_eid not in idx:
            continue

        oi = int(idx[o_eid])
        di = int(idx[d_eid])

        if oi not in dist_cache_o or di not in dist_cache_to_d:
            continue

        dist_o, pred_o, reach_o = dist_cache_o[oi]
        dist_to_d, pred_to_d = dist_cache_to_d[di]

        ssp = float(dist_o[di])
        if not np.isfinite(ssp):
            continue

        if seed_idx.size > 0:
            touched_seeds = compute_touched_seeds_for_od(
                reach_o,
                dist_o,
                dist_to_d,
                di,
                float(cfg.dag_slack),
                node_to_seedid,
            )
        else:
            touched_seeds = np.empty(0, dtype=np.int64)

        od_indptr, od_indices, od_pos, slot_decay, core_nodes, start_loc, end_loc = build_od_twotree_web_csr_numba(
            oi,
            di,
            indptr,
            indices,
            base_w,
            dist_o,
            dist_to_d,
            reach_o,
            float(cfg.dag_slack),
            decay_node,
            stamp,
            g2l,
            pred_o,
            pred_to_d,
            seed_ptr,
            seed_u,
            seed_v,
            touched_seeds,
            arc_stamp,
            arc_counter,
            int(cfg.twotree_max_web_edges),
        )

        if len(od_indices) == 0 or start_loc < 0 or end_loc < 0:
            ssp_path_idx = reconstruct_from_predecessors(pred_o, oi, di)
            if not ssp_path_idx:
                continue
            seq = [nodes[i] for i in ssp_path_idx]
            od_paths[key] = [(seq, trips)]
            continue

        K = max(1, int(cfg.routes_per_od))
        assigned = [trips // K] * K
        for i in range(trips % K):
            assigned[i] += 1

        mloc = int(len(od_pos))
        nloc = int(len(od_indptr) - 1)

        if "toll_buf" not in locals():
            toll_buf = np.empty(mloc, dtype=np.float64)
            out_sum = np.empty(nloc, dtype=np.float64)
            end_pos = np.empty(nloc, dtype=np.int64)
            path_nodes_buf = np.empty(nloc + 1, dtype=np.int64)
            used_slots_buf = np.empty(nloc, dtype=np.int64)
        else:
            if toll_buf.shape[0] < mloc:
                toll_buf = np.empty(mloc, dtype=np.float64)
            if out_sum.shape[0] != nloc:
                out_sum = np.empty(nloc, dtype=np.float64)
            if end_pos.shape[0] != nloc:
                end_pos = np.empty(nloc, dtype=np.int64)
            if path_nodes_buf.shape[0] < (nloc + 1):
                path_nodes_buf = np.empty(nloc + 1, dtype=np.int64)
            if used_slots_buf.shape[0] < nloc:
                used_slots_buf = np.empty(nloc, dtype=np.int64)

        np.copyto(toll_buf[:mloc], global_toll[od_pos])

        _compute_out_sums(od_indptr, toll_buf[:mloc], out_sum)
        _compute_end_pos(od_indptr, od_indices, int(end_loc), end_pos)

        alpha_global = 0.00
        plist: list[tuple[list[str], int]] = []

        for k in range(K):
            path_arr, used_arr, plen, ulen = sample_path_uniform_dag(
                od_indptr,
                od_indices,
                toll_buf[:mloc],
                slot_decay,
                start_loc,
                end_loc,
                rng=rng,
                max_steps=nloc,
                out_sum=out_sum,
                end_pos=end_pos,
                path_nodes_buf=path_nodes_buf,
                used_slots_buf=used_slots_buf,
            )

            if plen <= 0:
                continue

            if ulen > 0:
                used_global_pos = od_pos[used_arr[:ulen]]
                global_toll[used_global_pos] += alpha_global / (1.0 + global_toll[used_global_pos])

            loc_path = path_arr[:plen]
            gl_path = [int(core_nodes[int(i)]) for i in loc_path]
            seq = [nodes[i] for i in gl_path]

            plist.append((seq, int(assigned[k])))

        if not plist:
            ssp_path_idx = reconstruct_from_predecessors(pred_o, oi, di)
            if not ssp_path_idx:
                continue
            seq = [nodes[i] for i in ssp_path_idx]
            od_paths[key] = [(seq, trips)]
            continue

        od_paths[key] = plist

        if cfg.debug_print_examples > 0 and len(od_paths) <= cfg.debug_print_examples:
            print(
                f"\n[DEBUG] two-tree/web OD {o_eid}->{d_eid}: "
                f"SSP={ssp:.1f}s, core_nodes={len(core_nodes):,}, core_arcs={len(od_indices):,}, "
                f"touched_seeds={len(touched_seeds):,}"
            )

    return od_paths
