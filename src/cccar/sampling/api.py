import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm
from scipy.sparse.csgraph import dijkstra as cs_dijkstra

from config import Config
from corridor.core import _precompute_incoming, _build_od_core_csr_numba_fast
from osm.attributes import _sumo_roadclass, _DECAY
from routes.build import reconstruct_from_predecessors
from .numba_kernels import _compute_out_sums, _compute_end_pos
from .python_impl import sample_path_uniform_dag

def dag_sample_centroid_od_paths(
    od_pairs: pd.DataFrame,
    nodes: list[str],
    idx: dict[str, int],
    indptr: np.ndarray,
    indices: np.ndarray,
    base_w: np.ndarray,
    edge_attrs: dict[str, dict],
    cfg: Config,
    rng: np.random.Generator,
) -> dict[tuple[str, str], list[tuple[list[str], int]]]:
    """
    For each centroid OD, build the mixed-potential feasibility DAG and sample K routes.
    Key rule: NEVER recompute Dijkstra per-OD. Precompute once for all unique origins + destinations.
    """
    n = len(nodes)
    A = sp.csr_matrix((base_w, indices, indptr), shape=(n, n))
    AT = A.transpose().tocsr()

    # For fast backward core computation
    in_indptr, in_pos, src_of_pos = _precompute_incoming(indptr, indices)

    # -------------------------------------------------------------------------
    # Precompute Dijkstra once per unique origin and once per unique destination
    # -------------------------------------------------------------------------
    # Collect unique centroid-edge IDs that are present in the graph index.
    o_eids = pd.Series(od_pairs["origin_centroid_edge"].astype(str).unique())
    d_eids = pd.Series(od_pairs["dest_centroid_edge"].astype(str).unique())

    o_idx = [idx[e] for e in o_eids if e in idx]
    d_idx = [idx[e] for e in d_eids if e in idx]

    # Deduplicate while preserving order
    o_seen = set()
    o_idx = [i for i in o_idx if (i not in o_seen and not o_seen.add(i))]
    d_seen = set()
    d_idx = [i for i in d_idx if (i not in d_seen and not d_seen.add(i))]

    dist_cache_o: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    reach_cache_o: dict[int, np.ndarray] = {}
    dist_cache_to_d: dict[int, np.ndarray] = {}

    print(f"  Precompute Dijkstra: {len(o_idx):,} unique origins, {len(d_idx):,} unique destinations")

    for oi in tqdm(o_idx, desc="Precompute origin Dijkstra", leave=False):
        dist, pred = cs_dijkstra(A, directed=True, indices=[oi], return_predecessors=True)
        dist0 = dist[0].astype(np.float64)
        dist_cache_o[int(oi)] = (dist0, pred[0].astype(np.int64))
        # reachable nodes from this origin (avoid O(n) scans per OD)
        reach_cache_o[int(oi)] = np.flatnonzero(np.isfinite(dist0)).astype(np.int64)

    for di in tqdm(d_idx, desc="Precompute dest Dijkstra (reverse)", leave=False):
        dist_to = cs_dijkstra(AT, directed=True, indices=[di], return_predecessors=False)
        dist_cache_to_d[int(di)] = dist_to[0].astype(np.float64)


    # Precompute per-node decay (used for intra-OD arc toll decay).
    # This must be defined outside the OD loop so we never do string lookups per-OD.
    decay_node = np.empty(n, dtype=np.float64)
    for gi in range(n):
        eid = nodes[gi]
        rc = _sumo_roadclass(edge_attrs, eid)
        decay_node[gi] = float(_DECAY.get(rc, 0.05))

    global_toll = np.ones(len(base_w), dtype=np.float64)

    # Reused per-OD work buffers (avoid O(n) resets)
    stamp = np.zeros(n, dtype=np.int32)
    g2l = np.zeros(n, dtype=np.int64)
    cur_stamp = np.int32(1)


    alpha_global = 0.05   # keep small! (tune later)

    # -------------------------------------------------------------------------
    # Per-OD DAG corridor build + sampling (cheap compared to all-pairs Dijkstra)
    # -------------------------------------------------------------------------
    od_paths: dict[tuple[str, str], list[tuple[list[str], int]]] = {}

    for row in tqdm(od_pairs.itertuples(index=False), total=len(od_pairs), desc="Centroid OD DAG sampling"):
        o_eid = str(row.origin_centroid_edge)
        d_eid = str(row.dest_centroid_edge)
        trips = int(row.trip_count)
        key = (o_eid, d_eid)

        if o_eid not in idx or d_eid not in idx:
            continue
        oi = int(idx[o_eid])
        di = int(idx[d_eid])

        # Dijkstra results are precomputed; if missing, skip.
        if oi not in dist_cache_o or di not in dist_cache_to_d:
            continue
        dist_o, pred_o = dist_cache_o[oi]
        dist_to_d = dist_cache_to_d[di]

        ssp = float(dist_o[di])
        if not np.isfinite(ssp):
            continue

        
        # Build OD-core CSR directly (Numba) — no per-OD keep-mask, no Python core compression.
        od_indptr, od_indices, od_pos, slot_decay, core_nodes, start_loc, end_loc = _build_od_core_csr_numba_fast(
            oi, di, indptr, indices, base_w, dist_o, dist_to_d,
            reach_cache_o[int(oi)], cfg.dag_slack, decay_node,
            stamp, g2l, int(cur_stamp)
        )
        cur_stamp = np.int32(cur_stamp + 1)
        if cur_stamp == 0:
            stamp[:] = 0
            cur_stamp = np.int32(1)

        # If corridor is empty, fall back to SSP predecessor path
        if len(od_indices) == 0 or start_loc < 0 or end_loc < 0:
            ssp_path_idx = reconstruct_from_predecessors(pred_o, oi, di)
            if not ssp_path_idx:
                continue
            seq = [nodes[i] for i in ssp_path_idx]
            od_paths[key] = [(seq, trips)]
            continue

        # Sample K routes with intra-OD decay

        K = max(1, int(cfg.routes_per_od))
        assigned = [trips // K] * K
        for i in range(trips % K):
            assigned[i] += 1

        
        # -----------------------------
        # Reuse buffers to reduce allocs
        # -----------------------------
        mloc = int(len(od_pos))
        nloc = int(len(od_indptr) - 1)

        # Reuse/resize toll buffer
        if "toll_buf" not in locals():
            toll_buf = np.empty(mloc, dtype=np.float64)
            out_sum = np.empty(nloc, dtype=np.float64)
            end_pos = np.empty(nloc, dtype=np.int64)
            # sample buffers (sized to nloc, since DAG walk can't exceed nloc steps)
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

        # Initialize toll for this OD (same as copy(), but reuses buffer)
        np.copyto(toll_buf[:mloc], global_toll[od_pos])

        # Precompute per-node outgoing sums and end-edge positions once per OD
        _compute_out_sums(od_indptr, toll_buf[:mloc], out_sum)
        _compute_end_pos(od_indptr, od_indices, int(end_loc), end_pos)

        plist: list[tuple[list[str], int]] = []
        for k in range(K):
            # Walk using precomputed arrays and preallocated buffers.
            # max_steps is effectively nloc due to buffer sizing.
            path_arr, used_arr, plen, ulen = sample_path_uniform_dag(
                od_indptr, od_indices,
                toll_buf[:mloc], slot_decay,
                start_loc, end_loc,
                rng=rng,
                max_steps=nloc,
                out_sum=out_sum,
                end_pos=end_pos,
                path_nodes_buf=path_nodes_buf,
                used_slots_buf=used_slots_buf,
            )

            if plen <= 0:
                continue

            # Convert OD-local arc slots → global arc positions
            if ulen > 0:
                used_global_pos = od_pos[used_arr[:ulen]]
                global_toll[used_global_pos] += alpha_global / (1.0 + global_toll[used_global_pos])

            # path_arr contains OD-local node ids; map to global graph node ids
            loc_path = path_arr[:plen]
            gl_path = [int(core_nodes[int(i)]) for i in loc_path]
            seq = [nodes[i] for i in gl_path]

            plist.append((seq, int(assigned[k])))


        od_paths[key] = plist

        if cfg.debug_print_examples > 0 and len(od_paths) <= cfg.debug_print_examples:
            print(f"\n[DEBUG] OD {o_eid}->{d_eid}: SSP={ssp:.1f}s, core_nodes={len(core_nodes):,}, core_arcs={len(od_indices):,}")

    return od_paths
