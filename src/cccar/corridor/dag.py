import numpy as np
from core import _reachable_forward, _reachable_backward

def build_dag_corridor_for_od(
    origin_i: int,
    dest_i: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    w: np.ndarray,
    dist_o: np.ndarray,
    dist_to_d: np.ndarray,
    slack: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      core_mask[n] : uint8 (1 if in core)  [currently unused by caller]
      keep_pos[m]  : bool  (1 if arc position p is kept)

    IMPORTANT: This must be *corridor-sized*, not O(|E|) per OD.
    We prune to candidate nodes first, then only scan outgoing arcs from those nodes.
    """
    ssp = float(dist_o[dest_i])
    m = len(indices)
    n = len(indptr) - 1
    keep = np.zeros(m, dtype=np.bool_)

    if (not np.isfinite(ssp)) or ssp <= 0.0:
        return np.zeros(n, dtype=np.uint8), keep

    budget = slack * ssp

    # Candidate nodes: must be reachable from origin and able to reach dest
    # and must satisfy the slack budget lower bound dist_o + dist_to_d <= budget.
    cand = np.isfinite(dist_o) & np.isfinite(dist_to_d) & ((dist_o + dist_to_d) <= budget)

    # Mixed potential F = dist_o - dist_to_d; define only where finite to avoid inf-inf -> nan.
    F = np.full(n, np.inf, dtype=np.float64)
    mask = np.isfinite(dist_o) & np.isfinite(dist_to_d)
    F[mask] = dist_o[mask] - dist_to_d[mask]

    cand_nodes = np.nonzero(cand)[0]
    for u in cand_nodes:
        gu = float(dist_o[u])
        Fu = float(F[u])
        s = int(indptr[u]); e = int(indptr[u + 1])
        for p in range(s, e):
            v = int(indices[p])
            if not cand[v]:
                continue
            # Slack feasibility
            if gu + float(w[p]) + float(dist_to_d[v]) > budget:
                continue
            # Strict mixed-potential monotonicity => DAG
            if float(F[v]) > Fu:
                keep[p] = True

    return np.zeros(n, dtype=np.uint8), keep

def compress_core_subgraph(
    origin_i: int,
    dest_i: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    w: np.ndarray,
    keep: np.ndarray,
    in_indptr: np.ndarray,
    in_pos: np.ndarray,
    src_of_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, int], np.ndarray]:
    """
    Build a compressed CSR for the OD core (only nodes in core and kept arcs).

    Returns:
      od_indptr, od_indices, od_w, od_pos (original arc positions), map_global_to_local, local_to_global
    """
    n = len(indptr) - 1

    fwd = _reachable_forward(origin_i, indptr, indices, keep)
    bwd = _reachable_backward(dest_i, in_indptr, in_pos, src_of_pos, keep, n)
    core = (fwd & bwd).astype(np.uint8)

    if core[origin_i] == 0 or core[dest_i] == 0:
        return (np.zeros(1, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.int64),
                {},
                np.zeros(0, dtype=np.int64))

    core_nodes = np.nonzero(core)[0].astype(np.int64)
    map_gl_to_loc = {int(g): i for i, g in enumerate(core_nodes)}

    m2 = 0
    # count kept arcs inside core
    for g_u in core_nodes:
        s = int(indptr[g_u]); e = int(indptr[g_u+1])
        for p in range(s, e):
            if not keep[p]:
                continue
            g_v = int(indices[p])
            if core[g_v]:
                m2 += 1

    od_indptr = np.zeros(len(core_nodes) + 1, dtype=np.int64)
    od_indices = np.empty(m2, dtype=np.int64)
    od_w = np.empty(m2, dtype=np.float64)
    od_pos = np.empty(m2, dtype=np.int64)

    cursor = 0
    for i_u, g_u in enumerate(core_nodes):
        od_indptr[i_u] = cursor
        s = int(indptr[g_u]); e = int(indptr[g_u+1])
        for p in range(s, e):
            if not keep[p]:
                continue
            g_v = int(indices[p])
            if not core[g_v]:
                continue
            od_indices[cursor] = map_gl_to_loc[int(g_v)]
            od_w[cursor] = float(w[p])
            od_pos[cursor] = int(p)
            cursor += 1
        # end of row
    od_indptr[len(core_nodes)] = cursor

    return od_indptr, od_indices, od_w, od_pos, map_gl_to_loc, core_nodes
