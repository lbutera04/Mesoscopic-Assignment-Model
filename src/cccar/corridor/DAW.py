""" 
Dual-Arborescence and Webbing (DAW) methodology for s->t corridor DAG construction. 
"""

import numba as nb
import numpy as np

# ============================================================
# Small utilities
# ============================================================

@nb.njit(cache=True)
def _empty_result():
    return (
        np.zeros(1, dtype=np.int64),   # od_indptr
        np.zeros(0, dtype=np.int64),   # od_indices
        np.zeros(0, dtype=np.int64),   # od_pos
        np.zeros(0, dtype=np.float64), # slot_decay
        np.zeros(0, dtype=np.int64),   # core_nodes_final
        -1,                            # start_new
        -1,                            # end_new
        np.int64(0),                   # web_edges_unique
        np.int64(0),                   # web_edges_attempted
    )

@nb.njit(cache=True)
def _clear_probe(arc_stamp: np.ndarray):
    m = arc_stamp.shape[0]
    if m > 0:
        arc_stamp[0] = np.int64(0)  # primary failure code
    if m > 1:
        arc_stamp[1] = np.int64(0)  # payload A
    if m > 2:
        arc_stamp[2] = np.int64(0)  # payload B
    if m > 3:
        arc_stamp[3] = np.int64(0)  # payload C

@nb.njit(cache=True)
def _fail_with_probe(
    arc_stamp: np.ndarray,
    code: int,
    a: int = 0,
    b: int = 0,
    c: int = 0,
):
    if arc_stamp.shape[0] > 0:
        arc_stamp[0] = np.int64(code)
    if arc_stamp.shape[0] > 1:
        arc_stamp[1] = np.int64(a)
    if arc_stamp.shape[0] > 2:
        arc_stamp[2] = np.int64(b)
    if arc_stamp.shape[0] > 3:
        arc_stamp[3] = np.int64(c)
    return _empty_result()

@nb.njit(cache=True)
def _float_eq_rel(a: float, b: float, eps_rel: float) -> bool:
    return abs(a - b) <= eps_rel * max(1.0, abs(a), abs(b))

@nb.njit(cache=True)
def _find_arc_pos(indptr: np.ndarray, indices: np.ndarray, gu: int, gv: int) -> int:
    a = int(indptr[gu])
    b = int(indptr[gu + 1])
    for p in range(a, b):
        if int(indices[p]) == gv:
            return p
    return -1

@nb.njit(cache=True)
def _edge_supports_relation(
    indptr: np.ndarray,
    indices: np.ndarray,
    w: np.ndarray,
    gu: int,
    gv: int,
    expected: float,
    eps_rel: float,
) -> bool:
    a = int(indptr[gu])
    b = int(indptr[gu + 1])
    for p in range(a, b):
        if int(indices[p]) == gv:
            if _float_eq_rel(float(w[p]), expected, eps_rel):
                return True
    return False

@nb.njit(cache=True)
def _has_local_edge(edge_u: np.ndarray, edge_v: np.ndarray, edge_ct: int, u: int, v: int) -> bool:
    for i in range(edge_ct):
        if int(edge_u[i]) == u and int(edge_v[i]) == v:
            return True
    return False

@nb.njit(cache=True)
def _append_edge_strict(
    u: int,
    v: int,
    core_nodes: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_pos: np.ndarray,
    edge_ct: int,
) -> int:
    """
    Strict append with explicit failure reasons.

    Returns:
      >= 0 : new edge_ct on success
      -1   : duplicate local edge attempt
      -2   : no corresponding global CSR arc exists
      -3   : local edge buffers exhausted
    """
    if _has_local_edge(edge_u, edge_v, edge_ct, u, v):
        return -1

    gu = int(core_nodes[u])
    gv = int(core_nodes[v])
    p = _find_arc_pos(indptr, indices, gu, gv)
    if p < 0:
        return -2
    if edge_ct >= edge_u.shape[0]:
        return -3

    edge_u[edge_ct] = u
    edge_v[edge_ct] = v
    edge_pos[edge_ct] = p
    return edge_ct + 1

@nb.njit(cache=True)
def _advance_node_stamp_counter(stamp: np.ndarray, counter: np.ndarray) -> int:
    """
    Advance a separate generation counter for node-mark stamping.

    stamp is used only for per-node marks.
    counter[0] stores the generation counter.
    On wrap, clear stamp and restart at 1.
    """
    cur = int(counter[0]) + 1
    if cur == 0:
        stamp[:] = 0
        cur = 1
    counter[0] = cur
    return cur

# ============================================================
# Seed BFS cache builder (kept compatible with old public API)
# ============================================================

@nb.njit(cache=True)
def _bfs_hops_count_one_seed(
    indptr: np.ndarray,
    indices: np.ndarray,
    s: int,
    max_hops: int,
    stamp: np.ndarray,
    parent: np.ndarray,
    depth: np.ndarray,
    queue: np.ndarray,
    cur_stamp: int,
) -> int:
    n = indptr.shape[0] - 1
    if s < 0 or s >= n:
        return 0

    h = 0
    t = 0
    queue[t] = s
    t += 1
    stamp[s] = cur_stamp
    parent[s] = -1
    depth[s] = np.int16(0)

    m = 0
    while h < t:
        u = int(queue[h])
        h += 1

        du = int(depth[u])
        if du >= max_hops:
            continue

        a = int(indptr[u])
        b = int(indptr[u + 1])
        for p in range(a, b):
            v = int(indices[p])
            if stamp[v] == cur_stamp:
                continue
            stamp[v] = cur_stamp
            parent[v] = u
            depth[v] = np.int16(du + 1)
            queue[t] = v
            t += 1
            m += 1

    return m

@nb.njit(cache=True)
def _bfs_hops_fill_one_seed(
    indptr: np.ndarray,
    indices: np.ndarray,
    s: int,
    max_hops: int,
    stamp: np.ndarray,
    parent: np.ndarray,
    depth: np.ndarray,
    queue: np.ndarray,
    cur_stamp: int,
    out_u: np.ndarray,
    out_v: np.ndarray,
    write_base: int,
) -> int:
    n = indptr.shape[0] - 1
    if s < 0 or s >= n:
        return 0

    h = 0
    t = 0
    queue[t] = s
    t += 1
    stamp[s] = cur_stamp
    parent[s] = -1
    depth[s] = np.int16(0)

    wpos = 0
    while h < t:
        u = int(queue[h])
        h += 1

        du = int(depth[u])
        if du >= max_hops:
            continue

        a = int(indptr[u])
        b = int(indptr[u + 1])
        for p in range(a, b):
            v = int(indices[p])
            if stamp[v] == cur_stamp:
                continue
            stamp[v] = cur_stamp
            parent[v] = u
            depth[v] = np.int16(du + 1)
            queue[t] = v
            t += 1

            out_u[write_base + wpos] = u
            out_v[write_base + wpos] = v
            wpos += 1

    return wpos

def build_seed_bfs_tree_cache_hops_numba(
    indptr: np.ndarray,
    indices: np.ndarray,
    seeds_idx: np.ndarray,
    *,
    max_hops: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build per-seed hop-bounded BFS tree cache.

    The returned convention is stable for the builder below:
    for any non-empty seed slot sid, the seed root is recovered as
    seed_u[seed_ptr[sid]], because edges are written in BFS discovery
    order from the root.
    """
    if max_hops < 0:
        raise ValueError("max_hops must be non-negative")

    n = indptr.shape[0] - 1
    seeds = seeds_idx.astype(np.int64, copy=False)
    S = seeds.shape[0]

    stamp = np.zeros(n, dtype=np.int32)
    parent = np.empty(n, dtype=np.int64)
    depth = np.empty(n, dtype=np.int16)
    queue = np.empty(n, dtype=np.int64)

    seed_ptr = np.zeros(S + 1, dtype=np.int64)

    cur_stamp = np.int32(1)
    for si in range(S):
        cur_stamp = np.int32(cur_stamp + 1)
        if cur_stamp == 0:
            stamp.fill(0)
            cur_stamp = np.int32(1)

        s = int(seeds[si])
        if s < 0 or s >= n:
            seed_ptr[si + 1] = seed_ptr[si]
            continue

        m = _bfs_hops_count_one_seed(
            indptr, indices, s, int(max_hops),
            stamp, parent, depth, queue, int(cur_stamp),
        )
        seed_ptr[si + 1] = seed_ptr[si] + int(m)

    total_edges = int(seed_ptr[-1])
    all_u = np.empty(total_edges, dtype=np.int64)
    all_v = np.empty(total_edges, dtype=np.int64)

    stamp.fill(0)
    cur_stamp = np.int32(1)

    for si in range(S):
        cur_stamp = np.int32(cur_stamp + 1)
        if cur_stamp == 0:
            stamp.fill(0)
            cur_stamp = np.int32(1)

        s = int(seeds[si])
        if s < 0 or s >= n:
            continue

        base = int(seed_ptr[si])
        wrote = _bfs_hops_fill_one_seed(
            indptr, indices, s, int(max_hops),
            stamp, parent, depth, queue, int(cur_stamp),
            all_u, all_v, base,
        )

        expected = int(seed_ptr[si + 1] - seed_ptr[si])
        if wrote != expected:
            raise RuntimeError(f"Seed {si}: fill wrote {wrote} edges, expected {expected}")

    return seed_ptr, all_u, all_v

def compute_touched_seeds_for_od(
    reach_o: np.ndarray,
    dist_o: np.ndarray,
    dist_to_d: np.ndarray,
    end: int,
    slack: float,
    node_to_seedid: np.ndarray,
    *,
    eps_rel: float = 1e-9,
) -> np.ndarray:
    """
    Feasible-set-based touched-seed finder.

    Nodes are touched if they lie in the exact node-feasible set
    dist_o[u] + dist_to_d[u] <= slack * dist_o[end] + tol,
    and node_to_seedid[u] >= 0.
    """
    ssp = float(dist_o[int(end)])
    if (not np.isfinite(ssp)) or ssp <= 0.0:
        return np.zeros(0, dtype=np.int64)

    budget = float(slack) * ssp
    tol = float(eps_rel) * max(1.0, budget)

    u = reach_o.astype(np.int64, copy=False)
    good = np.isfinite(dist_o[u]) & np.isfinite(dist_to_d[u])
    feas = good & ((dist_o[u] + dist_to_d[u]) <= (budget + tol))
    if feas.size == 0:
        return np.zeros(0, dtype=np.int64)

    sid = node_to_seedid[u[feas]]
    sid = sid[sid >= 0]
    if sid.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.unique(sid.astype(np.int64, copy=False))

# ============================================================
# Parent / child helpers on exact V_B
# ============================================================

@nb.njit(cache=True)
def _build_core_nodes_from_stamp(
    reach_o: np.ndarray,
    stamp: np.ndarray,
    cand_stamp: int,
    g2l: np.ndarray,
    nloc: int,
) -> np.ndarray:
    core_nodes = np.empty(nloc, dtype=np.int64)
    for i in range(reach_o.shape[0]):
        gu = int(reach_o[i])
        if stamp[gu] == cand_stamp:
            core_nodes[int(g2l[gu])] = gu
    return core_nodes

@nb.njit(cache=True)
def _build_forward_parent_restricted(
    origin_i: int,
    core_nodes: np.ndarray,
    g2l: np.ndarray,
    stamp: np.ndarray,
    cand_stamp: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    w: np.ndarray,
    dist_o: np.ndarray,
    pred_to_start: np.ndarray,
    eps_rel: float,
) -> tuple[np.ndarray, np.ndarray, np.int64, np.int64, np.int64]:
    """
    Returns:
      parF, ok_mask, fail_node, fail_parent, fail_reason

    fail_reason:
      1 = predecessor missing / outside exact V_B
      2 = predecessor relation not supported by a matching CSR arc + weight
    """
    nloc = core_nodes.shape[0]
    parF = -np.ones(nloc, dtype=np.int64)
    ok = np.ones(1, dtype=np.uint8)

    fail_node = np.int64(-1)
    fail_parent = np.int64(-1)
    fail_reason = np.int64(0)

    for lv in range(nloc):
        gv = int(core_nodes[lv])
        if gv == origin_i:
            continue

        gp = int(pred_to_start[gv])
        if gp < 0 or stamp[gp] != cand_stamp:
            ok[0] = 0
            fail_node = np.int64(gv)
            fail_parent = np.int64(gp)
            fail_reason = np.int64(1)
            return parF, ok, fail_node, fail_parent, fail_reason

        expected = float(dist_o[gv]) - float(dist_o[gp])
        if not _edge_supports_relation(indptr, indices, w, gp, gv, expected, eps_rel):
            ok[0] = 0
            fail_node = np.int64(gv)
            fail_parent = np.int64(gp)
            fail_reason = np.int64(2)
            return parF, ok, fail_node, fail_parent, fail_reason

        parF[lv] = int(g2l[gp])

    return parF, ok, fail_node, fail_parent, fail_reason

@nb.njit(cache=True)
def _build_reverse_parent_restricted(
    dest_i: int,
    core_nodes: np.ndarray,
    g2l: np.ndarray,
    stamp: np.ndarray,
    cand_stamp: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    w: np.ndarray,
    dist_to_d: np.ndarray,
    pred_to_end: np.ndarray,
    eps_rel: float,
) -> tuple[np.ndarray, np.ndarray, np.int64, np.int64, np.int64]:
    """
    Returns:
      parR, ok_mask, fail_node, fail_parent, fail_reason

    fail_reason:
      1 = predecessor missing / outside exact V_B
      2 = successor-toward-d relation not supported by a matching CSR arc + weight
    """
    nloc = core_nodes.shape[0]
    parR = -np.ones(nloc, dtype=np.int64)
    ok = np.ones(1, dtype=np.uint8)

    fail_node = np.int64(-1)
    fail_parent = np.int64(-1)
    fail_reason = np.int64(0)

    for lu in range(nloc):
        gu = int(core_nodes[lu])
        if gu == dest_i:
            continue

        gp = int(pred_to_end[gu])
        if gp < 0 or stamp[gp] != cand_stamp:
            ok[0] = 0
            fail_node = np.int64(gu)
            fail_parent = np.int64(gp)
            fail_reason = np.int64(1)
            return parR, ok, fail_node, fail_parent, fail_reason

        expected = float(dist_to_d[gu]) - float(dist_to_d[gp])
        if not _edge_supports_relation(indptr, indices, w, gu, gp, expected, eps_rel):
            ok[0] = 0
            fail_node = np.int64(gu)
            fail_parent = np.int64(gp)
            fail_reason = np.int64(2)
            return parR, ok, fail_node, fail_parent, fail_reason

        parR[lu] = int(g2l[gp])

    return parR, ok, fail_node, fail_parent, fail_reason

@nb.njit(cache=True)
def _build_children_from_parent(
    parent_local: np.ndarray,
    root_local: int,
) -> tuple[np.ndarray, np.ndarray]:
    nloc = parent_local.shape[0]
    counts = np.zeros(nloc, dtype=np.int64)

    for v in range(nloc):
        if v == root_local:
            continue
        p = int(parent_local[v])
        if p >= 0:
            counts[p] += 1

    ptr = np.zeros(nloc + 1, dtype=np.int64)
    for i in range(nloc):
        ptr[i + 1] = ptr[i] + counts[i]

    idx = np.empty(int(ptr[nloc]), dtype=np.int64)
    fill = np.zeros(nloc, dtype=np.int64)

    for v in range(nloc):
        if v == root_local:
            continue
        p = int(parent_local[v])
        if p >= 0:
            pos = int(ptr[p] + fill[p])
            idx[pos] = v
            fill[p] += 1

    return ptr, idx

# ============================================================
# Halo
# ============================================================

@nb.njit(cache=True)
def _run_halo(
    s_local: int,
    t_local: int,
    childF_ptr: np.ndarray,
    childF_idx: np.ndarray,
    childR_ptr: np.ndarray,
    childR_idx: np.ndarray,
) -> np.ndarray:
    nloc = childF_ptr.shape[0] - 1
    side = np.zeros(nloc, dtype=np.int8)

    qF = np.empty(nloc, dtype=np.int64)
    qR = np.empty(nloc, dtype=np.int64)

    hF = 0
    tF = 0
    hR = 0
    tR = 0

    side[s_local] = 1
    side[t_local] = 2
    qF[tF] = s_local
    tF += 1
    qR[tR] = t_local
    tR += 1

    while hF < tF or hR < tR:
        if hF < tF:
            u = int(qF[hF])
            hF += 1

            if side[u] == 1:
                a = int(childF_ptr[u])
                b = int(childF_ptr[u + 1])
                for p in range(a, b):
                    v = int(childF_idx[p])
                    sv = int(side[v])
                    if sv == 0:
                        side[v] = 1
                        qF[tF] = v
                        tF += 1
                    elif sv == 2:
                        side[v] = 3

        if hR < tR:
            u = int(qR[hR])
            hR += 1

            if side[u] == 2:
                a = int(childR_ptr[u])
                b = int(childR_ptr[u + 1])
                for p in range(a, b):
                    v = int(childR_idx[p])
                    sv = int(side[v])
                    if sv == 0:
                        side[v] = 2
                        qR[tR] = v
                        tR += 1
                    elif sv == 1:
                        side[v] = 3

    return side

# ============================================================
# Edge-list -> CSR / topo / reachability helpers
# ============================================================

@nb.njit(cache=True)
def _build_csr_from_edges(
    nloc: int,
    active: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_ct: int,
) -> tuple[np.ndarray, np.ndarray]:
    deg = np.zeros(nloc, dtype=np.int64)
    for i in range(edge_ct):
        u = int(edge_u[i])
        v = int(edge_v[i])
        if active[u] == 1 and active[v] == 1:
            deg[u] += 1

    ptr = np.zeros(nloc + 1, dtype=np.int64)
    for i in range(nloc):
        ptr[i + 1] = ptr[i] + deg[i]

    idx = np.empty(int(ptr[nloc]), dtype=np.int64)
    fill = np.zeros(nloc, dtype=np.int64)

    for i in range(edge_ct):
        u = int(edge_u[i])
        v = int(edge_v[i])
        if active[u] == 1 and active[v] == 1:
            pos = int(ptr[u] + fill[u])
            idx[pos] = v
            fill[u] += 1

    return ptr, idx

@nb.njit(cache=True)
def _build_reverse_csr_from_edges(
    nloc: int,
    active: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_ct: int,
) -> tuple[np.ndarray, np.ndarray]:
    deg = np.zeros(nloc, dtype=np.int64)
    for i in range(edge_ct):
        u = int(edge_u[i])
        v = int(edge_v[i])
        if active[u] == 1 and active[v] == 1:
            deg[v] += 1

    ptr = np.zeros(nloc + 1, dtype=np.int64)
    for i in range(nloc):
        ptr[i + 1] = ptr[i] + deg[i]

    idx = np.empty(int(ptr[nloc]), dtype=np.int64)
    fill = np.zeros(nloc, dtype=np.int64)

    for i in range(edge_ct):
        u = int(edge_u[i])
        v = int(edge_v[i])
        if active[u] == 1 and active[v] == 1:
            pos = int(ptr[v] + fill[v])
            idx[pos] = u
            fill[v] += 1

    return ptr, idx

@nb.njit(cache=True)
def _forward_reachable_from_csr(
    start: int,
    active: np.ndarray,
    ptr: np.ndarray,
    idx: np.ndarray,
) -> np.ndarray:
    nloc = active.shape[0]
    seen = np.zeros(nloc, dtype=np.uint8)
    if start < 0 or start >= nloc or active[start] == 0:
        return seen

    q = np.empty(nloc, dtype=np.int64)
    h = 0
    t = 0
    q[t] = start
    t += 1
    seen[start] = 1

    while h < t:
        u = int(q[h])
        h += 1
        a = int(ptr[u])
        b = int(ptr[u + 1])
        for p in range(a, b):
            v = int(idx[p])
            if active[v] == 1 and seen[v] == 0:
                seen[v] = 1
                q[t] = v
                t += 1

    return seen

@nb.njit(cache=True)
def _kahn_toposort_edges(
    nloc: int,
    active: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_ct: int,
) -> tuple[np.ndarray, int]:
    indeg = np.zeros(nloc, dtype=np.int64)
    for i in range(edge_ct):
        u = int(edge_u[i])
        v = int(edge_v[i])
        if active[u] == 1 and active[v] == 1:
            indeg[v] += 1

    q = np.empty(nloc, dtype=np.int64)
    topo = np.empty(nloc, dtype=np.int64)

    h = 0
    t = 0
    active_ct = 0
    for u in range(nloc):
        if active[u] == 1:
            active_ct += 1
            if indeg[u] == 0:
                q[t] = u
                t += 1

    out = 0
    while h < t:
        u = int(q[h])
        h += 1
        topo[out] = u
        out += 1

        for i in range(edge_ct):
            uu = int(edge_u[i])
            vv = int(edge_v[i])
            if uu == u and active[vv] == 1:
                indeg[vv] -= 1
                if indeg[vv] == 0:
                    q[t] = vv
                    t += 1

    return topo, out

@nb.njit(cache=True)
def _fill_rank_from_topo(topo: np.ndarray, topo_len: int, nloc: int, active: np.ndarray) -> np.ndarray:
    rank = -np.ones(nloc, dtype=np.int64)
    for i in range(topo_len):
        u = int(topo[i])
        if active[u] == 1:
            rank[u] = i
    return rank

@nb.njit(cache=True)
def _insert_internal_block_before_y(
    topo: np.ndarray,
    topo_len: int,
    rank: np.ndarray,
    y: int,
    internal_path: np.ndarray,
    k_internal: int,
) -> int:
    """
    Insert internal_path[0:k_internal] contiguously immediately before y.
    Returns new topo_len.
    Assumes:
      - y already present in topo
      - internal vertices are not active / not yet in topo
    """
    if k_internal <= 0:
        return topo_len

    ry = int(rank[y])
    if ry < 0 or ry >= topo_len:
        return -1

    # shift suffix right
    for i in range(topo_len - 1, ry - 1, -1):
        topo[i + k_internal] = topo[i]

    # write block
    for i in range(k_internal):
        topo[ry + i] = internal_path[i]

    new_len = topo_len + k_internal

    # refresh ranks on affected suffix
    for i in range(ry, new_len):
        rank[int(topo[i])] = i

    return new_len

# ============================================================
# Seed-cache reconstruction
# ============================================================

@nb.njit(cache=True)
def _reconstruct_seed_parent_local_from_cache(
    seed_sid: int,
    seed_ptr: np.ndarray,
    seed_u: np.ndarray,
    seed_v: np.ndarray,
    stamp: np.ndarray,
    cand_stamp: int,
    g2l: np.ndarray,
    nloc: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Recover local root and local parent array from cached seed tree arcs.
    Production-compatible convention:
      root_global = seed_u[seed_ptr[seed_sid]] for non-empty slots.

    Returns:
      root_local, parent_local, ok_mask
    """
    ok = np.ones(1, dtype=np.uint8)
    parent_local = -np.ones(nloc, dtype=np.int64)

    if seed_sid < 0 or seed_sid + 1 >= seed_ptr.shape[0]:
        ok[0] = 0
        return -1, parent_local, ok

    a = int(seed_ptr[seed_sid])
    b = int(seed_ptr[seed_sid + 1])
    if a >= b:
        ok[0] = 0
        return -1, parent_local, ok

    root_global = int(seed_u[a])
    if root_global < 0 or stamp[root_global] != cand_stamp:
        ok[0] = 0
        return -1, parent_local, ok

    root_local = int(g2l[root_global])

    for k in range(a, b):
        gu = int(seed_u[k])
        gv = int(seed_v[k])

        if gu < 0 or gv < 0:
            continue
        if stamp[gu] != cand_stamp or stamp[gv] != cand_stamp:
            continue

        ul = int(g2l[gu])
        vl = int(g2l[gv])

        if vl == root_local:
            ok[0] = 0
            return -1, parent_local, ok

        if parent_local[vl] == -1:
            parent_local[vl] = ul
        elif int(parent_local[vl]) != ul:
            ok[0] = 0
            return -1, parent_local, ok

    parent_local[root_local] = -1
    return root_local, parent_local, ok

@nb.njit(cache=True)
def _recover_seed_path_to_hit(
    root_local: int,
    y_local: int,
    parent_local: np.ndarray,
    path_rev: np.ndarray,
) -> int:
    """
    Writes reversed y->root chain into path_rev[0:plen_rev].
    Returns plen_rev on success, -1 on failure.
    """
    cur = y_local
    plen = 0
    while True:
        if plen >= path_rev.shape[0]:
            return -1
        path_rev[plen] = cur
        plen += 1
        if cur == root_local:
            break
        p = int(parent_local[cur])
        if p < 0:
            return -1
        cur = p
    return plen

# ============================================================
# Main builder
# ============================================================

@nb.njit(cache=True)
def build_od_twotree_web_csr_numba(
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
    pred_to_start: np.ndarray,
    pred_to_end: np.ndarray,
    seed_ptr: np.ndarray,
    seed_u: np.ndarray,
    seed_v: np.ndarray,
    touched_seeds: np.ndarray,
    arc_stamp: np.ndarray,
    arc_counter: np.ndarray,
    max_edges_added_total: int = 8000,
    eps_rel: float = 1e-9,
):
    """
    Faithful OD-local DAG builder in Numba.

    Semantic behavior:
      - exact V_B only
      - fixed predecessor trees only
      - tree-only halo with meeting barriers
      - base core acyclic by construction
      - required prune to current directed s->t sub-DAG
      - branch-wise webbing from cached seed BFS tree arcs
      - valid topological order maintained during insertion
      - final mapping to global CSR arc slots only after final DAG construction

    Output contract matches the original production kernel:
      od_indptr, od_indices, od_pos, slot_decay, core_nodes_final,
      start_new, end_new, web_edges_unique, web_edges_attempted

    Notes on compatibility / unused parameters:
      - arc_stamp is unused here.
      - arc_counter[0] is used only as a generation counter for safe node-stamp marking.
      - reach_o is used only as a scan universe for building exact V_B efficiently;
        the actual local universe is the exact feasible set, not "all reachable".
      - w is used only to verify shortest-consistency of the fixed predecessor trees.
      - failure returns the same empty-graph style output as the original kernel
        rather than raising Python exceptions inside the hot path.
    """
    _clear_probe(arc_stamp)
    # ------------------------------------------------------------
    # Basic endpoint / budget validation
    # ------------------------------------------------------------

    n = indptr.shape[0] - 1
    if origin_i < 0 or origin_i >= n or dest_i < 0 or dest_i >= n:
        return _fail_with_probe(arc_stamp, 1, origin_i, dest_i, n)

    ssp = float(dist_o[int(dest_i)])
    if (not np.isfinite(ssp)):
        return _fail_with_probe(arc_stamp, 2, dest_i, 0, 0)
    if slack < 1.0:
        return _fail_with_probe(arc_stamp, 3, int(slack * 1_000_000.0), 0, 0)

    budget = float(slack) * ssp
    tol = float(eps_rel) * max(1.0, budget)

    # ------------------------------------------------------------
    # (A) Exact feasible set V_B, using reach_o only as scan universe
    # ------------------------------------------------------------
    cand_stamp = _advance_node_stamp_counter(stamp, arc_counter)

    nloc = 0
    for i in range(reach_o.shape[0]):
        gu = int(reach_o[i])
        if gu < 0 or gu >= n:
            continue
        if stamp[gu] == cand_stamp:
            continue

        ds = float(dist_o[gu])
        dt = float(dist_to_d[gu])
        if (not np.isfinite(ds)) or (not np.isfinite(dt)):
            continue
        if ds + dt > (budget + tol):
            continue

        stamp[gu] = cand_stamp
        g2l[gu] = nloc
        nloc += 1

    if stamp[int(origin_i)] != cand_stamp or stamp[int(dest_i)] != cand_stamp or nloc == 0:
        return _fail_with_probe(
            arc_stamp,
            10,
            int(stamp[int(origin_i)] == cand_stamp),
            int(stamp[int(dest_i)] == cand_stamp),
            nloc,
        )

    core_nodes = _build_core_nodes_from_stamp(reach_o, stamp, cand_stamp, g2l, nloc)

    s_local = int(g2l[int(origin_i)])
    t_local = int(g2l[int(dest_i)])

    # ------------------------------------------------------------
    # (B) Restrict the two fixed shortest-consistent trees to V_B
    # ------------------------------------------------------------

    parF, okF, fail_vF, fail_pF, fail_rF = _build_forward_parent_restricted(
        int(origin_i), core_nodes, g2l, stamp, cand_stamp,
        indptr, indices, w, dist_o, pred_to_start, float(eps_rel),
    )
    if okF[0] == 0:
        return _fail_with_probe(arc_stamp, 20, fail_vF, fail_pF, fail_rF)

    parR, okR, fail_vR, fail_pR, fail_rR = _build_reverse_parent_restricted(
        int(dest_i), core_nodes, g2l, stamp, cand_stamp,
        indptr, indices, w, dist_to_d, pred_to_end, float(eps_rel),
    )
    if okR[0] == 0:
        return _fail_with_probe(arc_stamp, 21, fail_vR, fail_pR, fail_rR)

    childF_ptr, childF_idx = _build_children_from_parent(parF, s_local)
    childR_ptr, childR_idx = _build_children_from_parent(parR, t_local)

    # ------------------------------------------------------------
    # (C) Halo: tree-only, meeting nodes are barriers
    # ------------------------------------------------------------
    side = _run_halo(s_local, t_local, childF_ptr, childF_idx, childR_ptr, childR_idx)

    # ------------------------------------------------------------
    # (D) Base core: activate side>0; emit only the two tree families.
    #     Shared forward/reverse implied edge is emitted once by construction.
    # ------------------------------------------------------------
    active = np.zeros(nloc, dtype=np.uint8)
    active_base = np.zeros(nloc, dtype=np.uint8)
    for u in range(nloc):
        if int(side[u]) > 0:
            active[u] = 1
            active_base[u] = 1

    max_edge_cap = max(4 * nloc + 4 * int(max_edges_added_total) + 32, 16)
    edge_u = np.empty(max_edge_cap, dtype=np.int64)
    edge_v = np.empty(max_edge_cap, dtype=np.int64)
    edge_pos = np.empty(max_edge_cap, dtype=np.int64)
    edge_ct = 0

    # forward family
    for v in range(nloc):
        sv = int(side[v])
        if active[v] == 0:
            continue
        if v == s_local:
            continue
        if not (sv == 1 or sv == 3):
            continue

        p = int(parF[v])

        if p < 0 or active[p] == 0:
            gp = -1 if p < 0 else int(core_nodes[p])
            return _fail_with_probe(arc_stamp, 30, int(core_nodes[v]), gp, 0)

        edge_ct_new = _append_edge_strict(
            p, v, core_nodes, indptr, indices, edge_u, edge_v, edge_pos, edge_ct
        )

        if edge_ct_new < 0:
            return _fail_with_probe(
                arc_stamp,
                31,
                int(core_nodes[p]),
                int(core_nodes[v]),
                int(-edge_ct_new),
            )

        edge_ct = edge_ct_new

    # reverse family
    for u in range(nloc):
        su = int(side[u])
        if active[u] == 0:
            continue
        if u == t_local:
            continue
        if not (su == 2 or su == 3):
            continue

        p = int(parR[u])

        if p < 0 or active[p] == 0:
            gp = -1 if p < 0 else int(core_nodes[p])
            return _fail_with_probe(arc_stamp, 32, int(core_nodes[u]), gp, 0)

        # Shared-family overlap case handled by construction:
        # if forward family already implies p-parent relation as u -> p, do not re-emit.
        skip_overlap = False
        if p != s_local:
            sp = int(side[p])
            if (sp == 1 or sp == 3) and int(parF[p]) == u:
                skip_overlap = True

        if skip_overlap:
            continue

        edge_ct_new = _append_edge_strict(
            u, p, core_nodes, indptr, indices, edge_u, edge_v, edge_pos, edge_ct
        )

        if edge_ct_new < 0:
            return _fail_with_probe(
                arc_stamp,
                33,
                int(core_nodes[u]),
                int(core_nodes[p]),
                int(-edge_ct_new),
            )

        edge_ct = edge_ct_new

    # ------------------------------------------------------------
    # (E) Required prune: keep only nodes on some directed s->t path
    # ------------------------------------------------------------
    base_ptr, base_idx = _build_csr_from_edges(nloc, active, edge_u, edge_v, edge_ct)
    base_rptr, base_ridx = _build_reverse_csr_from_edges(nloc, active, edge_u, edge_v, edge_ct)

    fwd_keep = _forward_reachable_from_csr(s_local, active, base_ptr, base_idx)
    rev_keep = _forward_reachable_from_csr(t_local, active, base_rptr, base_ridx)

    for u in range(nloc):
        if active[u] == 1:
            if fwd_keep[u] == 0 or rev_keep[u] == 0:
                active[u] = 0

    if active[s_local] == 0 or active[t_local] == 0:
        return _fail_with_probe(
            arc_stamp,
            40,
            int(active[s_local]),
            int(active[t_local]),
            0,
        )

    # keep only pruned edge set
    kept_edge_ct = 0
    for i in range(edge_ct):
        u = int(edge_u[i])
        v = int(edge_v[i])
        if active[u] == 1 and active[v] == 1:
            edge_u[kept_edge_ct] = u
            edge_v[kept_edge_ct] = v
            edge_pos[kept_edge_ct] = edge_pos[i]
            kept_edge_ct += 1
    edge_ct = kept_edge_ct
    tree_edge_ct = edge_ct

    # ------------------------------------------------------------
    # (F) Validate/materialize topological order of pruned base core
    # ------------------------------------------------------------
    topo, topo_len = _kahn_toposort_edges(nloc, active, edge_u, edge_v, edge_ct)

    active_ct = 0
    for u in range(nloc):
        if active[u] == 1:
            active_ct += 1

    if topo_len != active_ct:
        return _fail_with_probe(arc_stamp, 50, topo_len, active_ct, 0)

    # room for future insertions
    topo_work = np.empty(nloc, dtype=np.int64)
    for i in range(topo_len):
        topo_work[i] = topo[i]

    rank = _fill_rank_from_topo(topo_work, topo_len, nloc, active)

    # ------------------------------------------------------------
    # (G) Webbing from eligible touched seeds
    # ------------------------------------------------------------
    web_edges_added = 0

    seed_seen_stamp = np.zeros(nloc, dtype=np.int32)
    seed_parent = -np.ones(nloc, dtype=np.int64)
    path_rev = np.empty(nloc, dtype=np.int64)
    internal_path = np.empty(nloc, dtype=np.int64)

    for si in range(touched_seeds.shape[0]):
        if web_edges_added >= max_edges_added_total:
            break

        sid = int(touched_seeds[si])
        if sid < 0 or sid + 1 >= seed_ptr.shape[0]:
            continue

        root_local, seed_parent_local, ok_seed = _reconstruct_seed_parent_local_from_cache(
            sid, seed_ptr, seed_u, seed_v, stamp, cand_stamp, g2l, nloc
        )
        if ok_seed[0] == 0:
            continue

        if root_local < 0 or root_local >= nloc:
            continue
        if active[root_local] == 0:
            continue

        # copy local parent into reusable work array
        for j in range(nloc):
            seed_parent[j] = seed_parent_local[j]

        seed_child_ptr, seed_child_idx = _build_children_from_parent(seed_parent, root_local)

        seed_mark = int(si) + 1  # cheap per-seed marker in this loop
        if seed_mark == 0:
            seed_seen_stamp[:] = 0
            seed_mark = 1

        q = np.empty(nloc, dtype=np.int64)
        h = 0
        t = 0

        q[t] = root_local
        t += 1
        seed_seen_stamp[root_local] = seed_mark

        exhausted_budget = False

        while h < t:
            u = int(q[h])
            h += 1

            a = int(seed_child_ptr[u])
            b = int(seed_child_ptr[u + 1])
            for p in range(a, b):
                v = int(seed_child_idx[p])

                if seed_seen_stamp[v] == seed_mark:
                    # malformed seed cache; fail closed on this seed only
                    continue
                seed_seen_stamp[v] = seed_mark

                if active[v] == 1:
                    # First DAG hit on this branch
                    y = v

                    if int(rank[root_local]) >= int(rank[y]):
                        # downstream-cut / backward crossing: reject branch
                        continue

                    plen_rev = _recover_seed_path_to_hit(root_local, y, seed_parent, path_rev)
                    if plen_rev < 2:
                        continue

                    plen = plen_rev
                    # path forward = reverse of path_rev
                    # internal path = forward[1:-1]
                    k_internal = plen - 2

                    # direct-edge corner
                    if k_internal == 0:
                        if _has_local_edge(edge_u, edge_v, edge_ct, root_local, y):
                            # allowed skip only in this direct-edge corner
                            continue

                        if web_edges_added + 1 > max_edges_added_total:
                            exhausted_budget = True
                            break

                        edge_ct_new = _append_edge_strict(
                            root_local, y, core_nodes, indptr, indices,
                            edge_u, edge_v, edge_pos, edge_ct
                        )

                        if edge_ct_new < 0:
                            return _fail_with_probe(
                                arc_stamp,
                                60,
                                int(core_nodes[root_local]),
                                int(core_nodes[y]),
                                int(-edge_ct_new),
                            )

                        edge_ct = edge_ct_new
                        web_edges_added += 1
                        continue

                    # recover internal vertices in forward order
                    # path_rev = [y, ..., root]
                    # internal forward = path_rev[plen-2], ..., path_rev[1]
                    good_internal = True
                    write_k = 0
                    for r in range(plen - 2, 0, -1):
                        z = int(path_rev[r])
                        if z < 0 or z >= nloc:
                            good_internal = False
                            break
                        if active[z] == 1:
                            # internal vertex already active is forbidden
                            good_internal = False
                            break
                        internal_path[write_k] = z
                        write_k += 1

                    if (not good_internal) or write_k != k_internal:
                        continue

                    needed_edges = plen - 1
                    if web_edges_added + needed_edges > max_edges_added_total:
                        exhausted_budget = True
                        break

                    new_topo_len = _insert_internal_block_before_y(
                        topo_work, topo_len, rank, y, internal_path, k_internal
                    )

                    if new_topo_len < 0:
                        return _fail_with_probe(
                            arc_stamp,
                            61,
                            int(core_nodes[root_local]),
                            int(core_nodes[y]),
                            k_internal,
                        )

                    topo_len = new_topo_len

                    for i2 in range(k_internal):
                        z = int(internal_path[i2])
                        active[z] = 1

                    # Add exactly the path edges in order:
                    # root -> internal[0] -> ... -> internal[-1] -> y
                    prev = root_local
                    for i2 in range(k_internal):
                        z = int(internal_path[i2])
                        edge_ct_new = _append_edge_strict(
                            prev, z, core_nodes, indptr, indices,
                            edge_u, edge_v, edge_pos, edge_ct
                        )

                        if edge_ct_new < 0:
                            return _fail_with_probe(
                                arc_stamp,
                                62,
                                int(core_nodes[prev]),
                                int(core_nodes[z]),
                                int(-edge_ct_new),
                            )

                        edge_ct = edge_ct_new
                        prev = z

                    edge_ct_new = _append_edge_strict(
                        prev, y, core_nodes, indptr, indices,
                        edge_u, edge_v, edge_pos, edge_ct
                    )

                    if edge_ct_new < 0:
                        return _fail_with_probe(
                            arc_stamp,
                            63,
                            int(core_nodes[prev]),
                            int(core_nodes[y]),
                            int(-edge_ct_new),
                        )

                    edge_ct = edge_ct_new

                    web_edges_added += needed_edges

                    # branch terminates at first DAG hit: do NOT enqueue children of y
                    continue

                # Non-DAG node: keep exploring this branch
                q[t] = v
                t += 1

            if exhausted_budget:
                break

        if exhausted_budget:
            break

    # ------------------------------------------------------------
    # (H) Final validation
    # ------------------------------------------------------------
    topo_final, topo_final_len = _kahn_toposort_edges(nloc, active, edge_u, edge_v, edge_ct)

    active_ct_final = 0
    for u in range(nloc):
        if active[u] == 1:
            active_ct_final += 1

    if topo_final_len != active_ct_final:
        return _fail_with_probe(arc_stamp, 70, topo_final_len, active_ct_final, 0)

    final_ptr, final_idx = _build_csr_from_edges(nloc, active, edge_u, edge_v, edge_ct)
    fwd_final = _forward_reachable_from_csr(s_local, active, final_ptr, final_idx)

    if fwd_final[t_local] == 0:
        return _fail_with_probe(arc_stamp, 71, int(core_nodes[s_local]), int(core_nodes[t_local]), 0)

    # ------------------------------------------------------------
    # (I) Final compact output CSR + edge-slot-aligned decay
    # ------------------------------------------------------------
    old2new = -np.ones(nloc, dtype=np.int64)
    final_n = topo_final_len
    core_nodes_final = np.empty(final_n, dtype=np.int64)

    for i in range(final_n):
        old = int(topo_final[i])
        old2new[old] = i
        core_nodes_final[i] = int(core_nodes[old])

    start_new = int(old2new[s_local])
    end_new = int(old2new[t_local])

    outdeg = np.zeros(final_n, dtype=np.int64)
    final_m = 0
    for i in range(edge_ct):
        u = int(edge_u[i])
        v = int(edge_v[i])
        if active[u] == 1 and active[v] == 1:
            nu = int(old2new[u])
            outdeg[nu] += 1
            final_m += 1

    od_indptr = np.zeros(final_n + 1, dtype=np.int64)
    for i in range(final_n):
        od_indptr[i + 1] = od_indptr[i] + outdeg[i]

    od_indices = np.empty(final_m, dtype=np.int64)
    od_pos = np.empty(final_m, dtype=np.int64)
    slot_decay = np.empty(final_m, dtype=np.float64)
    fill = np.zeros(final_n, dtype=np.int64)

    for i in range(edge_ct):
        u = int(edge_u[i])
        v = int(edge_v[i])
        if active[u] == 0 or active[v] == 0:
            continue

        nu = int(old2new[u])
        nv = int(old2new[v])
        pos = int(od_indptr[nu] + fill[nu])

        od_indices[pos] = nv
        od_pos[pos] = int(edge_pos[i])
        slot_decay[pos] = float(decay_node[int(core_nodes[v])])

        fill[nu] += 1

    web_edges_unique = np.int64(edge_ct - tree_edge_ct)
    web_edges_attempted = np.int64(web_edges_added)

    return (
        od_indptr,
        od_indices,
        od_pos,
        slot_decay,
        core_nodes_final,
        start_new,
        end_new,
        web_edges_unique,
        web_edges_attempted,
    )