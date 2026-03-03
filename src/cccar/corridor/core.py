import numpy as np
from numba import njit

def _precompute_incoming(indptr: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Precompute incoming-arc ranges so we can do backward reachability fast.

    Returns:
      in_indptr[v]..in_indptr[v+1] indexes into in_pos for arcs that enter v.
      in_pos[k] = forward-arc position p (0..m-1)
      src_of_pos[p] = u for forward arc position p
    """
    n = len(indptr) - 1
    m = len(indices)
    src_of_pos = np.empty(m, dtype=np.int64)
    indeg = np.zeros(n, dtype=np.int64)

    for u in range(n):
        s = int(indptr[u]); e = int(indptr[u+1])
        for p in range(s, e):
            src_of_pos[p] = u
            indeg[int(indices[p])] += 1

    in_indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(indeg, out=in_indptr[1:])

    in_pos = np.empty(m, dtype=np.int64)
    cursor = in_indptr[:-1].copy()
    for p in range(m):
        v = int(indices[p])
        k = cursor[v]
        in_pos[k] = p
        cursor[v] += 1

    return in_indptr, in_pos, src_of_pos

def _reachable_forward(origin: int, indptr: np.ndarray, indices: np.ndarray, keep: np.ndarray) -> np.ndarray:
    n = len(indptr) - 1
    seen = np.zeros(n, dtype=np.uint8)
    q = [origin]
    seen[origin] = 1
    while q:
        u = q.pop()
        s = int(indptr[u]); e = int(indptr[u+1])
        for p in range(s, e):
            if not keep[p]:
                continue
            v = int(indices[p])
            if seen[v] == 0:
                seen[v] = 1
                q.append(v)
    return seen

def _reachable_backward(dest: int, in_indptr: np.ndarray, in_pos: np.ndarray, src_of_pos: np.ndarray, keep: np.ndarray, n: int) -> np.ndarray:
    seen = np.zeros(n, dtype=np.uint8)
    q = [dest]
    seen[dest] = 1
    while q:
        v = q.pop()
        s = int(in_indptr[v]); e = int(in_indptr[v+1])
        for k in range(s, e):
            p = int(in_pos[k])
            if not keep[p]:
                continue
            u = int(src_of_pos[p])
            if seen[u] == 0:
                seen[u] = 1
                q.append(u)
    return seen

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
    """
    IDENTICAL LOGIC to v3_fast, but replaces Numba typed-list stacks with fixed-size
    array stacks for forward/backward reachability to reduce overhead.
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
    # 2) Count kept arcs in candidate graph (monotone mixed-potential + slack).
    # ------------------------------------------------------------------
    outdeg = np.zeros(k, np.int64)
    F = np.empty(k, np.float64)
    for i in range(k):
        u = cand_nodes[i]
        F[i] = dist_o[u] - dist_to_d[u]

    for i_u in range(k):
        u = cand_nodes[i_u]
        gu = dist_o[u]
        Fu = F[i_u]
        s = indptr[u]
        e = indptr[u + 1]
        for p in range(s, e):
            v = int(indices[p])
            if stamp[v] != cand_stamp:
                continue
            if gu + w[p] + dist_to_d[v] > budget:
                continue
            j_v = int(g2l[v])
            if F[j_v] > Fu:
                outdeg[i_u] += 1

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
        Fu = F[i_u]
        s = indptr[u]
        e = indptr[u + 1]
        for p in range(s, e):
            v = int(indices[p])
            if stamp[v] != cand_stamp:
                continue
            if gu + w[p] + dist_to_d[v] > budget:
                continue
            j_v = int(g2l[v])
            if F[j_v] > Fu:
                t = cursor[i_u]
                cand_indices[t] = j_v
                cand_pos[t] = p
                cursor[i_u] = t + 1

    # ------------------------------------------------------------------
    # 3) Reachability prune: forward from origin, backward from dest
    #     (array stacks instead of typed lists)
    # ------------------------------------------------------------------
    fwd = np.zeros(k, np.uint8)
    stack = np.empty(k, np.int64)
    top = 0
    stack[top] = o_loc
    top += 1
    fwd[o_loc] = 1
    while top > 0:
        top -= 1
        u = int(stack[top])
        s = cand_indptr[u]
        e = cand_indptr[u + 1]
        for t in range(s, e):
            v = int(cand_indices[t])
            if fwd[v] == 0:
                fwd[v] = 1
                stack[top] = v
                top += 1

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

    bwd = np.zeros(k, np.uint8)
    stack2 = np.empty(k, np.int64)
    top2 = 0
    stack2[top2] = d_loc
    top2 += 1
    bwd[d_loc] = 1
    while top2 > 0:
        top2 -= 1
        v = int(stack2[top2])
        s = rev_indptr[v]
        e = rev_indptr[v + 1]
        for t in range(s, e):
            u = int(rev_src[t])
            if bwd[u] == 0:
                bwd[u] = 1
                stack2[top2] = u
                top2 += 1

    core_mask = np.zeros(k, np.uint8)
    core_count = 0
    for i in range(k):
        if fwd[i] and bwd[i]:
            core_mask[i] = 1
            core_count += 1

    if core_count == 0 or core_mask[o_loc] == 0 or core_mask[d_loc] == 0:
        return (np.zeros(1, np.int64),
                np.zeros(0, np.int64),
                np.zeros(0, np.int64),
                np.zeros(0, np.float64),
                np.zeros(0, np.int64),
                -1, -1)

    # ------------------------------------------------------------------
    # 4) Compress to core CSR, compute per-slot decay
    # ------------------------------------------------------------------
    cand2core = np.full(k, -1, np.int64)
    core_nodes_global = np.empty(core_count, np.int64)
    ci = 0
    for i in range(k):
        if core_mask[i]:
            cand2core[i] = ci
            core_nodes_global[ci] = cand_nodes[i]
            ci += 1

    outdeg2 = np.zeros(core_count, np.int64)
    for u in range(k):
        if core_mask[u] == 0:
            continue
        u2 = int(cand2core[u])
        s = cand_indptr[u]
        e = cand_indptr[u + 1]
        for t in range(s, e):
            v = int(cand_indices[t])
            if core_mask[v]:
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

    for u in range(k):
        if core_mask[u] == 0:
            continue
        u2 = int(cand2core[u])
        s = cand_indptr[u]
        e = cand_indptr[u + 1]
        for t in range(s, e):
            v = int(cand_indices[t])
            if core_mask[v] == 0:
                continue
            v2 = int(cand2core[v])
            tt = cur3[u2]
            od_indices[tt] = v2
            p_global = int(cand_pos[t])
            od_pos[tt] = p_global
            # decay based on the *next node* (v_global), matching prior logic
            v_gl = int(core_nodes_global[v2])
            slot_decay[tt] = decay_node[v_gl]
            cur3[u2] = tt + 1

    start_loc = int(cand2core[o_loc])
    end_loc = int(cand2core[d_loc])
    return od_indptr, od_indices, od_pos, slot_decay, core_nodes_global, start_loc, end_loc
