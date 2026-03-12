from __future__ import annotations

import numba as nb
import numpy as np

from ..osm.attributes import _sumo_roadclass

_ARTERIAL_RCS = {
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
}


def compute_arterial_seed_nodes_from_edgeattrs(
    Gfull,
    edge_attrs,
    *,
    min_incident: int = 4,
    min_distinct_bases: int = 2,
    require_link_present: bool = False,
) -> list[str]:
    """
    Seed selection copied from dag_benchmarks.py.
    """

    def base_rc(rc: str) -> str:
        return rc[:-5] if rc.endswith("_link") else rc

    seeds = []
    for n in Gfull.nodes():
        cnt = 0
        bases = set()
        has_link = False

        for v in Gfull.successors(n):
            rc = _sumo_roadclass(edge_attrs, v)
            if rc in _ARTERIAL_RCS:
                cnt += 1
                bases.add(base_rc(rc))
                if rc.endswith("_link"):
                    has_link = True

        for u in Gfull.predecessors(n):
            rc = _sumo_roadclass(edge_attrs, u)
            if rc in _ARTERIAL_RCS:
                cnt += 1
                bases.add(base_rc(rc))
                if rc.endswith("_link"):
                    has_link = True

        if cnt >= min_incident and len(bases) >= min_distinct_bases and (has_link or not require_link_present):
            seeds.append(n)

    return seeds


@nb.njit(cache=True)
def _bfs_hops_count_one_seed(
    indptr: np.ndarray,
    indices: np.ndarray,
    seed: int,
    max_hops: int,
    stamp: np.ndarray,
    parent: np.ndarray,
    depth: np.ndarray,
    queue: np.ndarray,
    cur_stamp: int,
) -> int:
    head = 0
    tail = 0

    stamp[seed] = cur_stamp
    parent[seed] = -1
    depth[seed] = 0
    queue[tail] = seed
    tail += 1

    edges = 0

    while head < tail:
        u = queue[head]
        head += 1

        du = depth[u]
        if du >= max_hops:
            continue

        a0 = indptr[u]
        a1 = indptr[u + 1]
        for a in range(a0, a1):
            v = indices[a]
            if stamp[v] == cur_stamp:
                continue
            stamp[v] = cur_stamp
            parent[v] = u
            depth[v] = du + 1
            queue[tail] = v
            tail += 1
            edges += 1

    return edges


@nb.njit(cache=True)
def _bfs_hops_fill_one_seed(
    indptr: np.ndarray,
    indices: np.ndarray,
    seed: int,
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
    head = 0
    tail = 0

    stamp[seed] = cur_stamp
    parent[seed] = -1
    depth[seed] = 0
    queue[tail] = seed
    tail += 1

    wpos = 0

    while head < tail:
        u = queue[head]
        head += 1

        du = depth[u]
        if du >= max_hops:
            continue

        a0 = indptr[u]
        a1 = indptr[u + 1]
        for a in range(a0, a1):
            v = indices[a]
            if stamp[v] == cur_stamp:
                continue
            stamp[v] = cur_stamp
            parent[v] = u
            depth[v] = du + 1
            queue[tail] = v
            tail += 1

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
    Vectorized touched-seed finder copied from dag_benchmarks.py.
    """
    ssp = float(dist_o[int(end)])
    budget = float(slack) * ssp
    tol = float(eps_rel) * max(1.0, budget)

    u = reach_o
    feas = (dist_o[u] + dist_to_d[u]) <= (budget + tol)

    sids = node_to_seedid[u[feas]]
    sids = sids[sids >= 0]
    if sids.size == 0:
        return np.empty(0, dtype=np.int64)

    return np.unique(sids.astype(np.int64, copy=False))


@nb.njit(cache=True)
def _build_tree_children_from_pred_local(
    reach_o: np.ndarray,
    stamp: np.ndarray,
    cand_stamp: int,
    g2l: np.ndarray,
    pred: np.ndarray,
    child_ptr: np.ndarray,
    child_idx: np.ndarray,
) -> int:
    nloc = child_ptr.shape[0] - 1

    for i in range(reach_o.shape[0]):
        gv = int(reach_o[i])
        if stamp[gv] != cand_stamp:
            continue
        gp = int(pred[gv])
        if gp < 0 or stamp[gp] != cand_stamp:
            continue
        pl = int(g2l[gp])
        child_ptr[pl + 1] += 1

    for u in range(nloc):
        child_ptr[u + 1] += child_ptr[u]

    m = int(child_ptr[nloc])
    write = child_ptr.copy()
    for i in range(reach_o.shape[0]):
        gv = int(reach_o[i])
        if stamp[gv] != cand_stamp:
            continue
        gp = int(pred[gv])
        if gp < 0 or stamp[gp] != cand_stamp:
            continue
        pl = int(g2l[gp])
        vl = int(g2l[gv])
        j = int(write[pl])
        child_idx[j] = vl
        write[pl] = j + 1

    return m


@nb.njit(cache=True)
def _twotree_halo_grow(
    start_loc: int,
    end_loc: int,
    f_child_ptr: np.ndarray,
    f_child_idx: np.ndarray,
    r_child_ptr: np.ndarray,
    r_child_idx: np.ndarray,
    side: np.ndarray,
    qF: np.ndarray,
    qR: np.ndarray,
) -> None:
    side[start_loc] = 1
    side[end_loc] = 2

    hF = 0
    tF = 0
    hR = 0
    tR = 0
    qF[tF] = start_loc
    tF += 1
    qR[tR] = end_loc
    tR += 1

    while hF < tF or hR < tR:
        if hF < tF:
            u = int(qF[hF])
            hF += 1
            if side[u] == 1:
                a = int(f_child_ptr[u])
                b = int(f_child_ptr[u + 1])
                for p in range(a, b):
                    v = int(f_child_idx[p])
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
                a = int(r_child_ptr[u])
                b = int(r_child_ptr[u + 1])
                for p in range(a, b):
                    v = int(r_child_idx[p])
                    sv = int(side[v])
                    if sv == 0:
                        side[v] = 2
                        qR[tR] = v
                        tR += 1
                    elif sv == 1:
                        side[v] = 3


@nb.njit(cache=True)
def _reachability_prune_to_dest(
    end_loc: int,
    side: np.ndarray,
    f_parent: np.ndarray,
    r_child_ptr: np.ndarray,
    r_child_idx: np.ndarray,
    keep: np.ndarray,
    q: np.ndarray,
) -> int:
    for i in range(keep.shape[0]):
        keep[i] = 0

    if side[end_loc] == 0:
        return 0

    h = 0
    t = 0
    keep[end_loc] = 1
    q[t] = end_loc
    t += 1

    while h < t:
        x = int(q[h])
        h += 1

        a = int(r_child_ptr[x])
        b = int(r_child_ptr[x + 1])
        for p in range(a, b):
            ch = int(r_child_idx[p])
            if side[ch] != 0 and keep[ch] == 0:
                keep[ch] = 1
                q[t] = ch
                t += 1

        pf = int(f_parent[x])
        if pf >= 0 and side[pf] != 0 and keep[pf] == 0:
            keep[pf] = 1
            q[t] = pf
            t += 1

    return t

import numba as nb
import numpy as np

@nb.njit(cache=True)
def _reachability_prune_to_dest_masked(
    end_loc: int,
    side: np.ndarray,
    f_parent: np.ndarray,
    r_child_ptr: np.ndarray,
    r_child_idx: np.ndarray,
    allowed: np.ndarray,   # uint8: 1 if node allowed to exist
    keep: np.ndarray,      # uint8 output
    q: np.ndarray,
) -> int:
    for i in range(keep.shape[0]):
        keep[i] = 0

    # end must exist + be allowed
    if side[end_loc] == 0 or allowed[end_loc] == 0:
        return 0

    h = 0
    t = 0
    keep[end_loc] = 1
    q[t] = end_loc
    t += 1

    while h < t:
        x = int(q[h])
        h += 1

        # walk reverse-tree children (toward origin)
        a = int(r_child_ptr[x])
        b = int(r_child_ptr[x + 1])
        for p in range(a, b):
            ch = int(r_child_idx[p])
            if side[ch] != 0 and allowed[ch] == 1 and keep[ch] == 0:
                keep[ch] = 1
                q[t] = ch
                t += 1

        # walk forward-tree parent (toward origin)
        pf = int(f_parent[x])
        if pf >= 0 and side[pf] != 0 and allowed[pf] == 1 and keep[pf] == 0:
            keep[pf] = 1
            q[t] = pf
            t += 1

    return t

# @nb.njit(cache=True)
# def _kahn_toposort_twotree_core(
#     side: np.ndarray,
#     keep: np.ndarray,
#     f_child_ptr: np.ndarray,
#     f_child_idx: np.ndarray,
#     r_parent: np.ndarray,
#     topo: np.ndarray,
#     indeg: np.ndarray,
#     q: np.ndarray,
# ) -> int:
#     nloc = side.shape[0]

#     for i in range(nloc):
#         indeg[i] = 0

#     for u in range(nloc):
#         if keep[u] == 0:
#             continue

#         a = int(f_child_ptr[u])
#         b = int(f_child_ptr[u + 1])
#         for p in range(a, b):
#             v = int(f_child_idx[p])
#             if keep[v] == 0:
#                 continue
#             indeg[v] += 1

#         pr = int(r_parent[u])
#         if pr >= 0 and keep[pr] == 1:
#             indeg[pr] += 1

#     h = 0
#     t = 0
#     for u in range(nloc):
#         if keep[u] == 1 and indeg[u] == 0:
#             q[t] = u
#             t += 1

#     out = 0
#     while h < t:
#         u = int(q[h])
#         h += 1
#         topo[out] = u
#         out += 1

#         a = int(f_child_ptr[u])
#         b = int(f_child_ptr[u + 1])
#         for p in range(a, b):
#             v = int(f_child_idx[p])
#             if keep[v] == 0:
#                 continue
#             indeg[v] -= 1
#             if indeg[v] == 0:
#                 q[t] = v
#                 t += 1

#         pr = int(r_parent[u])
#         if pr >= 0 and keep[pr] == 1:
#             indeg[pr] -= 1
#             if indeg[pr] == 0:
#                 q[t] = pr
#                 t += 1

#     return out

import numba as nb
import numpy as np

@nb.njit(cache=True)
def _kahn_toposort_twotree_core(
    side: np.ndarray,      # int8: 0 none, 1 F, 2 R, 3 meet
    keep: np.ndarray,      # uint8
    f_child_ptr: np.ndarray,
    f_child_idx: np.ndarray,
    r_parent: np.ndarray,  # int64 local parent toward dest
    topo: np.ndarray,      # int64 out
    indeg: np.ndarray,     # int32 scratch
    q: np.ndarray,         # int64 scratch
) -> int:
    """
    Faithful to the algorithm spec:
      - include forward-tree edges only for nodes in F or meet (side 1 or 3)
      - include reverse-tree edges only for nodes in R or meet (side 2 or 3)
    Edge directions:
      - forward edges: u -> v (parent to child)
      - reverse edges: u -> r_parent[u] (toward destination)
    """
    nloc = side.shape[0]

    # reset indeg
    for i in range(nloc):
        indeg[i] = 0

    # build indegrees under faithful edge set
    for u in range(nloc):
        if keep[u] == 0:
            continue

        su = int(side[u])

        # forward edges only from F or meet
        if su == 1 or su == 3:
            a = int(f_child_ptr[u])
            b = int(f_child_ptr[u + 1])
            for p in range(a, b):
                v = int(f_child_idx[p])
                if keep[v] == 0:
                    continue
                sv = int(side[v])
                # forward-tree nodes should live in F or meet (optional but safer)
                if sv == 1 or sv == 3:
                    indeg[v] += 1

        # reverse edge only from R or meet
        if su == 2 or su == 3:
            pr = int(r_parent[u])
            if pr >= 0 and keep[pr] == 1:
                spr = int(side[pr])
                # reverse-tree nodes should live in R or meet (optional but safer)
                if spr == 2 or spr == 3:
                    indeg[pr] += 1

    # init queue with indeg==0
    h = 0
    t = 0
    for u in range(nloc):
        if keep[u] == 1 and indeg[u] == 0:
            q[t] = u
            t += 1

    # Kahn
    out = 0
    while h < t:
        u = int(q[h])
        h += 1
        topo[out] = u
        out += 1

        su = int(side[u])

        if su == 1 or su == 3:
            a = int(f_child_ptr[u])
            b = int(f_child_ptr[u + 1])
            for p in range(a, b):
                v = int(f_child_idx[p])
                if keep[v] == 0:
                    continue
                sv = int(side[v])
                if sv == 1 or sv == 3:
                    indeg[v] -= 1
                    if indeg[v] == 0:
                        q[t] = v
                        t += 1

        if su == 2 or su == 3:
            pr = int(r_parent[u])
            if pr >= 0 and keep[pr] == 1:
                spr = int(side[pr])
                if spr == 2 or spr == 3:
                    indeg[pr] -= 1
                    if indeg[pr] == 0:
                        q[t] = pr
                        t += 1

    return out

@nb.njit(cache=True)
def _relabel_full_list(head: int, nxt: np.ndarray, label: np.ndarray) -> None:
    label_stride = 1024.0
    n = label.shape[0]
    cur = head
    i = 0
    while cur != -1:
        label[cur] = label_stride * float(i)
        cur = int(nxt[cur])
        i += 1
        if i > n:
            raise RuntimeError("Topo linked list corrupted: cycle detected during relabel")


@nb.njit(cache=True)
def _splice_insert_path_before_y(
    y: int,
    path_nodes: np.ndarray,
    k: int,
    head_ref: np.ndarray,
    tail_ref: np.ndarray,
    nxt: np.ndarray,
    prv: np.ndarray,
    label: np.ndarray,
    in_dag: np.ndarray,
) -> None:
    if k <= 0:
        return

    prev_y = int(prv[y])

    if prev_y == -1:
        _relabel_full_list(int(head_ref[0]), nxt, label)
        prev_y = int(prv[y])

    ly = float(label[y])
    lp = float(label[prev_y]) if prev_y != -1 else (ly - 2.0 * (k + 2))

    if ly - lp <= float(k + 1):
        _relabel_full_list(int(head_ref[0]), nxt, label)
        ly = float(label[y])
        prev_y = int(prv[y])
        lp = float(label[prev_y]) if prev_y != -1 else (ly - 2.0 * (k + 2))

    first = int(path_nodes[0])
    if prev_y == -1:
        head_ref[0] = first
        prv[first] = -1
    else:
        nxt[prev_y] = first
        prv[first] = prev_y

    for i in range(k):
        v = int(path_nodes[i])
        in_dag[v] = 1
        label[v] = lp + float(i + 1)

        if i + 1 < k:
            w = int(path_nodes[i + 1])
            nxt[v] = w
            prv[w] = v
        else:
            nxt[v] = y
            prv[y] = v

    if tail_ref[0] == -1:
        tail_ref[0] = y


@nb.njit(cache=True)
def _find_arc_pos(indptr: np.ndarray, indices: np.ndarray, gu: int, gv: int) -> int:
    a = int(indptr[gu])
    b = int(indptr[gu + 1])
    for p in range(a, b):
        if int(indices[p]) == gv:
            return p
    return -1


@nb.njit(cache=True)
def _append_edge_if_new(
    ul: int,
    vl: int,
    core_nodes: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    arc_stamp: np.ndarray,
    arc_mark: int,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_pos: np.ndarray,
    edge_ct: int,
) -> int:
    gu = int(core_nodes[ul])
    gv = int(core_nodes[vl])
    p = _find_arc_pos(indptr, indices, gu, gv)
    if p < 0:
        return edge_ct
    if arc_stamp[p] == arc_mark:
        return edge_ct
    if edge_ct >= edge_u.shape[0]:
        return edge_ct

    arc_stamp[p] = arc_mark
    edge_u[edge_ct] = ul
    edge_v[edge_ct] = vl
    edge_pos[edge_ct] = p
    return edge_ct + 1

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
    ssp = float(dist_o[int(dest_i)])
    if (not np.isfinite(ssp)) or ssp <= 0.0:
        return (
            np.zeros(1, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int64),
            -1,
            -1,
            np.int64(0),  # web_edges_unique
            np.int64(0),  # web_edges_attempted
        )

    budget = float(slack) * ssp
    tol = float(eps_rel) * max(1.0, budget)

    # ------------------------------------------------------------
    # (A) Candidate set = ALL nodes reachable from origin (reach_o)
    #     NO slack gating here. This preserves tree connectivity.
    # ------------------------------------------------------------
    cand_stamp = stamp[0] + 1
    stamp[0] = cand_stamp

    nloc = 0
    for i in range(reach_o.shape[0]):
        gu = int(reach_o[i])
        if stamp[gu] == cand_stamp:
            continue

        do = float(dist_o[gu])
        dd = float(dist_to_d[gu])
        if (not np.isfinite(do)) or (not np.isfinite(dd)):
            continue
        if do + dd > (budget + tol):
            continue

        stamp[gu] = cand_stamp
        g2l[gu] = nloc
        nloc += 1

    if stamp[int(origin_i)] != cand_stamp:
        stamp[int(origin_i)] = cand_stamp
        g2l[int(origin_i)] = nloc
        nloc += 1

    if stamp[int(dest_i)] != cand_stamp:
        stamp[int(dest_i)] = cand_stamp
        g2l[int(dest_i)] = nloc
        nloc += 1

    if nloc == 0 or stamp[int(origin_i)] != cand_stamp or stamp[int(dest_i)] != cand_stamp:
        return (
            np.zeros(1, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int64),
            -1,
            -1,
            np.int64(0),  # web_edges_unique
            np.int64(0),  # web_edges_attempted
        )

    core_nodes = np.empty(nloc, dtype=np.int64)
    for i in range(reach_o.shape[0]):
        gu = int(reach_o[i])
        if stamp[gu] == cand_stamp:
            core_nodes[int(g2l[gu])] = gu

    start_loc = int(g2l[int(origin_i)])
    end_loc = int(g2l[int(dest_i)])

    # ------------------------------------------------------------
    # (A.1) Feasible-set mask over the candidate set (V_B)
    # ------------------------------------------------------------
    allowed = np.zeros(nloc, dtype=np.uint8)
    for u in range(nloc):
        gu = int(core_nodes[u])
        do = float(dist_o[gu])
        dd = float(dist_to_d[gu])
        if (not np.isfinite(do)) or (not np.isfinite(dd)):
            continue
        if do + dd <= (budget + tol):
            allowed[u] = 1

    # Keep endpoints robustly (they should already satisfy this)
    allowed[start_loc] = 1
    allowed[end_loc] = 1

    # parents in local indexing (tree edges, ungated)
    f_parent = -np.ones(nloc, dtype=np.int64)
    r_parent = -np.ones(nloc, dtype=np.int64)

    for i in range(reach_o.shape[0]):
        gv = int(reach_o[i])
        if stamp[gv] != cand_stamp:
            continue
        vl = int(g2l[gv])

        gp = int(pred_to_start[gv])
        if gp >= 0 and stamp[gp] == cand_stamp:
            f_parent[vl] = int(g2l[gp])

        gr = int(pred_to_end[gv])
        if gr >= 0 and stamp[gr] == cand_stamp:
            r_parent[vl] = int(g2l[gr])

    # children CSR for forward-tree edges and reverse-tree edges
    f_child_ptr = np.zeros(nloc + 1, dtype=np.int64)
    f_child_idx = np.empty(nloc, dtype=np.int64)
    _ = _build_tree_children_from_pred_local(
        reach_o, stamp, cand_stamp, g2l, pred_to_start, f_child_ptr, f_child_idx
    )

    r_child_ptr = np.zeros(nloc + 1, dtype=np.int64)
    r_child_idx = np.empty(nloc, dtype=np.int64)
    _ = _build_tree_children_from_pred_local(
        reach_o, stamp, cand_stamp, g2l, pred_to_end, r_child_ptr, r_child_idx
    )

    # ------------------------------------------------------------
    # (B) Halo grow on ungated trees (structure first)
    # ------------------------------------------------------------
    side = np.zeros(nloc, dtype=np.int8)
    qf = np.empty(nloc, dtype=np.int64)
    qr = np.empty(nloc, dtype=np.int64)
    _twotree_halo_grow(start_loc, end_loc, f_child_ptr, f_child_idx, r_child_ptr, r_child_idx, side, qf, qr)

    if side[end_loc] == 0:
        return (
            np.zeros(1, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int64),
            -1,
            -1,
            np.int64(0),  # web_edges_unique
            np.int64(0),  # web_edges_attempted
        )

    # First prune: keep only nodes that can reach dest via (reverse children + forward parent)
    keep0 = np.zeros(nloc, dtype=np.uint8)
    qp = np.empty(nloc, dtype=np.int64)
    _ = _reachability_prune_to_dest(end_loc, side, f_parent, r_child_ptr, r_child_idx, keep0, qp)

    if keep0[start_loc] == 0 or keep0[end_loc] == 0:
        return (
            np.zeros(1, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int64),
            -1,
            -1,
            np.int64(0),  # web_edges_unique
            np.int64(0),  # web_edges_attempted
        )
    
    keep = keep0

    if keep[start_loc] == 0 or keep[end_loc] == 0:
        return (
            np.zeros(1, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int64),
            -1,
            -1,
            np.int64(0),  # web_edges_unique
            np.int64(0),  # web_edges_attempted
        )

    # ------------------------------------------------------------
    # (D) Toposort core DAG, init linked list topo structure
    # ------------------------------------------------------------
    topo = np.empty(nloc, dtype=np.int64)
    indeg = np.empty(nloc, dtype=np.int32)
    qk = np.empty(nloc, dtype=np.int64)
    topo_len = _kahn_toposort_twotree_core(side, keep, f_child_ptr, f_child_idx, r_parent, topo, indeg, qk)

    nxt = -np.ones(nloc, dtype=np.int64)
    prv = -np.ones(nloc, dtype=np.int64)
    label = np.zeros(nloc, dtype=np.float64)
    in_dag = np.zeros(nloc, dtype=np.uint8)

    head_ref = np.empty(1, dtype=np.int64)
    tail_ref = np.empty(1, dtype=np.int64)

    first = -1
    last = -1
    out_ct = 0
    for i in range(topo_len):
        u = int(topo[i])
        if keep[u] == 0:
            continue
        in_dag[u] = 1
        label[u] = 1024.0 * out_ct
        out_ct += 1
        if first == -1:
            first = u
            last = u
        else:
            nxt[last] = u
            prv[u] = last
            last = u

    head_ref[0] = first
    tail_ref[0] = last

    # ------------------------------------------------------------
    # (E) Seed initial edge set from tree edges (no edge slack filter)
    # ------------------------------------------------------------
    arc_mark = int(arc_counter[0]) + 1
    if arc_mark == 0:
        arc_stamp[:] = 0
        arc_mark = 1
    arc_counter[0] = arc_mark

    max_extra = 4 * nloc + 4 * int(max_edges_added_total) + 32
    edge_u = np.empty(max_extra, dtype=np.int64)
    edge_v = np.empty(max_extra, dtype=np.int64)
    edge_pos = np.empty(max_extra, dtype=np.int64)
    edge_ct = 0

    for v in range(nloc):
        if keep[v] == 0:
            continue

        sv = int(side[v])

        # forward tree edge only for nodes in F or meet: f_parent[v] -> v
        if sv == 1 or sv == 3:
            pf = int(f_parent[v])
            if pf >= 0 and keep[pf] == 1:
                spf = int(side[pf])
                if spf == 1 or spf == 3:
                    edge_ct = _append_edge_if_new(
                        pf, v, core_nodes, indptr, indices,
                        arc_stamp, arc_mark,
                        edge_u, edge_v, edge_pos, edge_ct,
                    )

        # reverse tree edge only for nodes in R or meet: v -> r_parent[v]
        if sv == 2 or sv == 3:
            pr = int(r_parent[v])
            if pr >= 0 and keep[pr] == 1:
                spr = int(side[pr])
                if spr == 2 or spr == 3:
                    edge_ct = _append_edge_if_new(
                        v, pr, core_nodes, indptr, indices,
                        arc_stamp, arc_mark,
                        edge_u, edge_v, edge_pos, edge_ct,
                    )

    tree_edge_ct = edge_ct

    # ------------------------------------------------------------
    # (F) Webbing: node-gated only (no dist_o+dist_to_d edge test)
    # ------------------------------------------------------------
    tmp_parent = -np.ones(nloc, dtype=np.int64)
    tmp_stamp = np.zeros(nloc, dtype=np.int32)
    cur_tmp_stamp = np.int32(1)

    full_chain = np.empty(nloc, dtype=np.int64)
    insert_chain = np.empty(nloc, dtype=np.int64)

    edges_added = 0

    for si in range(touched_seeds.shape[0]):
        sid = int(touched_seeds[si])
        if sid < 0:
            continue
        a = int(seed_ptr[sid])
        b = int(seed_ptr[sid + 1])
        if a >= b:
            continue

        gx = int(seed_u[a])
        if gx < 0 or stamp[gx] != cand_stamp:
            continue

        x = int(g2l[gx])
        if keep[x] == 0 or in_dag[x] == 0:
            continue

        cur_tmp_stamp = np.int32(cur_tmp_stamp + 1)
        if cur_tmp_stamp == 0:
            tmp_stamp[:] = 0
            cur_tmp_stamp = np.int32(1)

        tmp_stamp[x] = cur_tmp_stamp
        tmp_parent[x] = -1

        for k in range(a, b):
            if edges_added >= max_edges_added_total:
                break

            gu = int(seed_u[k])
            gv = int(seed_v[k])
            if stamp[gu] != cand_stamp or stamp[gv] != cand_stamp:
                continue

            ul = int(g2l[gu])
            vl = int(g2l[gv])

            # node-gated only: require both endpoints currently allowed to exist
            if allowed[ul] == 0 or allowed[vl] == 0:
                continue

            if tmp_stamp[vl] != cur_tmp_stamp:
                tmp_stamp[vl] = cur_tmp_stamp
                tmp_parent[vl] = ul

            if in_dag[vl] == 0:
                continue
            if label[vl] <= label[x]:
                continue

            y = vl

            cur = y
            plen = 0
            while cur != x:
                p = int(tmp_parent[cur])
                if p < 0:
                    plen = 0
                    break
                cur = p
                if cur == x:
                    break
                full_chain[plen] = cur
                plen += 1
                if plen >= full_chain.shape[0]:
                    plen = 0
                    break

            if plen == 0:
                continue

            for i in range(plen // 2):
                ttmp = full_chain[i]
                full_chain[i] = full_chain[plen - 1 - i]
                full_chain[plen - 1 - i] = ttmp

            k2 = 0
            for i in range(plen):
                v = int(full_chain[i])
                if in_dag[v] == 0:
                    insert_chain[k2] = v
                    k2 += 1

            if k2 > 0:
                _splice_insert_path_before_y(
                    y,
                    insert_chain,
                    k2,
                    head_ref,
                    tail_ref,
                    nxt,
                    prv,
                    label,
                    in_dag,
                )

                prev = x
                for i in range(plen):
                    vv = int(full_chain[i])
                    edge_ct = _append_edge_if_new(
                        prev, vv, core_nodes, indptr, indices,
                        arc_stamp, arc_mark,
                        edge_u, edge_v, edge_pos, edge_ct,
                    )
                    prev = vv
                edge_ct = _append_edge_if_new(
                    prev, y, core_nodes, indptr, indices,
                    arc_stamp, arc_mark,
                    edge_u, edge_v, edge_pos, edge_ct,
                )

                edges_added += (k2 + 1)

        if edges_added >= max_edges_added_total:
            break

    # ------------------------------------------------------------
    # (G) Final compaction to CSR over in_dag nodes (unchanged)
    # ------------------------------------------------------------
    old2new = -np.ones(nloc, dtype=np.int64)
    n_final = 0
    for i in range(nloc):
        if in_dag[i] == 1:
            old2new[i] = n_final
            n_final += 1

    if n_final == 0:
        return (
            np.zeros(1, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int64),
            -1,
            -1,
            np.int64(0),  # web_edges_unique
            np.int64(0),  # web_edges_attempted
        )

    start_new = int(old2new[start_loc])
    end_new = int(old2new[end_loc])
    if start_new < 0 or end_new < 0:
        return (
            np.zeros(1, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int64),
            -1,
            -1,
            np.int64(0),  # web_edges_unique
            np.int64(0),  # web_edges_attempted
        )

    outdeg = np.zeros(n_final, dtype=np.int64)
    m_final = 0
    for e in range(edge_ct):
        u_old = int(edge_u[e])
        v_old = int(edge_v[e])
        u_new = int(old2new[u_old])
        v_new = int(old2new[v_old])
        if u_new < 0 or v_new < 0:
            continue
        outdeg[u_new] += 1
        m_final += 1

    od_indptr = np.zeros(n_final + 1, dtype=np.int64)
    for u in range(n_final):
        od_indptr[u + 1] = od_indptr[u] + outdeg[u]

    od_indices = np.empty(m_final, dtype=np.int64)
    od_pos = np.empty(m_final, dtype=np.int64)
    slot_decay = np.empty(m_final, dtype=np.float64)

    core_nodes_final = np.empty(n_final, dtype=np.int64)
    for old in range(nloc):
        new = int(old2new[old])
        if new >= 0:
            core_nodes_final[new] = int(core_nodes[old])

    cursor = od_indptr[:-1].copy()
    for e in range(edge_ct):
        u_old = int(edge_u[e])
        v_old = int(edge_v[e])
        p = int(edge_pos[e])
        u_new = int(old2new[u_old])
        v_new = int(old2new[v_old])
        if u_new < 0 or v_new < 0:
            continue
        tcur = int(cursor[u_new])
        od_indices[tcur] = v_new
        od_pos[tcur] = p
        slot_decay[tcur] = float(decay_node[int(core_nodes[v_old])])
        cursor[u_new] = tcur + 1

    web_edges_unique = edge_ct - tree_edge_ct
    web_edges_attempted = edges_added

    return od_indptr, od_indices, od_pos, slot_decay, core_nodes_final, start_new, end_new, np.int64(web_edges_unique), np.int64(web_edges_attempted),