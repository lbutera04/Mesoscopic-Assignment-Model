from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, Any

import numpy as np


# ============================================================
# Exceptions / simple utilities
# ============================================================

class ODDAGBuildError(RuntimeError):
    """Raised when the OD-local DAG cannot be built faithfully."""


def _is_finite(x: float) -> bool:
    return np.isfinite(x)


def _as_np_int(arr: Any) -> np.ndarray:
    out = np.asarray(arr)
    if out.dtype.kind not in ("i", "u", "b"):
        out = out.astype(np.int64, copy=False)
    return out


def _as_np_float(arr: Any) -> np.ndarray:
    out = np.asarray(arr)
    if out.dtype.kind not in ("f", "i", "u"):
        out = out.astype(np.float64, copy=False)
    else:
        out = out.astype(np.float64, copy=False)
    return out


def _validate_csr(indptr: np.ndarray, indices: np.ndarray, weights: np.ndarray) -> None:
    if indptr.ndim != 1 or indices.ndim != 1 or weights.ndim != 1:
        raise ValueError("CSR arrays must be 1D.")
    if len(indptr) < 2:
        raise ValueError("indptr must have length >= 2.")
    if len(indices) != len(weights):
        raise ValueError("indices and weights must have the same length.")
    if indptr[0] != 0:
        raise ValueError("indptr[0] must be 0.")
    if indptr[-1] != len(indices):
        raise ValueError("indptr[-1] must equal len(indices).")
    if np.any(indptr[1:] < indptr[:-1]):
        raise ValueError("indptr must be nondecreasing.")


# ============================================================
# Seed tree representation
# ============================================================

def _build_local_parent_from_seed_cache(
    seed_sid: int,
    seed_root_global: int,
    seed_ptr: np.ndarray,
    seed_u: np.ndarray,
    seed_v: np.ndarray,
    g2l: np.ndarray,
    n_local: int,
) -> np.ndarray:
    """
    Build the local parent array for one seed's cached BFS tree,
    restricted to the current feasible local universe V_B.

    touched_seeds: Seed SLOT ids touched by this OD, as returned by 
        compute_touched_seeds_for_od. Each entry indexes 
        seed_idx / seed_ptr / seed_u / seed_v.
    seed_root_global is the global root node for this seed slot.
    """
    parent_local = np.full(n_local, -1, dtype=np.int64)

    a = int(seed_ptr[seed_sid])
    b = int(seed_ptr[seed_sid + 1])

    root_local = int(g2l[seed_root_global])
    if root_local < 0:
        return parent_local

    for k in range(a, b):
        gu = int(seed_u[k])
        gv = int(seed_v[k])

        lu = int(g2l[gu])
        lv = int(g2l[gv])

        # Restrict tree to V_B only
        if lu < 0 or lv < 0:
            continue

        if lv == root_local:
            raise ODDAGBuildError(
                f"Seed slot {seed_sid} cache is malformed: root appears as a child."
            )

        if parent_local[lv] != -1 and parent_local[lv] != lu:
            raise ODDAGBuildError(
                f"Seed slot {seed_sid} cache is not a tree over V_B: "
                f"local vertex {lv} has multiple parents."
            )

        parent_local[lv] = lu

    return parent_local

# ============================================================
# Edge lookup / consistency checks
# ============================================================

def _find_first_arc_pos(
    indptr: np.ndarray,
    indices: np.ndarray,
    u: int,
    v: int,
) -> int:
    """
    Return the first CSR slot for edge (u, v).
    Parallel arcs are allowed; we choose the first one deterministically.
    """
    start = int(indptr[u])
    end = int(indptr[u + 1])
    for eid in range(start, end):
        if int(indices[eid]) == v:
            return eid
    return -1


def _edge_supports_relation(
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    u: int,
    v: int,
    expected_delta: float,
    atol: float = 1e-9,
    rtol: float = 1e-9,
) -> bool:
    """
    Return True if there exists at least one parallel arc (u, v)
    whose weight satisfies the required shortest-consistency relation.
    """
    start = int(indptr[u])
    end = int(indptr[u + 1])
    for eid in range(start, end):
        if int(indices[eid]) != v:
            continue
        w = float(weights[eid])
        if abs(w - expected_delta) <= atol + rtol * max(abs(w), abs(expected_delta), 1.0):
            return True
    return False

def _edge_exists(
    indptr: np.ndarray,
    indices: np.ndarray,
    u: int,
    v: int,
) -> bool:
    start = int(indptr[u])
    end = int(indptr[u + 1])
    for eid in range(start, end):
        if int(indices[eid]) == v:
            return True
    return False

def _float_eq(a: float, b: float, atol: float = 1e-9, rtol: float = 1e-9) -> bool:
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b), 1.0)


# ============================================================
# Feasible set / local indexing
# ============================================================

def _build_feasible_local_index(
    s: int,
    t: int,
    alpha: float,
    dist_s: np.ndarray,
    dist_t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Build the exact feasible set V_B = {v : d_s(v) + d_t(v) <= alpha * d_s(t)}.
    """
    ds_t = float(dist_s[t])
    if not _is_finite(ds_t):
        raise ODDAGBuildError("Destination t is unreachable from s in dist_s; cannot build OD DAG.")

    B = alpha * ds_t

    feasible = np.isfinite(dist_s) & np.isfinite(dist_t) & ((dist_s + dist_t) <= B + 1e-9)
    if not feasible[s] or not feasible[t]:
        raise ODDAGBuildError(
            "Endpoints s and/or t do not lie in V_B. The builder fails rather than forcing them in."
        )

    core_nodes = np.flatnonzero(feasible).astype(np.int64)
    g2l = np.full(len(dist_s), -1, dtype=np.int64)
    g2l[core_nodes] = np.arange(len(core_nodes), dtype=np.int64)
    return core_nodes, g2l, B


# ============================================================
# Fixed shortest-consistent trees restricted to V_B
# ============================================================

def _build_forward_tree_restricted(
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    s: int,
    core_nodes: np.ndarray,
    g2l: np.ndarray,
    dist_s: np.ndarray,
    pred_s: np.ndarray,
) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Build par_F on V_B exactly from the fixed predecessor array pred_s.
    No tie-breaking, no parent substitution.
    """
    n_local = len(core_nodes)
    parF_local = np.full(n_local, -1, dtype=np.int64)
    childrenF: List[List[int]] = [[] for _ in range(n_local)]

    for lv, vg in enumerate(core_nodes):
        if vg == s:
            continue
        pg = int(pred_s[vg])
        if pg < 0:
            continue
        lp = int(g2l[pg])
        if lp < 0:
            continue

        expected = float(dist_s[vg]) - float(dist_s[pg])
        if not _edge_supports_relation(indptr, indices, weights, pg, vg, expected):
            raise ODDAGBuildError(
                f"Forward predecessor array violates shortest-consistency at vertex {vg}: "
                f"no parallel edge ({pg},{vg}) supports d_s[{vg}] - d_s[{pg}]."
            )

        parF_local[lv] = lp
        childrenF[lp].append(lv)

    return parF_local, childrenF


def _build_reverse_tree_restricted(
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    t: int,
    core_nodes: np.ndarray,
    g2l: np.ndarray,
    dist_t: np.ndarray,
    pred_t_rev: np.ndarray,
) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Build par_R on V_B exactly from the fixed reverse predecessor array.

    Semantics:
      par_R(u) = successor of u toward t in the original graph,
      and must satisfy d_t(u) = d_t(par_R(u)) + w(u, par_R(u)).
    """
    n_local = len(core_nodes)
    parR_local = np.full(n_local, -1, dtype=np.int64)
    childrenR: List[List[int]] = [[] for _ in range(n_local)]

    for lu, ug in enumerate(core_nodes):
        if ug == t:
            continue
        pg = int(pred_t_rev[ug])
        if pg < 0:
            continue
        lp = int(g2l[pg])
        if lp < 0:
            continue

        expected = float(dist_t[ug]) - float(dist_t[pg])
        if not _edge_supports_relation(indptr, indices, weights, ug, pg, expected):
            raise ODDAGBuildError(
                f"Reverse predecessor array violates shortest-consistency at vertex {ug}: "
                f"no parallel edge ({ug},{pg}) supports d_t[{ug}] - d_t[{pg}]."
            )

        parR_local[lu] = lp
        # children of parent in the tree rooted at t
        childrenR[lp].append(lu)

    return parR_local, childrenR


# ============================================================
# Halo growth: tree-only, barrier at first meeting
# ============================================================

def _run_halo(
    s_local: int,
    t_local: int,
    childrenF: List[List[int]],
    childrenR: List[List[int]],
) -> np.ndarray:
    """
    side states:
      0 unseen
      1 reached only from forward tree
      2 reached only from reverse tree
      3 meeting vertex (both)
    Interleaving policy:
      one pop from forward queue if nonempty,
      then one pop from reverse queue if nonempty,
      repeat until both queues empty.
    A meeting vertex is marked 3 and not enqueued again.
    """
    n_local = len(childrenF)
    side = np.zeros(n_local, dtype=np.int8)
    side[s_local] = 1
    side[t_local] = 2

    qf = deque([s_local])
    qr = deque([t_local])

    while qf or qr:
        if qf:
            u = qf.popleft()
            if side[u] == 1:
                for v in childrenF[u]:
                    sv = int(side[v])
                    if sv == 0:
                        side[v] = 1
                        qf.append(v)
                    elif sv == 1:
                        pass
                    elif sv == 2:
                        side[v] = 3
                    elif sv == 3:
                        pass
                    else:
                        raise AssertionError("invalid side state")

        if qr:
            u = qr.popleft()
            if side[u] == 2:
                for v in childrenR[u]:
                    sv = int(side[v])
                    if sv == 0:
                        side[v] = 2
                        qr.append(v)
                    elif sv == 2:
                        pass
                    elif sv == 1:
                        side[v] = 3
                    elif sv == 3:
                        pass
                    else:
                        raise AssertionError("invalid side state")

    return side


# ============================================================
# Strict local DAG container (no blanket dedupe)
# ============================================================

class StrictLocalDAG:
    """
    Local DAG over the fixed feasible-node universe V_B.

    active[v] indicates current DAG membership.
    adjacency[v] contains outgoing local neighbors for active v.
    out_sets[v] is used ONLY as a strict logic check for duplicate-attempt detection.
    It is not used as a 'repair' or normal-control-flow deduper.
    """

    def __init__(self, n_local: int):
        self.n_local = n_local
        self.active = np.zeros(n_local, dtype=np.bool_)
        self.adjacency: List[List[int]] = [[] for _ in range(n_local)]
        self.out_sets: List[set] = [set() for _ in range(n_local)]

    def activate(self, v: int) -> None:
        self.active[v] = True

    def is_active(self, v: int) -> bool:
        return bool(self.active[v])

    def has_edge(self, u: int, v: int) -> bool:
        return v in self.out_sets[u]

    def add_edge_strict(self, u: int, v: int, *, context: str) -> None:
        if v in self.out_sets[u]:
            raise ODDAGBuildError(
                f"Duplicate edge attempt detected for ({u} -> {v}) during {context}. "
                f"This implementation does not rely on dedupe as normal logic."
            )
        self.adjacency[u].append(v)
        self.out_sets[u].add(v)

    def add_edge_skip_if_present_direct_corner(self, u: int, v: int) -> bool:
        """
        This is ONLY for the allowed direct-edge branch corner case in webbing:
        path length 1, x->y, where y is already an outgoing DAG neighbor of x.
        Returns True if edge already existed (caller should skip branch).
        """
        if v in self.out_sets[u]:
            return True
        self.adjacency[u].append(v)
        self.out_sets[u].add(v)
        return False


# ============================================================
# Base core construction
# ============================================================

def _build_base_core(
    s_local: int,
    t_local: int,
    side: np.ndarray,
    parF_local: np.ndarray,
    parR_local: np.ndarray,
) -> StrictLocalDAG:
    """
    Base edge set:
      {(par_F(v) -> v): side(v) in {1,3}}
      union
      {(u -> par_R(u)): side(u) in {2,3}}

    No dedupe. If same edge would be emitted twice, that is treated as a logic error.
    """
    n_local = len(side)
    dag = StrictLocalDAG(n_local)

    V0 = np.where(side > 0)[0]
    for v in V0:
        dag.activate(int(v))

    for v in V0:
        sv = int(side[v])

        if v != s_local and sv in (1, 3):
            p = int(parF_local[v])
            if p < 0:
                raise ODDAGBuildError(
                    f"Vertex {v} is in forward halo side {sv} but has no forward-tree parent in V_B."
                )
            dag.add_edge_strict(p, int(v), context="base-core forward-tree emission")

        if v != t_local and sv in (2, 3):
            p = int(parR_local[v])
            if p < 0:
                raise ODDAGBuildError(
                    f"Vertex {v} is in reverse halo side {sv} but has no reverse-tree parent in V_B."
                )
            dag.add_edge_strict(int(v), p, context="base-core reverse-tree emission")

    return dag


# ============================================================
# Reachability prune (required for sampler-safe destination reachability)
# ============================================================

def _reverse_adjacency_of_active(dag: StrictLocalDAG) -> List[List[int]]:
    rev = [[] for _ in range(dag.n_local)]
    for u in range(dag.n_local):
        if not dag.active[u]:
            continue
        for v in dag.adjacency[u]:
            if dag.active[v]:
                rev[v].append(u)
    return rev


def _forward_reachable(dag: StrictLocalDAG, source: int) -> np.ndarray:
    seen = np.zeros(dag.n_local, dtype=np.bool_)
    if not dag.active[source]:
        return seen
    q = deque([source])
    seen[source] = True
    while q:
        u = q.popleft()
        for v in dag.adjacency[u]:
            if dag.active[v] and not seen[v]:
                seen[v] = True
                q.append(v)
    return seen


def _reverse_reachable_to_target(dag: StrictLocalDAG, target: int) -> np.ndarray:
    seen = np.zeros(dag.n_local, dtype=np.bool_)
    if not dag.active[target]:
        return seen
    rev = _reverse_adjacency_of_active(dag)
    q = deque([target])
    seen[target] = True
    while q:
        u = q.popleft()
        for p in rev[u]:
            if dag.active[p] and not seen[p]:
                seen[p] = True
                q.append(p)
    return seen


def _prune_to_st_subdag(
    dag: StrictLocalDAG,
    s_local: int,
    t_local: int,
) -> None:
    """
    The only allowed prune:
      keep vertices that are both:
        reachable from s
        and can reach t
    """
    fwd = _forward_reachable(dag, s_local)
    rev = _reverse_reachable_to_target(dag, t_local)
    keep = fwd & rev

    if not keep[s_local] or not keep[t_local]:
        raise ODDAGBuildError(
            "Base core has no directed s->t connection after forward/reverse reachability prune."
        )

    for v in range(dag.n_local):
        if dag.active[v] and not keep[v]:
            dag.active[v] = False

    # Remove edges touching pruned vertices
    for u in range(dag.n_local):
        if not dag.active[u]:
            dag.adjacency[u] = []
            dag.out_sets[u].clear()
            continue
        new_row = []
        new_set = set()
        for v in dag.adjacency[u]:
            if dag.active[v]:
                new_row.append(v)
                new_set.add(v)
        dag.adjacency[u] = new_row
        dag.out_sets[u] = new_set


# ============================================================
# Topological order (validation/materialization only)
# ============================================================

def _toposort_active_or_raise(dag: StrictLocalDAG) -> List[int]:
    indeg = np.zeros(dag.n_local, dtype=np.int64)
    active_vertices = [u for u in range(dag.n_local) if dag.active[u]]
    for u in active_vertices:
        for v in dag.adjacency[u]:
            if dag.active[v]:
                indeg[v] += 1

    q = deque([u for u in active_vertices if indeg[u] == 0])
    order: List[int] = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in dag.adjacency[u]:
            if dag.active[v]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)

    if len(order) != len(active_vertices):
        raise ODDAGBuildError(
            "A cycle was detected during topological validation. "
            "This should be impossible if the construction rules were followed."
        )
    return order


def _rank_from_order(order: List[int], n_local: int) -> np.ndarray:
    rank = np.full(n_local, -1, dtype=np.int64)
    for i, v in enumerate(order):
        rank[v] = i
    return rank


# ============================================================
# Tree-path utilities for seed BFS trees
# ============================================================

# def _build_local_parent_for_seed(
#     seed_tree: SeedTree,
#     core_nodes: np.ndarray,
#     g2l: np.ndarray,
# ) -> np.ndarray:
#     """
#     Build local parent array over the fixed feasible universe V_B.

#     For nodes not represented / not in V_B / unreachable from seed tree: parent = -1.
#     """
#     n_local = len(core_nodes)
#     parent_local = np.full(n_local, -1, dtype=np.int64)

#     if seed_tree.parent_global is not None:
#         pg = seed_tree.parent_global
#         for lv, vg in enumerate(core_nodes):
#             if vg == seed_tree.root_global:
#                 continue
#             p = int(pg[vg])
#             if p >= 0:
#                 lp = int(g2l[p])
#                 if lp >= 0:
#                     parent_local[lv] = lp
#         return parent_local

#     # build parent from children
#     children = _get_seed_children(seed_tree)
#     root_g = int(seed_tree.root_global)
#     root_l = int(g2l[root_g])
#     if root_l < 0:
#         return parent_local

#     q = deque([root_g])
#     seen_g = set([root_g])
#     while q:
#         ug = q.popleft()
#         for vg in children[ug]:
#             if vg in seen_g:
#                 continue
#             seen_g.add(vg)
#             q.append(vg)
#             lv = int(g2l[vg])
#             lu = int(g2l[ug])
#             if lv >= 0 and lu >= 0:
#                 if parent_local[lv] != -1 and parent_local[lv] != lu:
#                     raise ODDAGBuildError(
#                         f"Seed tree rooted at {root_g} is not a proper tree over V_B; "
#                         f"vertex {vg} has multiple parents."
#                     )
#                 parent_local[lv] = lu

#     return parent_local


def _build_local_children_from_local_parent(
    parent_local: np.ndarray,
    root_local: int,
) -> List[List[int]]:
    n_local = len(parent_local)
    children = [[] for _ in range(n_local)]
    for v in range(n_local):
        if v == root_local:
            continue
        p = int(parent_local[v])
        if p >= 0:
            children[p].append(v)
    return children


def _recover_tree_path_local(
    root_local: int,
    y_local: int,
    parent_local: np.ndarray,
) -> List[int]:
    """
    Recover unique root-to-y path in the seed BFS tree (local ids).
    """
    path_rev = [y_local]
    cur = y_local
    while cur != root_local:
        p = int(parent_local[cur])
        if p < 0:
            raise ODDAGBuildError(
                f"Cannot recover tree path from seed root {root_local} to hit vertex {y_local}."
            )
        path_rev.append(p)
        cur = p
    path_rev.reverse()
    return path_rev


# ============================================================
# Webbing
# ============================================================

def _insert_block_before_y(
    topo_order: List[int],
    rank: np.ndarray,
    x_local: int,
    y_local: int,
    internal_path: List[int],
) -> Tuple[List[int], np.ndarray]:
    """
    Insert internal vertices as a contiguous block immediately before y.
    Requires pi(x) < pi(y).
    """
    rx = int(rank[x_local])
    ry = int(rank[y_local])
    if rx < 0 or ry < 0:
        raise ODDAGBuildError("Cannot insert web path: x or y is not in current DAG order.")
    if not (rx < ry):
        raise ODDAGBuildError(
            f"Web path violates forward topological crossing: rank({x_local}) >= rank({y_local})."
        )

    if len(internal_path) == 0:
        return topo_order, rank

    # internal_path vertices must not yet be in topo_order
    for v in internal_path:
        if rank[v] >= 0:
            raise ODDAGBuildError(
                "Attempted to insert a web path whose internal vertex is already in the DAG."
            )

    new_order = topo_order[:ry] + internal_path + topo_order[ry:]
    new_rank = _rank_from_order(new_order, len(rank))
    return new_order, new_rank


def _web_one_seed(
    dag: StrictLocalDAG,
    seed_local: int,
    seed_parent_local: np.ndarray,
    seed_children_local: List[List[int]],
    topo_order: List[int],
    rank: np.ndarray,
) -> Tuple[List[int], np.ndarray]:
    """
    Traverse the seed BFS tree outward from the seed in BFS-tree order.

    Branch-wise semantics:
      - only first DAG hit on a branch matters
      - once a DAG hit is encountered, that branch stops permanently
      - all qualifying branches are accepted, not just the first one per seed

    We realize this by:
      - BFS from the seed over seed-tree child edges
      - do NOT expand through a non-root DAG vertex
      - each such encountered non-root DAG vertex is the first DAG hit for its branch
    """
    if not dag.is_active(seed_local):
        return topo_order, rank

    q = deque([seed_local])

    # We do not need a global "visited" array because this is a tree.
    # Still, to protect against malformed cached input, we keep one.
    seen = np.zeros(dag.n_local, dtype=np.bool_)
    seen[seed_local] = True

    while q:
        u = q.popleft()

        for v in seed_children_local[u]:
            if seen[v]:
                raise ODDAGBuildError(
                    "Seed BFS cache is not a tree / contains repeated child traversal."
                )
            seen[v] = True

            if dag.is_active(v):
                # This is the first DAG hit on this branch,
                # because we only expand through non-DAG interior nodes.
                y = v
                path = _recover_tree_path_local(seed_local, y, seed_parent_local)
                # path = [seed_local, ..., y]
                if path[0] != seed_local or path[-1] != y:
                    raise AssertionError("bad recovered path")

                if len(path) < 2:
                    # impossible here since v is a child/descendant of root
                    continue

                # downstream-cut rule
                if int(rank[seed_local]) >= int(rank[y]):
                    # backward or non-forward crossing: reject branch
                    continue

                internal = path[1:-1]

                # all internal must be feasible automatically since local universe = V_B,
                # but keep an explicit audit check:
                for z in internal:
                    if z < 0 or z >= dag.n_local:
                        raise ODDAGBuildError("Internal web-path vertex lies outside local feasible universe.")
                    if dag.is_active(z):
                        raise ODDAGBuildError(
                            "Encountered web path whose internal vertex is already in the DAG; "
                            "this violates the first-hit / new-internal-vertices rule."
                        )

                # direct-edge corner case
                if len(path) == 2:
                    x = path[0]
                    y = path[1]
                    if dag.has_edge(x, y):
                        # Allowed skip. Not a blanket dedupe.
                        continue
                    dag.add_edge_strict(x, y, context="webbing direct edge")
                    # order unchanged
                    continue

                # insert internal vertices as a contiguous block before y
                topo_order, rank = _insert_block_before_y(
                    topo_order=topo_order,
                    rank=rank,
                    x_local=seed_local,
                    y_local=y,
                    internal_path=internal,
                )

                # activate internal vertices
                for z in internal:
                    dag.activate(z)

                # add exactly the path edges
                for a, b in zip(path[:-1], path[1:]):
                    if dag.has_edge(a, b):
                        raise ODDAGBuildError(
                            "Web path attempted to insert an already-existing non-direct edge. "
                            "This should not happen in a faithful implementation."
                        )
                    dag.add_edge_strict(a, b, context="webbing path insertion")

                # branch terminates at first DAG hit; do NOT enqueue children of y
                continue

            # Non-DAG node: keep traversing this branch
            q.append(v)

    return topo_order, rank


# ============================================================
# Final compression / CSR output
# ============================================================

def _compress_final_dag_to_output_csr(
    dag: StrictLocalDAG,
    core_nodes: np.ndarray,
    topo_order: List[int],
    s_local_old: int,
    t_local_old: int,
    indptr_global: np.ndarray,
    indices_global: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, np.ndarray]:
    """
    Compress the current active DAG to a final compact local indexing.
    The output local node order is the current topological order restricted to active vertices.
    """
    final_local_old = [v for v in topo_order if dag.active[v]]
    final_n = len(final_local_old)
    if final_n == 0:
        raise ODDAGBuildError("Final DAG is empty.")

    old_to_new = np.full(dag.n_local, -1, dtype=np.int64)
    for i, old_v in enumerate(final_local_old):
        old_to_new[old_v] = i

    if old_to_new[s_local_old] < 0 or old_to_new[t_local_old] < 0:
        raise ODDAGBuildError("Final DAG lost s or t, which should be impossible.")

    final_core_nodes = core_nodes[np.asarray(final_local_old, dtype=np.int64)]
    final_s_local = int(old_to_new[s_local_old])
    final_t_local = int(old_to_new[t_local_old])

    # Build final local CSR adjacency
    row_counts = np.zeros(final_n, dtype=np.int64)
    total_arcs = 0
    for old_u in final_local_old:
        new_u = int(old_to_new[old_u])
        cnt = 0
        for old_v in dag.adjacency[old_u]:
            if dag.active[old_v]:
                cnt += 1
        row_counts[new_u] = cnt
        total_arcs += cnt

    final_indptr = np.zeros(final_n + 1, dtype=np.int64)
    final_indptr[1:] = np.cumsum(row_counts)

    final_indices = np.empty(total_arcs, dtype=np.int64)
    final_arc_ids = np.empty(total_arcs, dtype=np.int64)

    cursor = final_indptr[:-1].copy()
    for old_u in final_local_old:
        new_u = int(old_to_new[old_u])
        gu = int(core_nodes[old_u])
        for old_v in dag.adjacency[old_u]:
            if not dag.active[old_v]:
                continue
            new_v = int(old_to_new[old_v])
            gv = int(core_nodes[old_v])

            pos = int(cursor[new_u])
            final_indices[pos] = new_v
            eid = _find_first_arc_pos(indptr_global, indices_global, gu, gv)
            if eid < 0:
                raise ODDAGBuildError(
                    f"Final DAG edge ({gu},{gv}) does not exist in the global CSR graph."
                )
            final_arc_ids[pos] = eid
            cursor[new_u] += 1

    return final_indptr, final_indices, final_core_nodes, final_s_local, final_t_local, final_arc_ids


# ============================================================
# Main builder
# ============================================================

def build_od_twotree_web_csr_numba(
    indptr,
    indices,
    weights,
    s,
    t,
    alpha,
    dist_s,
    dist_t,
    pred_s,
    pred_t_rev,
    seed_idx,
    seed_ptr,
    seed_u,
    seed_v,
    touched_seeds,
):
    """
    Faithful OD-local DAG builder from:
      - exact feasible set V_B
      - two fixed shortest-consistent trees
      - tree-only halo with barrier semantics
      - required s->t prune
      - branch-wise seed-tree webbing
      - order-preserving insertion
      - no blanket dedupe / no cycle-repair

    Parameters
    ----------
    indptr, indices, weights
        Directed graph G in CSR form.
    s, t
        Global origin and destination node ids.
    alpha
        Slack factor >= 1.
    dist_s
        Shortest-path distances from s in the original graph.
    dist_t
        Shortest-path distances to t, typically from reverse-graph shortest path rooted at t.
    pred_s
        Fixed predecessor array for one shortest-path tree rooted at s.
        Must satisfy d_s(v) = d_s(pred_s[v]) + w(pred_s[v], v) wherever defined and used.
    pred_t_rev
        Fixed reverse predecessor / successor-toward-t array.
        Must satisfy d_t(u) = d_t(pred_t_rev[u]) + w(u, pred_t_rev[u]) wherever defined and used.
    seed_trees
        Precomputed BFS trees, keyed by seed global id if dict, or parallel to touched_seeds if sequence.
        Each tree may be:
          - parent array over global nodes
          - {"parent": parent_array}
          - {"children": list_of_global_children_lists}
          - SeedTree(...)
    touched_seeds
        Seeds touched by this OD, in global node ids.

    Returns
    -------
    dag_indptr : np.ndarray
        CSR row pointer of final compact local DAG.
    dag_indices : np.ndarray
        CSR column indices of final compact local DAG.
    core_nodes_out : np.ndarray
        local_id -> global_id map for final compact DAG nodes.
    s_local_out : int
        Local node id of s in final compact DAG.
    t_local_out : int
        Local node id of t in final compact DAG.
    dag_arc_ids : np.ndarray
        For each final DAG arc, the corresponding global CSR edge slot.

    Notes
    -----
    This function does not do any arbitrary graph search inside the OD builder.
    The only traversals are:
      - forward/reverse tree child traversals during halo
      - graph reachability on the base/current DAG for the allowed prune/validation
      - precomputed seed BFS tree traversals during webbing
    """
    indptr = _as_np_int(indptr)
    indices = _as_np_int(indices)
    weights = _as_np_float(weights)
    dist_s = _as_np_float(dist_s)
    dist_t = _as_np_float(dist_t)
    pred_s = _as_np_int(pred_s)
    pred_t_rev = _as_np_int(pred_t_rev)
    touched_seeds = _as_np_int(touched_seeds)

    _validate_csr(indptr, indices, weights)

    n_global = len(indptr) - 1
    if len(dist_s) != n_global or len(dist_t) != n_global:
        raise ValueError("dist_s and dist_t must have length n_global.")
    if len(pred_s) != n_global or len(pred_t_rev) != n_global:
        raise ValueError("pred_s and pred_t_rev must have length n_global.")
    if not (0 <= s < n_global and 0 <= t < n_global):
        raise ValueError("s and t must be valid global node ids.")
    if alpha < 1.0:
        raise ValueError("alpha must be >= 1.")

    # --------------------------------------------------------
    # 1) Exact feasible set V_B and local indexing over V_B
    # --------------------------------------------------------
    core_nodes, g2l, _B = _build_feasible_local_index(
        s=s,
        t=t,
        alpha=float(alpha),
        dist_s=dist_s,
        dist_t=dist_t,
    )
    n_local = len(core_nodes)
    s_local = int(g2l[s])
    t_local = int(g2l[t])

    # --------------------------------------------------------
    # 2) Restrict the two fixed shortest-consistent trees to V_B
    # --------------------------------------------------------
    parF_local, childrenF = _build_forward_tree_restricted(
        indptr=indptr,
        indices=indices,
        weights=weights,
        s=s,
        core_nodes=core_nodes,
        g2l=g2l,
        dist_s=dist_s,
        pred_s=pred_s,
    )

    parR_local, childrenR = _build_reverse_tree_restricted(
        indptr=indptr,
        indices=indices,
        weights=weights,
        t=t,
        core_nodes=core_nodes,
        g2l=g2l,
        dist_t=dist_t,
        pred_t_rev=pred_t_rev,
    )

    # --------------------------------------------------------
    # 3) Halo: tree-only, with first-meeting barriers
    # --------------------------------------------------------
    side = _run_halo(
        s_local=s_local,
        t_local=t_local,
        childrenF=childrenF,
        childrenR=childrenR,
    )

    # --------------------------------------------------------
    # 4) Base core from union of the two tree edge families
    # --------------------------------------------------------
    dag = _build_base_core(
        s_local=s_local,
        t_local=t_local,
        side=side,
        parF_local=parF_local,
        parR_local=parR_local,
    )

    # --------------------------------------------------------
    # 5) Required prune: keep only nodes on some directed s->t path
    # --------------------------------------------------------
    _prune_to_st_subdag(
        dag=dag,
        s_local=s_local,
        t_local=t_local,
    )

    # Validate/materialize topological order of the already-acyclic base core
    topo_order = _toposort_active_or_raise(dag)
    rank = _rank_from_order(topo_order, n_local)

    # --------------------------------------------------------
    # 6) Webbing from eligible touched seeds
    # --------------------------------------------------------
    seed_idx = _as_np_int(seed_idx)
    seed_ptr = _as_np_int(seed_ptr)
    seed_u = _as_np_int(seed_u)
    seed_v = _as_np_int(seed_v)

    for seed_sid_raw in touched_seeds:
        seed_sid = int(seed_sid_raw)

        if seed_sid < 0 or seed_sid + 1 >= len(seed_ptr) or seed_sid >= len(seed_idx):
            continue

        seed_global = int(seed_idx[seed_sid])
        seed_local = int(g2l[seed_global])

        # seed must lie in V_B
        if seed_local < 0:
            continue

        # seed must already be in current DAG
        if not dag.is_active(seed_local):
            continue

        seed_parent_local = _build_local_parent_from_seed_cache(
            seed_sid=seed_sid,
            seed_root_global=seed_global,
            seed_ptr=seed_ptr,
            seed_u=seed_u,
            seed_v=seed_v,
            g2l=g2l,
            n_local=n_local,
        )

        seed_children_local = _build_local_children_from_local_parent(
            parent_local=seed_parent_local,
            root_local=seed_local,
        )

        topo_order, rank = _web_one_seed(
            dag=dag,
            seed_local=seed_local,
            seed_parent_local=seed_parent_local,
            seed_children_local=seed_children_local,
            topo_order=topo_order,
            rank=rank,
        )

    # --------------------------------------------------------
    # 7) Final validation: current DAG must still be acyclic and sampler-safe
    # --------------------------------------------------------
    topo_order = _toposort_active_or_raise(dag)

    fwd_final = _forward_reachable(dag, s_local)
    if not fwd_final[t_local]:
        raise ODDAGBuildError(
            "Final DAG is not s->t connected after webbing, which is not allowed for the sampler."
        )

    # Optional but recommended second prune after webbing is NOT applied here,
    # because the requested prune was defined for the base core.
    # If you want a final semantic-preserving cleanup, it must again be the same
    # forward/reverse s->t path prune. It is not necessary for correctness here.

    # --------------------------------------------------------
    # 8) Final compact output CSR + edge-slot mapping
    # --------------------------------------------------------
    dag_indptr, dag_indices, core_nodes_out, s_local_out, t_local_out, dag_arc_ids = (
        _compress_final_dag_to_output_csr(
            dag=dag,
            core_nodes=core_nodes,
            topo_order=topo_order,
            s_local_old=s_local,
            t_local_old=t_local,
            indptr_global=indptr,
            indices_global=indices,
        )
    )

    return dag_indptr, dag_indices, core_nodes_out, s_local_out, t_local_out, dag_arc_ids