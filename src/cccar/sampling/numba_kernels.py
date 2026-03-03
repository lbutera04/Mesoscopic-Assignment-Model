from numba import njit, types
import numpy as np

@njit(cache=True)
def _xorshift64star(x: np.uint64) -> np.uint64:
    x ^= (x >> np.uint64(12))
    x ^= (x << np.uint64(25))
    x ^= (x >> np.uint64(27))
    return x * np.uint64(2685821657736338717)

@njit(cache=True)
def _rand_float01(state: np.ndarray) -> float:
    state[0] = _xorshift64star(state[0])
    return float((state[0] >> np.uint64(11)) * (1.0 / 9007199254740992.0))

@njit(cache=True)
def _rand_int(state: np.ndarray, low: int, high: int) -> int:
    r = _rand_float01(state)
    return low + int(r * (high - low))

@njit(cache=True)
def _compute_out_sums(od_indptr: np.ndarray, toll: np.ndarray, out_sum: np.ndarray) -> None:
    """out_sum[u] = sum(toll[pos] for pos in outgoing(u))"""
    nloc = od_indptr.shape[0] - 1
    for u in range(nloc):
        s = int(od_indptr[u])
        e = int(od_indptr[u + 1])
        tot = 0.0
        for pos in range(s, e):
            tot += float(toll[pos])
        out_sum[u] = tot

@njit(cache=True)
def _compute_end_pos(od_indptr: np.ndarray, od_indices: np.ndarray, end_loc: int, end_pos: np.ndarray) -> None:
    """end_pos[u] = slot index pos of (u->end_loc) if present else -1."""
    nloc = od_indptr.shape[0] - 1
    for u in range(nloc):
        found = -1
        s = int(od_indptr[u])
        e = int(od_indptr[u + 1])
        for pos in range(s, e):
            if int(od_indices[pos]) == end_loc:
                found = pos
                break
        end_pos[u] = found

@njit(cache=True)
def _sample_path_uniform_dag_numba(
    od_indptr: np.ndarray,
    od_indices: np.ndarray,
    toll: np.ndarray,
    decay: np.ndarray,
    out_sum: np.ndarray,
    end_pos: np.ndarray,
    start_loc: int,
    end_loc: int,
    seed: np.uint64,
    path_nodes: np.ndarray,
    used_slots: np.ndarray,
):
    """
    IDENTICAL BEHAVIOR to v3_fast sampler, but:
      - avoids per-step sum() via maintained out_sum[u]
      - uses precomputed end_pos[u] for O(1) end-edge shortcut
      - writes into preallocated buffers (no allocations per sample)
    Returns: plen, ulen
    """
    max_steps = path_nodes.shape[0] - 1

    state = np.empty(1, np.uint64)
    state[0] = seed if seed != np.uint64(0) else np.uint64(88172645463325252)

    plen = 0
    ulen = 0
    cur = int(start_loc)
    path_nodes[plen] = cur
    plen += 1

    for _ in range(max_steps):
        if cur == end_loc:
            break

        s = int(od_indptr[cur])
        e = int(od_indptr[cur + 1])
        if s == e:
            break

        # O(1) end-edge shortcut (same logic as "scan for end neighbor then take it")
        ep = int(end_pos[cur])
        if ep >= 0:
            used_slots[ulen] = ep
            ulen += 1
            path_nodes[plen] = end_loc
            plen += 1

            # apply decay to the taken edge (same rule)
            old = float(toll[ep])
            new = old * (1.0 - float(decay[ep]))
            toll[ep] = new
            out_sum[cur] += (new - old)

            return plen, ulen

        tot = float(out_sum[cur])
        if (not np.isfinite(tot)) or tot <= 0.0:
            pos = _rand_int(state, s, e)
        else:
            r = _rand_float01(state) * tot
            c = 0.0
            pos = s
            for pp in range(s, e):
                c += float(toll[pp])
                if c >= r:
                    pos = pp
                    break

        used_slots[ulen] = pos
        ulen += 1

        # decay chosen edge and update out_sum[cur] consistently
        old = float(toll[pos])
        new = old * (1.0 - float(decay[pos]))
        toll[pos] = new
        out_sum[cur] += (new - old)

        nxt = int(od_indices[pos])
        path_nodes[plen] = nxt
        plen += 1
        cur = nxt

    return plen, ulen