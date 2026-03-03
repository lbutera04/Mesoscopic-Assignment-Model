from typing import Optional, Tuple
import numpy as np

from .numba_kernels import _compute_out_sums, _compute_end_pos, _sample_path_uniform_dag_numba

def sample_path_uniform_dag(
    od_indptr: np.ndarray,
    od_indices: np.ndarray,
    toll: np.ndarray,
    decay: np.ndarray,
    start_loc: int,
    end_loc: int,
    rng: np.random.Generator,
    max_steps: int = 20000,
    *,
    out_sum: Optional[np.ndarray] = None,
    end_pos: Optional[np.ndarray] = None,
    path_nodes_buf: Optional[np.ndarray] = None,
    used_slots_buf: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Backward-compatible wrapper, but now returns arrays + lengths:
      (path_nodes_buf, used_slots_buf, plen, ulen)

    If out_sum/end_pos/buffers are provided, this does *zero* allocations per call.
    """
    nloc = od_indptr.shape[0] - 1
    mloc = od_indices.shape[0]

    if max_steps <= 0:
        max_steps = 1

    # Allocate only if caller didn't provide buffers
    if path_nodes_buf is None or path_nodes_buf.shape[0] < (max_steps + 1):
        path_nodes_buf = np.empty(max_steps + 1, np.int64)
    if used_slots_buf is None or used_slots_buf.shape[0] < max_steps:
        used_slots_buf = np.empty(max_steps, np.int64)

    if out_sum is None or out_sum.shape[0] != nloc:
        out_sum = np.empty(nloc, np.float64)
        _compute_out_sums(od_indptr, toll, out_sum)

    if end_pos is None or end_pos.shape[0] != nloc:
        end_pos = np.empty(nloc, np.int64)
        _compute_end_pos(od_indptr, od_indices, int(end_loc), end_pos)

    seed = np.uint64(rng.integers(1, np.iinfo(np.uint64).max, dtype=np.uint64))
    plen, ulen = _sample_path_uniform_dag_numba(
        od_indptr, od_indices, toll, decay,
        out_sum, end_pos,
        int(start_loc), int(end_loc),
        seed,
        path_nodes_buf, used_slots_buf,
    )
    return path_nodes_buf, used_slots_buf, int(plen), int(ulen)
