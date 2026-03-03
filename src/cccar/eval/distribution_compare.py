from collections import Counter
import pandas as pd
import numpy as np
from typing import Optional, Iterable, Tuple

def compare_edge_usage_distributions(
    model_edge_vol: Counter,
    replica_edge_vol: Counter,
    *,
    # Bin edges in "trips per edge" (edit to taste)
    bin_edges: Optional[Iterable[float]] = None,
    # If True, include edges with zero volume (only meaningful if you know total edge count)
    include_zeros: bool = False,
    total_edges_model: Optional[int] = None,
    total_edges_replica: Optional[int] = None,
    # printing control
    top_n_debug: int = 0,
) -> Tuple[pd.DataFrame, float]:
    """
    ID-agnostic comparison of edge-volume distributions.

    Builds histograms over per-edge volumes for model and replica, then computes
    Total Variation Distance (TV) between the normalized histograms:
        TV = 0.5 * sum_i |p_i - q_i|

    Returns:
      (table_df, tv_distance)

    Notes:
    - This intentionally does NOT attempt to match edge IDs across datasets.
    - include_zeros only makes sense if you provide total_edges_* to back-fill
      unobserved edges as 0-volume.
    """

    if bin_edges is None:
        # Reasonable default bins for traffic/link volumes: lots of mass near 0, long tail.
        # Edit these if your volumes are larger/smaller.
        bin_edges = [
            0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 1e9
        ]
    bin_edges = np.asarray(list(bin_edges), dtype=np.float64)
    if bin_edges.ndim != 1 or bin_edges.size < 2 or not np.all(np.diff(bin_edges) > 0):
        raise ValueError("bin_edges must be a strictly increasing 1D sequence with >=2 entries.")

    def _vol_array(ctr: Counter, include_zeros: bool, total_edges: Optional[int]) -> np.ndarray:
        # Positive volumes from the counter
        vals = np.fromiter((float(v) for v in ctr.values()), dtype=np.float64, count=len(ctr))
        if not include_zeros:
            return vals

        if total_edges is None:
            raise ValueError("include_zeros=True requires total_edges_model/total_edges_replica.")

        zeros = int(total_edges) - int(vals.size)
        if zeros <= 0:
            return vals
        return np.concatenate([vals, np.zeros(zeros, dtype=np.float64)], axis=0)

    model_vals = _vol_array(model_edge_vol, include_zeros, total_edges_model)
    repl_vals  = _vol_array(replica_edge_vol, include_zeros, total_edges_replica)

    # Histogram = how many edges fall into each volume bin
    model_hist, _ = np.histogram(model_vals, bins=bin_edges)
    repl_hist,  _ = np.histogram(repl_vals,  bins=bin_edges)

    # Normalize to distributions over bins
    model_total = float(model_hist.sum())
    repl_total  = float(repl_hist.sum())

    if model_total <= 0 or repl_total <= 0:
        # nothing to compare
        table = pd.DataFrame({
            "bin": [f"[{bin_edges[i]}, {bin_edges[i+1]})" for i in range(bin_edges.size - 1)],
            "model_edges_in_bin": model_hist.astype(int),
            "replica_edges_in_bin": repl_hist.astype(int),
            "model_share": np.zeros(bin_edges.size - 1),
            "replica_share": np.zeros(bin_edges.size - 1),
            "abs_diff": np.zeros(bin_edges.size - 1),
        })
        return table, float("nan")

    p = model_hist / model_total
    q = repl_hist  / repl_total
    tv = 0.5 * float(np.abs(p - q).sum())

    # Pretty bin labels
    labels = []
    for i in range(bin_edges.size - 1):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        if i == bin_edges.size - 2:
            labels.append(f"[{lo:g}, {hi:g}]")   # last bin inclusive
        else:
            labels.append(f"[{lo:g}, {hi:g})")

    table = pd.DataFrame({
        "bin": labels,
        "model_edges_in_bin": model_hist.astype(int),
        "replica_edges_in_bin": repl_hist.astype(int),
        "model_share": p,
        "replica_share": q,
        "abs_diff": np.abs(p - q),
    })

    # Optional debug: print top bins with largest differences
    if top_n_debug and top_n_debug > 0:
        dbg = table.sort_values("abs_diff", ascending=False).head(int(top_n_debug))
        print("\nTop differing bins:")
        print(dbg.to_string(index=False))

    return table, tv
