import numpy as np
from typing import List
import pandas as pd
import networkx as nx

from config import Config

def reconstruct_from_predecessors(pred: np.ndarray, origin_i: int, dest_i: int) -> List[int]:
    """
    pred: array of predecessors where pred[v] is previous node on shortest path from origin to v.
    """
    path = []
    cur = int(dest_i)
    seen = set()
    while cur != -9999 and cur != origin_i:
        if cur in seen:
            # cycle in predecessor chain (shouldn't happen), abort
            return []
        seen.add(cur)
        path.append(cur)
        cur = int(pred[cur])
        if cur < 0:
            return []
    path.append(origin_i)
    path.reverse()
    return path

def build_routes_with_centroid_trunks(
    G: nx.DiGraph,
    mapped: pd.DataFrame,
    od_paths: dict[tuple[str, str], list[tuple[list[str], int]]],
    rng: np.random.Generator,
    cfg: Config,
):
    """
    Same external behavior as the prior pipeline:
      - Take each BG-level trip row (spawn_edge -> origin_centroid_edge -> ... -> dest_centroid_edge -> despawn_edge)
      - Use od_paths for centroid->centroid corridor choices (split by assigned counts)
      - Return:
          route_groups: DataFrame with columns used for SUMO flow emission
          route_paths: dict route_id -> list of edge IDs
    """
    # Build lookup for centroid OD -> list of (seq, assigned)
    od_lookup = od_paths

    # Build a route pool per centroid OD by expanding according to assigned
    # (route_groups will aggregate by dep_bin and route_id).
    route_paths: dict[str, list[str]] = {}
    groups = []

    route_id_counter = 0

    # Pre-expand OD pool so sampling is cheap
    expanded: dict[tuple[str, str], list[int]] = {}
    od_seq_store: dict[tuple[str, str], list[list[str]]] = {}

    for key, plist in od_lookup.items():
        seqs = [seq for (seq, _) in plist]
        cnts = [int(c) for (_, c) in plist]
        total = sum(cnts)
        if total <= 0:
            continue
        od_seq_store[key] = seqs
        expanded[key] = [i for i, c in enumerate(cnts) for _ in range(c)]

    # Iterate BG trips
    for row in mapped.itertuples(index=False):
        o_cent = getattr(row, "origin_centroid_edge")
        d_cent = getattr(row, "dest_centroid_edge")
        key = (o_cent, d_cent)
        pool = expanded.get(key)
        if not pool:
            continue

        choice_idx = pool[int(rng.integers(0, len(pool)))]
        seq_cent = od_seq_store[key][choice_idx]

        spawn = getattr(row, "spawn_edge")
        despawn = getattr(row, "despawn_edge")
        dep_bin = getattr(row, "dep_bin")
        vtype_key = getattr(row, "vtype_key")

        # stitch full route: spawn -> centroid corridor -> despawn
        full = []
        if isinstance(spawn, str) and spawn:
            full.append(spawn)
        full.extend(seq_cent)
        if isinstance(despawn, str) and despawn:
            full.append(despawn)

        rid = f"r{route_id_counter}"
        route_id_counter += 1
        route_paths[rid] = full
        groups.append({
            "route_id": rid,
            "dep_bin": dep_bin,
            "vtype_key": vtype_key,
            "count": 1,
        })

    route_groups = pd.DataFrame(groups)
    if len(route_groups) == 0:
        return route_groups, route_paths

    route_groups = route_groups.groupby(["route_id", "dep_bin", "vtype_key"], as_index=False)["count"].sum()
    return route_groups, route_paths
