from collections import Counter
import pandas as pd

def compute_edge_total_counts(route_paths: dict[str, list[str]], route_groups: pd.DataFrame) -> Counter:
    totals = Counter()
    # join groups -> route count
    route_to_count = route_groups.groupby("route_id")["count"].sum().to_dict()
    for rid, seq in route_paths.items():
        c = int(route_to_count.get(rid, 0))
        for e in seq:
            totals[e] += c
    return totals


def load_replica_link_volume_counts(path: str) -> Counter:
    """
    Loads Replica link volume file(s) into a Counter keyed by link identifier.
    Supports multiple known schemas.
    """
    df = pd.read_csv(path)

    # Edge/link identifier candidates
    edge_candidates = (
        "edge_id", "edge", "link_id", "id",
        "networkLinkId",      # <-- Replica file you attached
        "osmId",              # sometimes used as link key
    )

    # Volume candidates
    vol_candidates = (
        "volume", "vol", "count", "veh", "vehicles",
        "trip_count",         # <-- Replica file you attached
    )

    edge_col = next((c for c in edge_candidates if c in df.columns), None)
    vol_col  = next((c for c in vol_candidates if c in df.columns), None)

    if edge_col is None or vol_col is None:
        return Counter()

    ctr = Counter()
    # robust numeric parse
    vol = pd.to_numeric(df[vol_col], errors="coerce").fillna(0.0).to_numpy()
    for e, v in zip(df[edge_col].astype(str).to_numpy(), vol):
        if v:
            ctr[e] += float(v)
    return ctr
