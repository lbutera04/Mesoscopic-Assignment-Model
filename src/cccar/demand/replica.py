import pandas as pd
import numpy as np
from typing import Dict
import re

def load_replica(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["origin_bgrp_fips_2020"] = df["origin_bgrp_fips_2020"].astype(str).str.zfill(12)
    df["destination_bgrp_fips_2020"] = df["destination_bgrp_fips_2020"].astype(str).str.zfill(12)
    # Expect at least origin/destination BG + vtype_key + dep_bin (string bin label)
    return df

def map_replica_to_edges(
    replica_df: pd.DataFrame,
    spawn_tables: Dict[str, pd.DataFrame],
    rng: np.random.Generator,
    dep_bin_str: str,
    spread: bool = True,
    weighted_rounds: bool = False,
) -> pd.DataFrame:
    """
    Maps each replica trip row to a spawn edge and a despawn edge.
    This function preserves the prior I/O expectations in downstream code:
    - origin_bgrp_fips_2020
    - destination_bgrp_fips_2020
    - vtype_key
    - dep_bin (time bin label)
    plus adds spawn/despawn edges later.
    """
    df = replica_df.copy()

    for c in ["origin_bgrp_fips_2020", "destination_bgrp_fips_2020"]:
        if c in df.columns:
            # Replica often comes as int/float -> loses leading zeros.
            # Normalize to 12-digit GEOID string.
            df[c] = (
                df[c]
                .astype("string")
                .str.replace(r"\.0$", "", regex=True)
                .str.zfill(12)
            )

    if "dep_bin" not in df.columns:

        # Convert dep_bin_str like "5min" → seconds
        m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*min\s*$", str(dep_bin_str))
        bin_minutes = float(m.group(1)) if m else 5.0
        bin_seconds = bin_minutes * 60.0

        if "trip_start_time" in df.columns:

            # Convert "hh:mm:ss" → seconds
            hms = df["trip_start_time"].astype(str).str.split(":", expand=True)

            if hms.shape[1] != 3:
                raise ValueError("trip_start_time must be formatted as hh:mm:ss")

            h = hms[0].astype(float)
            m_ = hms[1].astype(float)
            s = hms[2].astype(float)

            t = (3600*h + 60*m_ + s).to_numpy(dtype=np.float64)

            # Bin to interval start
            b = np.floor(t / bin_seconds) * bin_seconds

            # Store numeric seconds
            df["dep_bin"] = b

        else:
            # Fallback (old behavior)
            df["dep_bin"] = dep_bin_str

    # origin
    spawn_edges = []
    for bg in df["origin_bgrp_fips_2020"].to_numpy():
        tab = spawn_tables.get(bg)
        if tab is None:
            spawn_edges.append(np.nan)
            continue
        w = tab["w"].to_numpy(dtype=np.float64)
        p = w / w.sum()
        spawn_edges.append(tab["edge_id"].to_numpy()[rng.choice(len(tab), p=p)])
    df["spawn_edge"] = spawn_edges

    # destination: re-use same sampling logic from edges_gdf membership
    # We build a quick lookup from BG to candidate edges using the same schema.
    # If your previous file had special despawn heuristics, swap them here.
    dest_tables = {}
    for bg in df["destination_bgrp_fips_2020"].value_counts().index:
        # best-effort: use same tables if available
        tab = spawn_tables.get(bg)
        if tab is None:
            dest_tables[bg] = None
        else:
            dest_tables[bg] = tab
    dest_edges = []
    for bg in df["destination_bgrp_fips_2020"].to_numpy():
        tab = dest_tables.get(bg)
        if tab is None:
            dest_edges.append(np.nan)
            continue
        w = tab["w"].to_numpy(dtype=np.float64)
        p = w / w.sum()
        dest_edges.append(tab["edge_id"].to_numpy()[rng.choice(len(tab), p=p)])
    df["despawn_edge"] = dest_edges

    return df
