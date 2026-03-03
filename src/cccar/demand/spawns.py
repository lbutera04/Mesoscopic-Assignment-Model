import geopandas as gpd
import numpy as np
import pandas as pd
from typing import Dict

def build_spawn_tables(edges_gdf: gpd.GeoDataFrame, origin_bg_counts: pd.Series) -> Dict[str, pd.DataFrame]:
    """
    For each origin BG, create a table of candidate edges inside that BG.
    Weighted sampling happens later in map_replica_to_edges.
    """
    tables = {}
    for bg, _ in origin_bg_counts.items():
        sub = edges_gdf[edges_gdf["bg_geoid"] == bg].copy()
        if len(sub) == 0:
            continue
        # simple weight: longer edges slightly more likely
        sub["w"] = np.maximum(sub["length"].to_numpy(), 1.0)
        tables[bg] = sub[["edge_id", "w"]]
    return tables
