import geopandas as gpd
from typing import Dict

def build_bg_centroid_edge_map(edges_gdf: gpd.GeoDataFrame, bg_path: str, prefer_trunk: bool = False) -> Dict[str, str]:
    """
    For each BG polygon, choose a representative "centroid edge".
    This is a best-effort replacement preserving prior interface.
    """
    bg = gpd.read_file(bg_path)[["GEOID", "geometry"]]
    if edges_gdf.crs != bg.crs:
        edges_gdf = edges_gdf.to_crs(bg.crs)

    out = {}
    # choose longest edge intersecting BG as centroid proxy
    for geoid, geom in zip(bg["GEOID"], bg["geometry"]):
        sub = edges_gdf[edges_gdf.intersects(geom)]
        if len(sub) == 0:
            continue
        pick = sub.iloc[sub["length"].argmax()]
        out[str(geoid)] = str(pick["edge_id"])
    return out
