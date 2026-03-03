import geopandas as gpd
from shapely import LineString

def build_edges_gdf(net, edge_attrs: dict[str, dict]) -> gpd.GeoDataFrame:
    rows = []
    for eid, a in edge_attrs.items():
        if a["is_internal"]:
            continue

        e = a["edge_obj"]
        shape_xy = e.getShape()
        if shape_xy is None or len(shape_xy) < 2:
            continue

        # CRITICAL: convert SUMO local XY -> lon/lat
        shape_ll = [net.convertXY2LonLat(x, y) for (x, y) in shape_xy]
        geom = LineString(shape_ll)

        rows.append({
            "edge_id": eid,
            "length": a["length"],
            "speed": a["speed"],
            "travel_time": a["travel_time"],
            "etype": a["etype"],
            "geometry": geom,
        })

    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

def attach_block_groups(edges_gdf: gpd.GeoDataFrame, bg_path: str) -> gpd.GeoDataFrame:
    bg = gpd.read_file(bg_path)
    if edges_gdf.crs != bg.crs:
        edges_gdf = edges_gdf.to_crs(bg.crs)
    joined = gpd.sjoin(edges_gdf, bg[["GEOID", "geometry"]], how="left", predicate="intersects")
    joined = joined.rename(columns={"GEOID": "bg_geoid"})
    joined = joined.drop(columns=[c for c in joined.columns if c.startswith("index_")], errors="ignore")
    return joined
