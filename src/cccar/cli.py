import sumolib
import numpy as np
import pandas as pd

from config import Config
from osm.attributes import load_bad_edges, build_edge_attributes
from osm.graph_build import build_connection_graph_no_internals, build_csr_from_graph
from osm.geo import build_edges_gdf, attach_block_groups
from demand.replica import load_replica, map_replica_to_edges
from demand.spawns import build_spawn_tables
from demand.centroids import build_bg_centroid_edge_map
from sampling.api import dag_sample_centroid_od_paths
from routes.build import build_routes_with_centroid_trunks
from routes.sumo_io import write_sumo_routes_xml
from eval.link_volumes import compute_edge_total_counts, load_replica_link_volume_counts
from eval.distribution_compare import compare_edge_usage_distributions

def main() -> None:
    cfg = Config()
    rng = np.random.default_rng(cfg.rng_seed)

    print("\n==================== CCCAR (MIXED-POTENTIAL DAG, ALL-DAY) ====================")

    # Step 1: load net, build graph
    print("\nStep 1: Load SUMO net and build edge-connection graph")
    net = sumolib.net.readNet(cfg.net_path)

    bad_edges = load_bad_edges(cfg.bad_edges_path)
    edge_attrs = build_edge_attributes(net, cfg.allowed_vtypes, bad_edges, cfg.bad_edge_penalty)
    G = build_connection_graph_no_internals(net, edge_attrs, cfg.allowed_vtypes)

    print(f"  Drivable edges (nodes): {len(G.nodes()):,}")
    print(f"  Connection arcs:        {G.number_of_edges():,}")

    # Step 2: geodata + BG
    print("\nStep 2: Build edges GeoDataFrame and attach block groups")
    edges_gdf = build_edges_gdf(net, edge_attrs)
    edges_gdf = attach_block_groups(edges_gdf, cfg.bg_path)

    print("edges CRS:", edges_gdf.crs)
    print("bg_geoid NaN rate:", edges_gdf["bg_geoid"].isna().mean())
    print("example bg_geoids:", edges_gdf["bg_geoid"].dropna().astype(str).head().tolist())

    # Step 3: load Replica + spawn/despawn edges
    print("\nStep 3: Load Replica trips and map to spawn/despawn edges")
    replica_df = load_replica(cfg.replica_trips_path)
    for c in ["origin_bgrp_fips_2020", "destination_bgrp_fips_2020"]:
        if c in replica_df.columns:
            replica_df[c] = (
                replica_df[c]
                .astype("string")
                .str.replace(r"\.0$", "", regex=True)
                .str.zfill(12)
            )

    origin_bg_counts = replica_df["origin_bgrp_fips_2020"].value_counts().rename("trip_count")
    spawn_tables = build_spawn_tables(edges_gdf, origin_bg_counts)

    mapped = map_replica_to_edges(
        replica_df,
        spawn_tables,
        rng=rng,
        dep_bin_str=cfg.dep_bin_str,
        spread=True,
        weighted_rounds=False,
    )

    print("spawn_edge NaN rate:", mapped["spawn_edge"].isna().mean())
    print("despawn_edge NaN rate:", mapped["despawn_edge"].isna().mean())
    print("unique origin BG examples:", mapped["origin_bgrp_fips_2020"].head().tolist())
    print("spawn_tables key example:", next(iter(spawn_tables.keys())) if spawn_tables else None)


    mapped = mapped[mapped["origin_bgrp_fips_2020"] != mapped["destination_bgrp_fips_2020"]].copy()

    if cfg.demand_scale < 1.0:
        before = len(mapped)
        mapped = mapped.sample(frac=cfg.demand_scale, random_state=cfg.rng_seed).reset_index(drop=True)
        print(f"  Demand scaling: kept {len(mapped):,} / {before:,} inter-BG trips ({100*cfg.demand_scale:.1f}%)")
    else:
        print("  Demand scaling: 1.0 (no scaling)")

    # Step 4: centroid edges + OD pairs (time-independent)
    print("\nStep 4: Assign BG centroid edges and build time-independent OD demand")
    bg_centroid_edge = build_bg_centroid_edge_map(edges_gdf, cfg.bg_path, prefer_trunk=False)
    bg_centroid_edge = {str(k).zfill(12): v for k, v in bg_centroid_edge.items()}
    mapped["origin_centroid_edge"] = mapped["origin_bgrp_fips_2020"].map(bg_centroid_edge)
    mapped["dest_centroid_edge"] = mapped["destination_bgrp_fips_2020"].map(bg_centroid_edge)

    before = len(mapped)
    mapped = mapped.dropna(subset=["origin_centroid_edge", "dest_centroid_edge", "spawn_edge", "despawn_edge"]).copy()
    print(f"  Centroid assignment kept {len(mapped):,} / {before:,} trips")

    # Ensure vtype_key exists (DROP-IN FIX)

    if "vtype_key" not in mapped.columns:
        if "primary_mode" in mapped.columns:
            mapped["vtype_key"] = (
                mapped["primary_mode"]
                .astype(str)
                .str.lower()
                .apply(lambda x: "truck" if ("truck" in x or "commercial" in x) else "car")
            )
        else:
            # Safe fallback if Replica schema differs
            mapped["vtype_key"] = "car"

    od_pairs = (
        mapped
        .groupby(["origin_centroid_edge", "dest_centroid_edge", "vtype_key"], observed=True)
        .size()
        .reset_index(name="trip_count")
    )
    print(f"  OD pairs (time-independent): {len(od_pairs):,}")

    # CSR build
    print("\nPrep: Build CSR adjacency (travel-time)")
    nodes = list(G.nodes())
    idx_map = {n: i for i, n in enumerate(nodes)}
    A = build_csr_from_graph(G, nodes)
    indptr, indices, base_w = A.indptr.astype(np.int64), A.indices.astype(np.int64), A.data.astype(np.float64)

    # Step 5: DAG corridor sampling per OD
    print("\nStep 5: Mixed-potential DAG corridor construction + uniform sampling")
    od_paths = dag_sample_centroid_od_paths(
        od_pairs=od_pairs,
        nodes=nodes,
        idx=idx_map,
        indptr=indptr,
        indices=indices,
        base_w=base_w,
        edge_attrs=edge_attrs,
        cfg=cfg,
        rng=rng,
    )

    # centroid splits CSV
    rows = []
    for (c_o, c_d), plist in od_paths.items():
        for pid, (seq, f) in enumerate(plist):
            rows.append({
                "origin_centroid_edge": c_o,
                "dest_centroid_edge": c_d,
                "path_id": pid,
                "assigned": int(f),
                "path_len_edges": len(seq),
            })
    pd.DataFrame(rows).to_csv(cfg.out_routes_centroid_csv, index=False)
    print(f"  Wrote centroid splits: {cfg.out_routes_centroid_csv}")

    # Step 6: build full routes (BG join); dep_bin is only for flows
    print("\nStep 6: Building full routes (BG join) and time-binned flow groups")
    route_groups, route_paths = build_routes_with_centroid_trunks(
        G=G,
        mapped=mapped,
        od_paths={(k[0], k[1]): v for k, v in od_paths.items()},
        rng=rng,
        cfg=cfg,
    )
    print(f"  Full routes: {len(route_paths):,} unique sequences")
    print(f"  Route groups: {len(route_groups):,} rows")

    route_groups.to_csv(cfg.out_route_groups_csv, index=False)
    print(f"  Wrote route groups: {cfg.out_route_groups_csv}")

    # Replica compare
    model_edge_counts = compute_edge_total_counts(route_paths, route_groups)
    replica_counts = load_replica_link_volume_counts(cfg.replica_edgevol_path)
    table, tv = compare_edge_usage_distributions(
                    model_edge_counts,
                    replica_counts,
                    )

    print("\nEdge volume distribution compare (ID-agnostic)")
    print(table.to_string(index=False))
    print(f"\nTotal Variation Distance (TV): {tv:.4f}")

    # Step 7: write SUMO routes
    write_sumo_routes_xml(route_paths, route_groups, cfg.out_routes_xml, dep_bin_minutes=cfg.dep_bin_minutes)
    print("\nDONE.\n")

if __name__ == "__main__":
    main()