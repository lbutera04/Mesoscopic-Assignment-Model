from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    # Paths
    net_path: str = r"/Users/lucabutera/sumo-abq/abq.net.xml"
    bg_path: str = r"/Users/lucabutera/Downloads/tl_2020_35_bg/tl_2020_35_bg.shp"
    replica_trips_path: str = r"/Users/lucabutera/Downloads/ABQ_OD_BGs.csv"
    replica_edgevol_path: str = r"/Users/lucabutera/Downloads/network-link-volumes_spring-2025_bernalillo/network-link-volumes_spring-2025_bernalillo.csv"
    bad_edges_path: str = r"/Users/lucabutera/sumo-abq/bad_edges.txt"

    # Outputs
    out_routes_xml: str = r"/Users/lucabutera/sumo-abq/cccar_dag_allday.rou.xml"
    out_route_groups_csv: str = r"/Users/lucabutera/sumo-abq/cccar_dag_allday_route_groups.csv"
    out_routes_centroid_csv: str = r"/Users/lucabutera/sumo-abq/cccar_dag_allday_centroid_splits.csv"

    # Demand sampling (routing uses aggregated OD totals; dep_bin is only for flow emission)
    demand_scale: float = 1.0
    dep_bin_minutes: float = 5.0
    dep_bin_str: str = "5min"
    bins_per_day: int = 288  # 24h / 5min

    # Drivable filtering
    allowed_vtypes: Tuple[str, ...] = ("truck", "car")

    # Optional "bad edge" penalty multiplier
    bad_edge_penalty: float = 1.0

    # Corridor slack as multiplicative budget on SSP time
    dag_slack: float = 1.25

    # How many diverse routes per centroid OD
    routes_per_od: int = 3

    # Routing model selection
    dag_model: str = "mixed"  # {"mixed", "twotree_web"}

    # Two-tree + webbing knobs (copied from dag_benchmarks defaults)
    twotree_seed_bfs_max_hops: int = 50
    twotree_max_web_edges: int = 20000
    twotree_seed_min_incident: int = 4
    twotree_seed_min_distinct_bases: int = 2
    twotree_seed_require_link: bool = False

    # Reproducibility
    rng_seed: int = 7

    # Debug
    debug_print_examples: int = 3
