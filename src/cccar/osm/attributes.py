from typing import Iterable

def lane_allows_any(lane, allowed_vtypes: Iterable[str]) -> bool:
    for vt in allowed_vtypes:
        if lane.allows(vt):
            return True
    return False


def edge_is_drivable(edge, allowed_vtypes: Iterable[str]) -> bool:
    for ln in edge.getLanes():
        if lane_allows_any(ln, allowed_vtypes):
            return True
    return False


def infer_edge_capacity(edge) -> float:
    """
    Very rough capacity proxy. If you have better edge-type capacities, swap here.
    Returns capacity in veh/hour for the edge.
    """
    # SUMO default lane capacity is typically ~1800 veh/h; scale by lane count.
    return 1800.0 * max(1, len(edge.getLanes()))


def compute_class_penalty(edge) -> float:
    """
    Optional penalty based on edge.getType() or other metadata.
    Keep 1.0 unless you deliberately want to bias away from certain classes.
    """
    return 1.0


def load_bad_edges(path: str) -> set[str]:
    try:
        with open(path, "r") as f:
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        return set()


def build_edge_attributes(
    net,
    allowed_vtypes: Iterable[str],
    bad_edge_ids: set[str],
    bad_edge_penalty: float,
) -> dict[str, dict]:
    edge_attrs: dict[str, dict] = {}

    for edge in net.getEdges():
        if not edge_is_drivable(edge, allowed_vtypes):
            continue

        eid = edge.getID()
        length = float(edge.getLength())
        speed = float(edge.getSpeed() or 0.1)
        travel_time = length / speed  # seconds
        cap_hr = infer_edge_capacity(edge)

        base_pen = compute_class_penalty(edge)
        if eid in bad_edge_ids:
            base_pen *= bad_edge_penalty

        edge_attrs[eid] = {
            "edge_obj": edge,
            "length": length,
            "speed": speed,
            "travel_time": travel_time,
            "capacity_hr": float(cap_hr),
            "is_internal": (edge.getFunction() == "internal"),
            "etype": (edge.getType() or "").lower(),
            "class_penalty": float(base_pen),
        }

    return edge_attrs

def _sumo_roadclass(edge_attrs: dict[str, dict], eid: str) -> str:
    """
    Best-effort road class for decay scheduling.
    - If SUMO edge type matches common strings, use them.
    - Else fall back to speed thresholds.
    """
    et = (edge_attrs.get(eid, {}).get("etype") or "").lower()
    for key in ("motorway", "trunk", "primary", "secondary", "tertiary", "residential", "service", "unclassified"):
        if key in et:
            return key

    spd = float(edge_attrs.get(eid, {}).get("speed", 0.0))
    # m/s thresholds (rough)
    if spd >= 30.0:   # ~67 mph
        return "motorway"
    if spd >= 25.0:   # ~56 mph
        return "trunk"
    if spd >= 20.0:   # ~45 mph
        return "primary"
    if spd >= 15.0:   # ~34 mph
        return "secondary"
    if spd >= 12.0:
        return "tertiary"
    if spd >= 8.0:
        return "residential"
    return "service"


_DECAY = {
    "motorway": 0.002,
    "trunk": 0.003,
    "primary": 0.005,
    "secondary": 0.008,
    "tertiary": 0.012,
    "unclassified": 0.050,
    "residential": 0.050,
    "service": 0.070,
}