import networkx as nx
import math
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import numpy as np
from typing import Iterable

from .attributes import lane_allows_any

def build_connection_graph_no_internals(net, edge_attrs: dict[str, dict], allowed_vtypes: Iterable[str]) -> nx.DiGraph:
    """
    Nodes are SUMO edge IDs. Arc u->v exists if a lane from u connects to a lane on v.
    Arc weight is the cost of entering v: travel_time(v) * class_penalty(v).
    Internal edges are excluded from the graph entirely.
    """
    G = nx.DiGraph()

    for eid, attrs in edge_attrs.items():
        if attrs["is_internal"]:
            continue
        G.add_node(eid)

    # Iterate physical edges in SUMO net; connect based on outgoing lane connections
    for edge in net.getEdges():
        # from-edge
        for from_lane in edge.getLanes():
            u = from_lane.getEdge().getID()
            if u not in edge_attrs:
                continue
            if edge_attrs[u]["is_internal"]:
                continue
            if not lane_allows_any(from_lane, allowed_vtypes):
                continue

            for conn in from_lane.getOutgoing():
                to_lane = (conn.getToLane() if hasattr(conn, "getToLane") else conn.getTo())
                if not to_lane:
                    continue
                if not lane_allows_any(to_lane, allowed_vtypes):
                    continue

                v = to_lane.getEdge().getID()
                if v not in edge_attrs:
                    continue
                if edge_attrs[v]["is_internal"]:
                    continue

                vattrs = edge_attrs[v]
                w = float(vattrs["travel_time"] * vattrs["class_penalty"])
                if G.has_edge(u, v):
                    # keep the smallest cost if multiple lane connections exist
                    if w < G[u][v].get("weight", math.inf):
                        G[u][v]["weight"] = w
                else:
                    G.add_edge(u, v, weight=w)

    return G

def build_csr_from_graph(G: nx.DiGraph, nodes: list[str]) -> csr_matrix:
    idx = {n: i for i, n in enumerate(nodes)}
    rows = []
    cols = []
    data = []
    for u, v, attrs in G.edges(data=True):
        rows.append(idx[u])
        cols.append(idx[v])
        data.append(float(attrs.get("weight", 1.0)))
    n = len(nodes)
    return sp.csr_matrix((np.asarray(data, dtype=np.float64), (np.asarray(rows), np.asarray(cols))), shape=(n, n))
