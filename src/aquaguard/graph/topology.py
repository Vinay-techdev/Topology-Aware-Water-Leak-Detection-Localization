from pathlib import Path

import numpy as np
import wntr


def load_static_graph(inp_path):
    """
    Build a static graph representation from a LeakDB .inp file, with node ordering that
    matches load_scenario()'s pressures/demands DataFrame columns exactly (numeric-sorted
    node IDs as strings: "1", "2", ..., "32").

    Valid for every scenario of the same network: LeakDB's per-scenario Scenario-X_info.csv
    records Uncertainty_Topology_(%) = NO, meaning connectivity does not change scenario to
    scenario, only pipe length/diameter/roughness. Only the topology is used here.

    Returns a dict:
      node_ids       list[str], numeric-sorted, matching pressures.columns order
      node_index     dict[str, int], node_id -> position in node_ids
      node_type      dict[str, str], node_id -> "Junction" or "Reservoir" (from WNTR)
      edge_index     np.ndarray shape (2, 2*num_pipes), int64, both directions per pipe
      edge_pipe_id   list[str], length 2*num_pipes, the pipe name for each edge_index column
      num_nodes, num_edges
    """
    wn = wntr.network.WaterNetworkModel(str(inp_path))

    node_ids = sorted(wn.node_name_list, key=int)
    node_index = {nid: i for i, nid in enumerate(node_ids)}
    node_type = {nid: wn.get_node(nid).node_type for nid in node_ids}

    src, dst, pipe_ids = [], [], []
    for pipe_name in wn.link_name_list:
        link = wn.get_link(pipe_name)
        a, b = node_index[link.start_node_name], node_index[link.end_node_name]
        src += [a, b]
        dst += [b, a]
        pipe_ids += [pipe_name, pipe_name]

    edge_index = np.array([src, dst], dtype=np.int64)

    return {
        "node_ids": node_ids,
        "node_index": node_index,
        "node_type": node_type,
        "edge_index": edge_index,
        "edge_pipe_id": pipe_ids,
        "num_nodes": len(node_ids),
        "num_edges": len(wn.link_name_list),
    }