import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from aquaguard.graph.topology import load_static_graph
from aquaguard.data.loaders import load_scenario

g = load_static_graph("data/raw/Scenario-1/Hanoi_CMH_Scenario-1.inp")
bundle = load_scenario("data/raw/Scenario-1")

print("num_nodes:", g["num_nodes"], " num_edges:", g["num_edges"])
print("reservoir count:", sum(1 for t in g["node_type"].values() if t == "Reservoir"))
print("edge_index shape:", g["edge_index"].shape)
print("Graph node order matches pressures columns:", g["node_ids"] == list(bundle["pressures"].columns))