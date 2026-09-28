import yaml
import wntr
import matplotlib
matplotlib.use("Agg")  # needed if running from terminal without a display window

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

inp_path = config["data"]["nominal_inp"]
wn = wntr.network.WaterNetworkModel(inp_path)

G = wn.to_graph()
print("Number of nodes in G:", G.number_of_nodes())
print("Number of edges in G:", G.number_of_edges())

ax = wntr.graphics.plot_network(wn, title="Hanoi_CMH Network Layout", node_labels=True)
fig = ax.get_figure()
fig.savefig("hanoi_network_plot.png", dpi=150, bbox_inches="tight")
print("Saved plot to hanoi_network_plot.png")