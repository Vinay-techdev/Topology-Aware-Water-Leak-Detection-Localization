import yaml
import wntr

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

inp_path = config["data"]["nominal_inp"]

wn = wntr.network.WaterNetworkModel(inp_path)

print("Number of junctions:", wn.num_junctions)
print("Number of reservoirs:", wn.num_reservoirs)
print("Number of pipes:", wn.num_pipes)
print("Total nodes:", wn.num_nodes)
print("Total links:", wn.num_links)