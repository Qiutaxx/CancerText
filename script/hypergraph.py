import torch
from torch_geometric.data import Data
import tqdm
import numpy as np


input_path = "patient_graphs_dataset.pt"
output_path = "patient_hypergraphs_dataset.pt"

def upgrade_to_hypergraph(data):
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    
    if hasattr(data, 'edge_weight') and data.edge_weight is not None:
        edge_weights = data.edge_weight
        has_weights = True
    elif hasattr(data, 'edge_attr') and data.edge_attr is not None and data.edge_attr.dtype == torch.float:
        edge_weights = data.edge_attr.squeeze()
        has_weights = True
    else:
        has_weights = False

    neighbors_dict = {i: [] for i in range(num_nodes)}
    weights_dict = {i: [] for i in range(num_nodes)}

    row, col = edge_index
    for i in range(row.size(0)):
        src = row[i].item()
        dst = col[i].item()
        if src != dst:
            neighbors_dict[src].append(dst)
            if has_weights:
                weights_dict[src].append(edge_weights[i].item())

    node_indices = []
    hyperedge_indices = []
    
    hyperedge_attrs_list = [] 
    hyperedge_id_counter = 0

    for center_node in range(num_nodes):
        neighbors = neighbors_dict[center_node]
        
        if len(neighbors) > 0:
            nodes_in_this_edge = [center_node] + neighbors
            for node in nodes_in_this_edge:
                node_indices.append(node)
                hyperedge_indices.append(hyperedge_id_counter)
            
            if has_weights:
                raw_weights = weights_dict[center_node]
                pos_ws = [w for w in raw_weights if w > 0]
                neg_ws = [w for w in raw_weights if w < 0]
                
                if len(pos_ws) > 0:
                    avg_pos = sum(pos_ws) / len(pos_ws)
                else:
                    avg_pos = 0.0
                
                if len(neg_ws) > 0:
                    avg_neg = sum(neg_ws) / len(neg_ws)
                else:
                    avg_neg = 0.0

                hyperedge_attrs_list.append([avg_pos, avg_neg])

            hyperedge_id_counter += 1

    new_hyperedge_index = torch.stack([
        torch.tensor(node_indices, dtype = torch.long),
        torch.tensor(hyperedge_indices, dtype = torch.long)
    ])
    

    new_hyperedge_attr = torch.tensor(hyperedge_attrs_list, dtype = torch.float)
    
    new_data = Data(
        x = data.x, 
        y = data.y,
        edge_index = data.edge_index,         
        edge_attr = data.edge_attr,       
        edge_weight = getattr(data, 'edge_weight', None), 
        hyperedge_index = new_hyperedge_index,
        hyperedge_attr = new_hyperedge_attr   
    )
    
    if hasattr(data, 'patient_id'):
        new_data.patient_id = data.patient_id
        
    return new_data


dataset = torch.load(input_path, weights_only = False)
new_dataset = []

for data in tqdm.tqdm(dataset):
    h_data = upgrade_to_hypergraph(data)
    new_dataset.append(h_data)

torch.save(new_dataset, output_path)
