import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
import tqdm


graph_file = "..\Edge_Construct_rules.csv"        
patient_file = "raw_data_forrelation.csv"  
output_file = "patient_graphs_dataset.pt"      

def get_bio_status(col_name_raw, value):
    name = str(col_name_raw).upper()
    if "MRNA" in name:
        if value > 1.30: return 1.0
        if value < -1.92: return -1.0

    elif "METH" in name:
        if value > 1.50: return 1.0
        if value < -1.47: return -1.0

    elif "MIRNA" in name:
        if value > 1.99: return 1.0
        if value < -1.25: return -1.0

    elif "CNV" in name: 
        if value < -1.10: return -1.0 # Deletion
        if -1.10 <= value < -0.10: return 0.0 # Normal
        if -0.10 <= value < 0.90: return 1.0 # Gain
        if 0.90 <= value < 1.90: return 2.0 # Amp
        if value >= 1.90: return 3.0 # HighAmp

    return 0.0
    


graph_df = pd.read_csv(graph_file)
all_nodes = pd.concat([graph_df['Source'], graph_df['Target']]).unique()
all_nodes.sort() 

node_to_idx = {node: i for i, node in enumerate(all_nodes)}
num_nodes = len(all_nodes)

src_idx = [node_to_idx[n] for n in graph_df['Source']]
dst_idx = [node_to_idx[n] for n in graph_df['Target']]
edge_index = torch.tensor([src_idx, dst_idx], dtype = torch.long)

edge_encoder = LabelEncoder()
edge_type_idx = edge_encoder.fit_transform(graph_df['Edge_Type'])
edge_attr = torch.tensor(edge_type_idx, dtype = torch.long)

def apply_sign(row):
    weight = row['Weight']
    edge_type = str(row['Edge_Type']).upper()

    if "NEG" in edge_type:
        return -1.0 * abs(weight) 
    else:
        return abs(weight) 

signed_weights = graph_df.apply(apply_sign, axis = 1)
edge_weight = torch.tensor(signed_weights.values, dtype = torch.float)
print("Sample Edge Weights (First 10):", edge_weight[:10].numpy())

patient_df = pd.read_csv(patient_file)
patient_df.set_index('PatientID', inplace = True)

clean_to_raw = {}
for raw_col in patient_df.columns:
    clean_col = str(raw_col).strip()
    for p in ['mRNA_', 'Methylation_', 'CNV_', 'miRNA_', 'Gene_']:
        if clean_col.startswith(p):
            clean_col = clean_col.replace(p, "")
            break
    clean_to_raw[clean_col] = raw_col

dataset_list = []
for patient_id, row in tqdm.tqdm(patient_df.iterrows(), total = len(patient_df)):
    
    node_features = []
    for node_name in all_nodes:
        raw_col = clean_to_raw.get(node_name)
        
        val = 0.0
        status = 0.0

        if raw_col and raw_col in row:
            val = row[raw_col] 
            status = get_bio_status(raw_col, val) 

        node_features.append([val, status])
    
    x = torch.tensor(node_features, dtype = torch.float)
    

    data = Data(
        x = x, 
        edge_index = edge_index,   
        edge_attr = edge_attr,     
        edge_weight = edge_weight, 
        patient_id = str(patient_id)
    )
    
    dataset_list.append(data)

torch.save(dataset_list, output_file)

