import pandas as pd
from sklearn.preprocessing import StandardScaler
from mrmr import mrmr_classif

print('----------------------------------------------Reading data----------------------------------------------')
labels = pd.read_csv('../Patient_Labels.csv', index_col = 0)
print(labels.head(1))
y = labels['Subtype']
y = y[y.isin(['LumA','LumB','Her2','Basal'])]
samples = y.index
print(samples)

mrna = pd.read_csv("../Features_mRNA.csv", index_col = 0).loc[samples]        # 3000
mirna = pd.read_csv("../Features_miRNA.csv", index_col = 0).loc[samples]      # 800
meth = pd.read_csv("../Features_Methylation.csv", index_col = 0).loc[samples] # 2000
cnv = pd.read_csv("../Features_CNV.csv", index_col = 0).loc[samples]          # 2000

print(f"The number of samples: {len(samples)}")
print(f"mRNA: {mrna.shape[1]} Features")
print(f"miRNA: {mirna.shape[1]} Features")
print(f"Methylation: {meth.shape[1]} Features")
print(f"CNV: {cnv.shape[1]} Features")

print('----------------------------------------------Preprocessing----------------------------------------------')
def preprocess_omics(df):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), index = df.index, columns = df.columns)

mrna_s = preprocess_omics(mrna)
mirna_s = preprocess_omics(mirna)
meth_s = preprocess_omics(meth)
cnv_s = preprocess_omics(cnv)


quotas = {
    'mRNA': 30,         
    'Methylation': 20,  
    'miRNA': 10,        
    'CNV': 5            
}

omics_map = {
    'mRNA': mrna_s,
    'miRNA': mirna_s,
    'Methylation': meth_s,
    'CNV': cnv_s
}

final_selected_features = []
final_dfs = []

print(f"{'Omics Type':<15} | {'Quota':<8} | {'Status'}")
print("-" * 50)

for name, quota in quotas.items():
    if name not in omics_map:
        print(f"Warning: {name} not found in data map!")
        continue
        
    df = omics_map[name]
    
    common_idx = df.index.intersection(y.index)
    
    if len(common_idx) == 0:
        print(f"Error: No common samples for {name}")
        continue
        
    X_aligned = df.loc[common_idx]
    y_aligned = y.loc[common_idx]
    
    selected_feats = mrmr_classif(X = X_aligned, y = y_aligned, K = quota, show_progress = False)
    df_subset = X_aligned[selected_feats].copy()
    
    new_columns = [f"{name}_{col}" for col in df_subset.columns]
    df_subset.columns = new_columns

    final_selected_features.extend(new_columns)
    final_dfs.append(df_subset)
    

X_signature_forced = pd.concat(final_dfs, axis=1)

print("-" * 50)
print(f"shape: {X_signature_forced.shape}")
print(f"number: {len(final_selected_features)}")

X_signature_forced.to_csv("Final_Forced_MultiOmics_Signature.csv")

with open("Final_Forced_Signature_List.txt", "w") as f:
    f.write(f"Total Features: {len(final_selected_features)}\n")
    f.write("-" * 30 + "\n")
    for feat in final_selected_features:
        f.write(feat + "\n")
