import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.nn as hnn
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import warnings
import argparse


warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HypergraphData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'hyperedge_index':
            return torch.tensor([[self.num_nodes], [self.num_hyperedges_count]])
        return super().__inc__(key, value, *args, **kwargs)

class HybridDataset(Dataset):
    def __init__(self, graph_list, text_embeddings, labels):
        super().__init__()
        self.graph_list = graph_list
        self.text_embeddings = text_embeddings
        self.labels = labels

    def len(self): 
        return len(self.graph_list)

    def get(self, idx):
        raw_data = self.graph_list[idx]
        data = HypergraphData()
        for key, item in raw_data:
            data[key] = item

        data.text_feat = self.text_embeddings[idx].view(1, -1)
        data.y = torch.tensor([self.labels[idx]], dtype=torch.long)
        data.num_nodes = data.x.shape[0]
        data.num_hyperedges_count = data.hyperedge_index[1].max().item() + 1

        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            if data.edge_attr.dim() == 1: 
                data.edge_attr = data.edge_attr.view(-1, 1)

        if hasattr(data, 'hyperedge_attr') and data.hyperedge_attr is not None:
            data.hyperedge_attr = data.hyperedge_attr.float()
            if data.hyperedge_attr.dim() == 1:
                data.hyperedge_attr = data.hyperedge_attr.view(-1, 1)
        return data

class CancerText(nn.Module):
    def __init__(self, args):
        super(CancerText, self).__init__()
        self.relu = nn.ReLU()
        
        self.zscore_encoder = nn.Sequential(
            nn.Linear(1, args.zscore_dim),
            nn.BatchNorm1d(args.zscore_dim),
            nn.ReLU()
        )
        self.category_embedding = nn.Embedding(num_embeddings=10, embedding_dim=args.status_dim)
        
        self.text_encoder = nn.Sequential(
            nn.Linear(args.text_in_dim, args.text_out_dim),
            nn.BatchNorm1d(args.text_out_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout)
        )

        self.gnn_in_dim = args.zscore_dim + args.status_dim + args.text_out_dim
        
        self.hyper_attr_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(64, self.gnn_in_dim),
            nn.BatchNorm1d(self.gnn_in_dim),
            nn.ReLU()
        )

        self.gat_conv = hnn.GATConv(self.gnn_in_dim, args.conv_dim, heads=args.heads, 
                                    dropout=args.dropout, edge_dim=1)
        
        self.hyper_conv = hnn.HypergraphConv(self.gnn_in_dim, args.conv_dim, heads=args.heads, 
                                             use_attention=True, dropout=args.dropout)
        
        graph_out_dim = args.conv_dim * args.heads 

        self.fusion_gate = nn.Sequential(
            nn.Linear(graph_out_dim * 2 + args.text_out_dim, 64), 
            nn.ReLU(),
            nn.Linear(64, 3), 
            nn.Softmax(dim=1)
        )

        total_dim = graph_out_dim * 2 + args.text_out_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(256, args.num_classes) 
        )

    def forward(self, data, return_att=False):
        x = data.x
        x_zscore, x_category = x[:, 0].view(-1, 1), (x[:, 1] + 1).long()
        text_raw, batch_idx = data.text_feat, data.batch 

        emb_zscore = self.zscore_encoder(x_zscore)
        emb_category = self.category_embedding(x_category)
        emb_text_graph = self.text_encoder(text_raw)
        emb_text_node = emb_text_graph[batch_idx] 

        x_init = torch.cat([emb_zscore, emb_category, emb_text_node], dim=1) 
        
        if hasattr(data, 'edge_weight') and data.edge_weight is not None:
            gat_edge_attr = data.edge_weight.view(-1, 1) 
        else:
            gat_edge_attr = data.edge_attr.float().view(-1, 1)
        
        if return_att:
            x_gat, (gat_edge_index, gat_alpha) = self.gat_conv(
                x_init, data.edge_index, edge_attr=gat_edge_attr, return_attention_weights=True
            )
        else:
            x_gat = self.gat_conv(x_init, data.edge_index, edge_attr=gat_edge_attr)
        x_gat = F.elu(x_gat)

        hyper_attr_emb = self.hyper_attr_encoder(data.hyperedge_attr) 
        if return_att:
            hyp_out_tuple = self.hyper_conv(
                x_init, data.hyperedge_index, hyperedge_attr=hyper_attr_emb, return_attention_weights=True
            )
            x_hyp = hyp_out_tuple[0]
            hyp_alpha = hyp_out_tuple[-1]
        else:
            x_hyp = self.hyper_conv(x_init, data.hyperedge_index, hyperedge_attr=hyper_attr_emb)
        x_hyp = F.elu(x_hyp)

        g_gat = global_mean_pool(x_gat, batch_idx)
        g_hyp = global_mean_pool(x_hyp, batch_idx)
 
        gate_input = torch.cat([g_gat, g_hyp, emb_text_graph], dim=1)
        weights = self.fusion_gate(gate_input) 
        
        w_gat = weights[:, 0].view(-1, 1)
        w_hyp = weights[:, 1].view(-1, 1)
        w_text = weights[:, 2].view(-1, 1)
        
        g_fused = torch.cat([g_gat * w_gat, g_hyp * w_hyp, emb_text_graph * w_text], dim=1)
        logits = self.classifier(g_fused)
        
        if return_att:
            return logits, {
                'gat_att': gat_alpha, 
                'hyp_att': hyp_alpha, 
                'gat_edge_index': gat_edge_index
            }
        return logits

def evaluate_metrics(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return acc, prec, rec, f1_weighted, f1_macro, all_labels, all_preds

def load_and_align_data(args):
    print("Loading data...")
    emb_data = torch.load(args.text_emb_file, weights_only=False)
    
    df_x = pd.DataFrame({'PatientID': emb_data['PatientIDs'], 'idx': range(len(emb_data['PatientIDs']))})
    df_x['PatientID'] = df_x['PatientID'].astype(str).str.strip()
    
    df_y = pd.read_csv(args.label_file)
    df_y['PatientID'] = df_y['PatientID'].astype(str).str.strip()
    
    graph_list_raw = torch.load(args.hypergraph_file, weights_only=False)
    df_merged = pd.merge(df_x, df_y, on='PatientID', how='inner')

    valid_graphs, valid_texts, valid_labels = [], [], []

    for _, row in df_merged.iterrows():
        orig_idx = row['idx']
        if orig_idx < len(graph_list_raw):
            g = graph_list_raw[orig_idx]
            if hasattr(g, 'hyperedge_index') and hasattr(g, 'edge_index'):
                valid_graphs.append(g)
                valid_texts.append(emb_data['embeddings'][orig_idx])
                valid_labels.append(row['Label_Code'])

    valid_texts = torch.stack(valid_texts)
    valid_labels = torch.tensor(valid_labels, dtype=torch.long)
    
    print(f"Total valid samples: {len(valid_graphs)}")
    return valid_graphs, valid_texts, valid_labels

def run_training(args):
    valid_graphs, valid_texts, valid_labels = load_and_align_data(args)
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    fold_results = []
    
    dummy_X = np.zeros(len(valid_labels))
    y_numpy = valid_labels.numpy()

    for fold, (train_idx, test_idx) in enumerate(skf.split(dummy_X, y_numpy)):
        print(f"Fold {fold+1}/{args.n_folds}")
        
        train_labels_raw = y_numpy[train_idx]
        TARGET_CLASS = 3
        lumb_indices = [train_idx[i] for i in range(len(train_idx)) if train_labels_raw[i] == TARGET_CLASS]
        augmented_train_idx = list(train_idx) + lumb_indices * 2
        
        train_dataset = HybridDataset(
            [valid_graphs[i] for i in augmented_train_idx], 
            valid_texts[augmented_train_idx], 
            valid_labels[augmented_train_idx]
        )
        test_dataset = HybridDataset(
            [valid_graphs[i] for i in test_idx], 
            valid_texts[test_idx], 
            valid_labels[test_idx]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        train_labels_augmented = y_numpy[augmented_train_idx] 
        class_weights_arr = compute_class_weight('balanced', classes=np.unique(train_labels_augmented), y=train_labels_augmented)
        class_weights = torch.tensor(class_weights_arr, dtype=torch.float).to(device)

        model = CancerText(args).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

        best_fold_f1 = 0.0
        best_fold_wts = copy.deepcopy(model.state_dict())
        patience_counter = 0

        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            loop = tqdm(train_loader, desc=f'Fold {fold+1} Ep {epoch+1}', leave=False, ascii=True)
            
            for batch in loop:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            scheduler.step()
            val_acc, val_prec, val_rec, val_f1_w, val_f1_m, _, _ = evaluate_metrics(model, test_loader, device)
            
            if val_f1_w > best_fold_f1:
                tqdm.write(f"Fold {fold+1} Ep {epoch+1:03d} | Acc: {val_acc:.2%} | F1_W: {val_f1_w:.4f}")
                best_fold_f1 = val_f1_w
                best_fold_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), f"fold_{fold+1}_best_model.pth")
                patience_counter = 0 
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    tqdm.write(f"Early stopping at epoch {epoch+1}")
                    break
        
        model.load_state_dict(best_fold_wts)
        final_acc, final_prec, final_rec, final_f1_w, final_f1_m, final_labels, final_preds = evaluate_metrics(model, test_loader, device)

        print(f"Fold {fold+1} Final: Acc={final_acc:.2%} | F1_W={final_f1_w:.4f}")
        fold_results.append({
            'fold': fold + 1, 'acc': final_acc, 'prec': final_prec,
            'rec': final_rec, 'f1_w': final_f1_w, 'f1_m': final_f1_m
        })
        print(confusion_matrix(final_labels, final_preds))

    return fold_results

def save_summary(fold_results, save_name="final_results_summary.csv"):
    df = pd.DataFrame(fold_results)
    numeric_cols = ['acc', 'f1_w', 'f1_m', 'prec', 'rec']
    means = df[numeric_cols].mean()
    stds = df[numeric_cols].std()
    
    print(f"\nOverall: Acc: {means['acc']:.2%} | F1-W: {means['f1_w']:.4f}")

    summary_row = {'fold': 'Average'}
    for col in numeric_cols:
        summary_row[col] = f"{means[col]:.4f} ± {stds[col]:.4f}"
    
    df_final = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    df_final.to_csv(save_name, index=False)
    print(f"Results saved to {save_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_emb_file", type=str, default="../Text_Embeddings.pt")
    parser.add_argument("--label_file", type=str, default="../Label.csv")
    parser.add_argument("--hypergraph_file", type=str, default="../patient_hypergraphs_dataset.pt")
    
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--n_folds", type=int, default=5)
    
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--graph_node_dim", type=int, default=2)
    parser.add_argument("--text_in_dim", type=int, default=768)
    parser.add_argument("--zscore_dim", type=int, default=32)
    parser.add_argument("--status_dim", type=int, default=32)
    parser.add_argument("--text_out_dim", type=int, default=16)
    parser.add_argument("--conv_dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)

    args = parser.parse_args()
    print(f"Device: {device}")
    
    results = run_training(args)
    save_summary(results)