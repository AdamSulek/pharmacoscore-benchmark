import pickle
import logging
from torch_geometric.loader import DataLoader
from gcn_model import GCN
import torch.optim as optim
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import numpy as np
import itertools
import pandas as pd
import argparse
import os
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(123)

lr_values = [0.001, 0.0001]#, 0.00001]  
batch_sizes = [32]#16, 32, 64]  
conv_layers = [3, 4]  
model_dims = [512]#128, 256, 512]  
dropout_rate = [0.0, 0.1]#, 0.2, 0.5] 
fc_hidden_dims = [128, 256] #64, 
num_fc_layers = [1, 2]#, 3]  

param_grid = list(itertools.product(lr_values, batch_sizes, conv_layers, model_dims, dropout_rate, fc_hidden_dims, num_fc_layers))

def train(model, train_loader, optimizer, criterion, threshold=0.5):
    model.train()
    total_loss = 0
    all_labels = []
    all_predictions = []
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data).squeeze(-1)
        loss = criterion(out, data.label.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        probabilities = torch.sigmoid(out)
        all_labels.extend(data.label.cpu().numpy())
        all_predictions.extend(probabilities.detach().cpu().numpy())
    
    accuracy = accuracy_score(all_labels, (np.array(all_predictions) > threshold).astype(int))
    roc_auc = roc_auc_score(all_labels, all_predictions)
    precision = precision_score(all_labels, (np.array(all_predictions) > threshold).astype(int))
    recall = recall_score(all_labels, (np.array(all_predictions) > threshold).astype(int))

    TP = ((np.array(all_labels) == 1) & (np.array(all_predictions) > threshold)).sum()
    FP = ((np.array(all_labels) == 0) & (np.array(all_predictions) > threshold)).sum()
    TN = ((np.array(all_labels) == 0) & (np.array(all_predictions) <= threshold)).sum()
    FN = ((np.array(all_labels) == 1) & (np.array(all_predictions) <= threshold)).sum()
    
    return total_loss / len(train_loader), accuracy, roc_auc, precision, recall, TP, FP, TN, FN

def test(model, loader, threshold=0.5):
    model.eval()
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data).squeeze()
            probabilities = torch.sigmoid(out)
            all_labels.extend(data.label.cpu().numpy())
            all_predictions.extend(probabilities.detach().cpu().numpy().flatten())
    
    accuracy = accuracy_score(all_labels, (np.array(all_predictions) > threshold).astype(int))
    roc_auc = roc_auc_score(all_labels, all_predictions)
    precision = precision_score(all_labels, (np.array(all_predictions) > threshold).astype(int))
    recall = recall_score(all_labels, (np.array(all_predictions) > threshold).astype(int))

    TP = ((np.array(all_labels) == 1) & (np.array(all_predictions) > threshold)).sum()
    FP = ((np.array(all_labels) == 0) & (np.array(all_predictions) > threshold)).sum()
    TN = ((np.array(all_labels) == 0) & (np.array(all_predictions) <= threshold)).sum()
    FN = ((np.array(all_labels) == 1) & (np.array(all_predictions) <= threshold)).sum()

    return accuracy, roc_auc, precision, recall, TP, FP, TN, FN

def identify_influential_nodes(model, data, top_k=5):
    model.eval()
    data = data.to(device)
    
    output = model(data)
    
    model.zero_grad() 
    output.backward()
    
    gradients = model.final_conv_grads  
    node_activations = model.final_conv_acts
   
    node_importance = (gradients * node_activations).sum(dim=1)  
    
    _, top_indices = torch.topk(node_importance, top_k)
    top_nodes = top_indices.cpu().numpy()  
    
    return top_nodes, node_importance.cpu().detach().numpy()


best_val_roc = 0.0  
best_model_state = None  
best_params = None

gcn_architecture = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script demonstrating argparse with a boolean flag.")
    parser.add_argument(
        "--concat_conv_layers",
        type=int,
        default=1,
        help="Enable or disable concatenation (default: True)"
    )
    args = parser.parse_args()
    
    with open('data/cdk2/graph_data.p', 'rb') as f:
        data_list = pickle.load(f)
        
    train_data = []
    val_data = []
    test_data = []

    for data in data_list:
        split_value = data.split
        
        if split_value == 'train':
            train_data.append(data)
        elif split_value == 'val':
            val_data.append(data)
        elif split_value == 'test':
            test_data.append(data)

    train_labels = [e.label for e in train_data]
    num_positive = sum(train_labels)
    num_negative = len(train_labels) - num_positive
    pos_weight = num_negative / num_positive

    checkpoint_dir = f'checkpoints/cdk2/'
    result_dir = f'results/cdk2/'
    os.makedirs(result_dir, exist_ok=True)

    for lr, batch_size, n_gcn_layers, model_dim, dropout_rate, fc_hidden_dim, num_fc_layers in param_grid:
        logging.info(f"Training model with lr={lr}, batch_size={batch_size}, n_layers={n_gcn_layers}, model_dim={model_dim}, fc_hidden_dim={fc_hidden_dim}, num_fc_layers={num_fc_layers}, concat_conv_layers={args.concat_conv_layers}, dropout_rate={dropout_rate}")
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        model = GCN(
            input_dim=42,
            model_dim=model_dim,
            concat_conv_layers=args.concat_conv_layers,
            n_layers=n_gcn_layers,
            dropout_rate=dropout_rate,
            fc_hidden_dim=fc_hidden_dim,
            num_fc_layers=num_fc_layers
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
        
        patience = 20  
        best_val_roc_epoch = -1
        epochs_without_improvement = 0

        for epoch in range(100):
            train_loss, train_acc, train_roc_auc, train_prec, train_rec, train_TP, train_FP, train_TN, train_FN = train(model, train_loader, optimizer, criterion, threshold=0.5)
            val_acc, val_roc_auc, val_prec, val_rec, val_TP, val_FP, val_TN, val_FN = test(model, val_loader)

            logging.info(f'Epoch {epoch+1}: Train Loss {train_loss:.4f}, Train Accuracy {train_acc:.4f}, Train ROC AUC {train_roc_auc:.4f}, Train Precision {train_prec:.4f}, Train Recall {train_rec:.4f}')
            logging.info(f'Val Accuracy {val_acc:.4f}, Val ROC AUC {val_roc_auc:.4f}, Val Precision {val_prec:.4f}, Val Recall {val_rec:.4f}')
            logging.info(f'Train TP: {train_TP}, FP: {train_FP}, TN: {train_TN}, FN: {train_FN}')
            logging.info(f'Val TP: {val_TP}, FP: {val_FP}, TN: {val_TN}, FN: {val_FN}')
            
            if val_roc_auc > best_val_roc:
                best_val_roc = val_roc_auc
                best_model_path = os.path.join(checkpoint_dir, f'model_split_concat_conv_layers_{args.concat_conv_layers}.pth')
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(model.state_dict(), best_model_path)
                logging.info(f'Saved best model with validation ROC AUC: {val_roc_auc:.4f}')
                logging.info(f"lr: {lr}, batch_size: {batch_size}, n_layers: {n_gcn_layers}, model_dim: {model_dim}, fc_hidden_dim: {fc_hidden_dim}, num_fc_layers: {num_fc_layers}")
                epochs_without_improvement = 0    
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= patience:
                logging.info(f'Early stopping at epoch {epoch+1} due to no improvement in validation ROC AUC.')
                break
        
        gcn_architecture.append({
            "lr": lr,
            "batch_size": batch_size,
            "n_layers": n_gcn_layers,
            "model_dim": model_dim,
            "fc_hidden_dim": fc_hidden_dim,
            "num_fc_layers": num_fc_layers,
            "val_roc_auc": best_val_roc,
            "model_path": best_model_path
        })

        df_architecture = pd.DataFrame(gcn_architecture)
        df_architecture.to_csv(os.path.join(result_dir, f'architecture_{args.concat_conv_layers}.csv'), index=False)
    
    best_model_entry = max(gcn_architecture, key=lambda x: x["val_roc_auc"]) 
    best_model_path = best_model_entry["model_path"]
    
    if best_model_path:
        logging.info(f"Loading best model from: {best_model_path}")
        best_model = GCN(
            input_dim=42,
            model_dim=best_model_entry["model_dim"],
            concat_conv_layers=args.concat_conv_layers,
            n_layers=best_model_entry["n_layers"],
            dropout_rate=dropout_rate,
            fc_hidden_dim=best_model_entry["fc_hidden_dim"],
            num_fc_layers=best_model_entry["num_fc_layers"], 
            use_hooks=True
        ).to(device)
        
        best_model.load_state_dict(torch.load(best_model_path))
        best_model.eval()

        test_acc, test_roc_auc, test_prec, test_rec, test_TP, test_FP, test_TN, test_FN = test(best_model, test_loader)
        
        total_correct = 0
        total_correct_perturbed = 0
        total_samples = 0
        results = {"index": [], "label": [], "pred_proba": [], "pred_proba_mask": [], 
                   "fidelity": [], "top_nodes": [], "importance_scores": []}

        for i, test_sample in enumerate(test_data):
            if i % 100 == 0:
                logging.info(f"Processing test sample {i + 1}/{len(test_data)}")
                
            label = test_sample.label
            test_sample = test_sample.to(device)
            
            top_nodes, importance_scores = identify_influential_nodes(best_model, test_sample)

            # Predict on Original Sample
            with torch.no_grad():
                original_logits = best_model(test_sample)
                original_prediction = torch.sigmoid(original_logits).item()
                original_label = 1 if original_prediction > 0.5 else 0
                total_correct += int(original_label == label)

            # Perturbation (Zero Out Top Nodes)
            modified_sample = test_sample.clone()
            for node in top_nodes:
                modified_sample.x[node] = torch.zeros_like(modified_sample.x[node])

            # Predict on Perturbed Sample
            with torch.no_grad():
                perturbed_logits = best_model(modified_sample)
                modified_prediction = torch.sigmoid(perturbed_logits).item()
                perturbed_label = 1 if modified_prediction > 0.5 else 0
                total_correct_perturbed += int(perturbed_label == label)

            fidelity = abs(test_acc - (total_correct_perturbed / total_samples))

            results["index"].append(i)
            results["label"].append(label)
            results["pred_proba"].append(original_prediction)
            results["pred_proba_mask"].append(modified_prediction)
            results["fidelity"].append(fidelity)
            results["top_nodes"].append(top_nodes.tolist())
            results["importance_scores"].append(importance_scores[top_nodes].tolist())
            
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(result_dir, f'fidelity_{args.concat_conv_layers}.csv'), index=False)

        logging.info(f'Test Accuracy: {test_acc:.4f}, Test ROC AUC: {test_roc_auc:.4f}, Test Precision: {test_prec:.4f}, Test Recall: {test_rec:.4f}')
        logging.info(f'Test TP: {test_TP}, FP: {test_FP}, TN: {test_TN}, FN: {test_FN}')
    else:
        logging.info("No valid TEST model was found!")
        

logging.info("Training completed!")


# nohup python train_cdk2_gcn.py > "logs/cdk2_gcn/train_graph_cdk2_gcn.log" 2>&1 &
