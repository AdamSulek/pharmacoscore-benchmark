import pickle
import logging
from torch_geometric.loader import DataLoader
from gcn_model import GCN
import torch.optim as optim
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import numpy as np
import itertools
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


best_val_roc = 0.0  
best_model_state = None  
best_params = None

gcn_architecture = []

def run_training(dataset, concat_conv_layers, label):
    with open(f'../data/{dataset}/graph_data_{label}.p', 'rb') as f:
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

    best_models_dir = f'best_models/{dataset}/class/gcn'
    os.makedirs(best_models_dir, exist_ok=True)

    for lr, batch_size, n_gcn_layers, model_dim, dropout_rate, fc_hidden_dim, num_fc_layers in param_grid:
        logging.info(
            f"Training model with lr={lr}, batch_size={batch_size}, n_layers={n_gcn_layers}, model_dim={model_dim}, fc_hidden_dim={fc_hidden_dim}, num_fc_layers={num_fc_layers}, concat_conv_layers={args.concat_conv_layers}, dropout_rate={dropout_rate}")
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        model = GCN(
            input_dim=42,
            model_dim=model_dim,
            concat_conv_layers=concat_conv_layers,
            n_layers=n_gcn_layers,
            dropout_rate=dropout_rate,
            fc_hidden_dim=fc_hidden_dim,
            num_fc_layers=num_fc_layers
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))

        patience = 20
        best_val_roc = -1
        epochs_without_improvement = 0

        for epoch in range(100):
            train_loss, train_acc, train_roc_auc, train_prec, train_rec, train_TP, train_FP, train_TN, train_FN = train(
                model, train_loader, optimizer, criterion, threshold=0.5)
            val_acc, val_roc_auc, val_prec, val_rec, val_TP, val_FP, val_TN, val_FN = test(model, val_loader)

            logging.info(
                f'Epoch {epoch + 1}: Train Loss {train_loss:.4f}, Train Accuracy {train_acc:.4f}, Train ROC AUC {train_roc_auc:.4f}, Train Precision {train_prec:.4f}, Train Recall {train_rec:.4f}')
            logging.info(
                f'Val Accuracy {val_acc:.4f}, Val ROC AUC {val_roc_auc:.4f}, Val Precision {val_prec:.4f}, Val Recall {val_rec:.4f}')
            logging.info(f'Train TP: {train_TP}, FP: {train_FP}, TN: {train_TN}, FN: {train_FN}')
            logging.info(f'Val TP: {val_TP}, FP: {val_FP}, TN: {val_TN}, FN: {val_FN}')

            if val_roc_auc > best_val_roc:
                best_val_roc = val_roc_auc
                best_model_path = os.path.join(best_models_dir,
                                               f'best_model.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'input_dim': 42,
                    'model_dim': model_dim,
                    'dropout_rate': dropout_rate,
                    'concat_conv_layers': concat_conv_layers,
                    'n_layers': n_gcn_layers,
                    'fc_hidden_dim': fc_hidden_dim,
                    'num_fc_layers':  num_fc_layers
                }, best_model_path)

                logging.info(f'Saved best model with validation ROC AUC: {val_roc_auc:.4f}')
                logging.info(
                    f"lr: {lr}, batch_size: {batch_size}, n_layers: {n_gcn_layers}, model_dim: {model_dim}, fc_hidden_dim: {fc_hidden_dim}, num_fc_layers: {num_fc_layers}")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logging.info(f'Early stopping at epoch {epoch + 1} due to no improvement in validation ROC AUC.')
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


    logging.info("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script demonstrating argparse with a boolean flag.")
    parser.add_argument("--concat_conv_layers", type=int, default=1, help="Enable or disable concatenation (default: True)")
    parser.add_argument("--dataset", choices=["cdk2"], default = "cdk2", required=False, help="Dataset choice.")
    parser.add_argument("--label", choices=['class', 'activity'], default='class', required=False, help="Y label column")
    args = parser.parse_args()

    dataset, concat_conv_layers, label = args.dataset, args.concat_conv_layers, args.label
    run_training(dataset, concat_conv_layers, label)

# nohup python train_gcn.py --dataset 'cdk2' --label 'class' > "logs/cdk2_gcn/train_graph_cdk2_gcn.log" 2>&1 &
