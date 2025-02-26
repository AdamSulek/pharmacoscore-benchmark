import argparse
import itertools
import logging
import os
import random

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from mlp_model import MLP

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# lr_values = [0.001, 0.0001, 0.00001]
# batch_sizes = [16, 32, 64]
# fc_layers = [3, 4, 5]
# dropout_rates = [0.0, 0.15, 0.3]
# fc_hidden_dims = [64, 128, 256]
lr_values = [0.0001, 0.00001]
batch_sizes = [16, 32, 64]
fc_layers = [3, 4]
dropout_rates = [0.15, 0.3]
fc_hidden_dims = [64, 128]

param_grid = list(itertools.product(lr_values, batch_sizes, fc_layers, dropout_rates, fc_hidden_dims))

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.y = torch.tensor(np.array(y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)

def load_data(df, label):
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []

    for _, row in df.iterrows():
        split_value = row['split']

        if split_value == 'train':
            X_train.append(row['ECFP_2'])
            y_train.append(row[label])
        elif split_value == 'val':
            X_val.append(row['ECFP_2'])
            y_val.append(row[label])
        elif split_value == 'test':
            X_test.append(row['ECFP_2'])
            y_test.append(row[label])

    train_dataset = MyDataset(X_train, y_train)
    val_dataset = MyDataset(X_val, y_val)
    test_dataset = MyDataset(X_test, y_test)

    pos_weight = torch.tensor([len(y_train) / sum(y_train)], dtype=torch.float32)

    return pos_weight, train_dataset, val_dataset, test_dataset

def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    for data in train_loader:
        X, y = data
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(X)
        out = out.squeeze()

        loss = criterion(out, y.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        probabilities = torch.sigmoid(out)
        binary_predictions = (probabilities > 0.5).float()

        correct += (binary_predictions == y).sum().item()
        total += y.size(0)

        all_labels.extend(y.cpu().numpy())
        all_predictions.extend(probabilities.detach().cpu().numpy())

    accuracy = correct / total
    roc_auc = roc_auc_score(all_labels, all_predictions)

    return total_loss / len(train_loader), accuracy, roc_auc


def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for data in loader:
            X, y = data
            X, y = X.to(device), y.to(device)

            out = model(X)
            probabilities = torch.sigmoid(out)
            binary_predictions = (probabilities > 0.5).float()

            correct += (binary_predictions == y).sum().item()
            total += y.size(0)

            all_labels.extend(y.cpu().numpy())
            all_predictions.extend(probabilities.detach().cpu().numpy())

    accuracy = accuracy_score(all_labels, (np.array(all_predictions) > 0.5).astype(int))
    roc_auc = roc_auc_score(all_labels, all_predictions)

    precision = precision_score(all_labels, (np.array(all_predictions) > 0.5).astype(int), zero_division=0)
    recall = recall_score(all_labels, (np.array(all_predictions) > 0.5).astype(int), zero_division=0)
    f1 = f1_score(all_labels, (np.array(all_predictions) > 0.5).astype(int), zero_division=0)
    conf_matrix = confusion_matrix(all_labels, (np.array(all_predictions) > 0.5).astype(int))

    return accuracy, roc_auc, precision, recall, f1, conf_matrix

def run_training(dataset, label):
    seed_everything(123)

    df = pd.read_parquet(f'../data/{dataset}/raw.parquet')
    pos_weight, train_dataset, val_dataset, test_dataset = load_data(df, label)

    best_val_roc_acc = 0.0

    checkpoint_dir = f'best_models/{dataset}/{label}/mlp'
    result_dir = f'results/{dataset}/{label}/mlp'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    mlp_results = []

    for lr, batch_size, fc_layer, dropout_rate, fc_hidden_dim in param_grid:
        logging.info(f"Training model with lr={lr}, batch_size={batch_size}, fc_layers={fc_layer}, dropout_rate={dropout_rate}, fc_hidden_dim={fc_hidden_dim}")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = MLP(hidden_dim=fc_hidden_dim, num_hidden_layers=fc_layer, dropout_rate=dropout_rate).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], dtype=torch.float32)).to(device)

        patience = 10
        epochs_without_improvement = 0

        for epoch in range(25):
            train_loss, train_acc, train_roc_auc = train(model, train_loader, optimizer, criterion)
            val_acc, val_roc_auc, val_precision, val_recall, val_f1, val_conf_matrix = test(model, val_loader)

            logging.info(
                f'Epoch {epoch + 1}: Train Loss {train_loss:.4f}, Train Accuracy {train_acc:.4f}, Train ROC AUC {train_roc_auc:.4f}, Val Accuracy {val_acc:.4f}, Val ROC AUC {val_roc_auc:.4f}')

            mlp_results.append({
                "lr": lr,
                "batch_size": batch_size,
                "fc_layer": fc_layer,
                "dropout_rate": dropout_rate,
                "fc_hidden_dim": fc_hidden_dim,
                "val_roc_auc": val_roc_auc
            })

            if val_roc_auc > best_val_roc_acc:
                best_val_roc_acc = val_roc_auc

                best_model_path = os.path.join(checkpoint_dir, f'best_model.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'hidden_dim': fc_hidden_dim,
                    'num_hidden_layers': fc_layer,
                    'dropout_rate': dropout_rate
                }, best_model_path)

                logging.info(f'Saved best model with validation ROC AUC: {val_roc_auc:.4f}')

                test_acc, test_roc_auc, test_precision, test_recall, test_f1, test_conf_matrix = test(model, test_loader)
                logging.info(f'Test Accuracy after epoch {epoch + 1}: {test_acc:.4f}, Test ROC AUC: {test_roc_auc:.4f}, '
                             f'Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}')
                tn, fp, fn, tp = test_conf_matrix.ravel()
                logging.info(f"  Confusion Matrix (TP, TN, FP, FN): {tp}, {tn}, {fp}, {fn}")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logging.info(f'Early stopping at epoch {epoch+1} due to no improvement in validation ROC AUC.')
                break

    df_results = pd.DataFrame(mlp_results)
    df_results.to_csv(os.path.join(result_dir, f'results.csv'), index=False)
    logging.info(f"done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MLP model.")
    parser.add_argument("--dataset", choices=["cdk2"], default = "cdk2", required=False, help="Dataset choice.")
    parser.add_argument("--label", choices=['class', 'activity'], default='class', required=False, help="Y label column")
    args = parser.parse_args()
    dataset, label = args.dataset, args.label
    run_training(dataset, label)

# nohup python "train_mlp.py" > "logs/mlp_scaffold.log" 2>&1 &
# python train_mlp.py --dataset 'cdk2' --label 'activity'
