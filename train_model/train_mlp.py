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

PARAM_GRID = {
    "learning_rate": [0.0001, 0.00001],
    "batch_size": [16, 32, 64],
    "num_hidden_layers": [3, 4],
    "dropout_rate": [0.15, 0.3],
    "hidden_dim": [64, 128]
}
PARAM_COMBINATIONS = list(itertools.product(*PARAM_GRID.values()))
PARAM_KEYS = list(PARAM_GRID.keys())

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
    torch.manual_seed(seed)

def load_data(df, label, fingerprint_col="ECFP_2"):
    datasets = {split: {"X": [], "y": []} for split in ["train", "val", "test"]}

    for _, row in df.iterrows():
        split = row['split']
        if split in datasets:
            datasets[split]["X"].append(row[fingerprint_col])
            datasets[split]["y"].append(row[label])

    return {split: MyDataset(data["X"], data["y"]) for split, data in datasets.items()}

def train_and_evaluate(dataset, label, df, test_df, fingerprint_col, filename):
    best_model, best_roc_auc, best_params = None, 0, None
    datasets = load_data(df, label, fingerprint_col)
    test_loader = DataLoader(datasets["test"], batch_size=32, shuffle=False)

    checkpoint_dir = f"best_models/{dataset}/{fingerprint_col}/{label}/mlp"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_roc_acc, best_model = 0, None

    num_positive = df[label].sum()
    num_negative = len(df) - num_positive
    pos_weight = torch.tensor(num_negative / num_positive, dtype=torch.float32).to(device)

    for param_values in PARAM_COMBINATIONS:
        params = dict(zip(PARAM_KEYS, param_values))
        logging.info(f"Training with params: {params}")

        train_loader = DataLoader(datasets["train"], batch_size=params["batch_size"], shuffle=True)
        val_loader = DataLoader(datasets["val"], batch_size=params["batch_size"], shuffle=False)

        model = MLP(hidden_dim=params["hidden_dim"], num_hidden_layers=params["num_hidden_layers"],
                    dropout_rate=params["dropout_rate"], in_features=2048).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        epochs_without_improvement = 0
        patience = 10

        for epoch in range(25):
            model.train()

            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(X).squeeze()
                loss = criterion(out, y.float())
                loss.backward()
                optimizer.step()

            model.eval()
            val_labels, val_preds = [], []
            val_losses = []

            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    out = model(X).squeeze()
                    loss = criterion(out, y.float())
                    val_losses.append(loss.item())

                    val_labels.extend(y.cpu().numpy())
                    val_preds.extend(torch.sigmoid(out).cpu().detach().numpy())

            val_roc_auc = roc_auc_score(val_labels, val_preds)

            if val_roc_auc > best_val_roc_acc:
                best_val_roc_acc = val_roc_auc
                best_model = model

                best_model_path = os.path.join(checkpoint_dir, f'best_model_{filename}.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'hidden_dim': params["hidden_dim"],
                    'num_hidden_layers': params["num_hidden_layers"],
                    'dropout_rate': params["dropout_rate"],
                }, best_model_path)

                logging.info(f'Saved best model with validation ROC AUC: {val_roc_auc:.4f}')
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break

    best_model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            out = best_model(X).squeeze()
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(torch.sigmoid(out).cpu().detach().numpy())

    test_roc_auc = roc_auc_score(all_labels, all_preds)
    logging.info(f"Final Test ROC-AUC: {test_roc_auc:.4f}")
    logging.info(f"Best Params: {best_params}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MLP model.")
    parser.add_argument("--dataset", default="cdk4", help="Dataset choice.", required=False)
    parser.add_argument("--label", choices=["y", "class", "activity"], default="y", help="Y label column.", required=False)
    parser.add_argument("--fingerprint", choices=['FGP', 'ECFP_2', 'ECFP_count_2', 'ECFP_3', 'ECFP_count_3', 'KLEKOTA',
                                                   'KLEKOTA_count', 'MORDRED'], default="ECFP_2", required=False, help="Fingerprint method.")
    parser.add_argument("--filename", required=False, default="raw", help="Dataset filename")

    args = parser.parse_args()
    dataset, label, fingerprint, filename = args.dataset, args.label, args.fingerprint, args.filename

    seed_everything(123)

    df = pd.read_parquet(f"../data/{dataset}/{filename}.parquet")
    test_df = df[df["split"] == "test"]

    train_and_evaluate(dataset, label, df, test_df, fingerprint, filename)

    logging.info("Done!")

# nohup python "train_mlp.py" --dataset 'cdk2' --label 'y' > "mlp_scaffold.log" 2>&1 &
