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

PARAM_GRID = {
    "lr_values": [0.001, 0.0001],  # Fixed to your new lr values
    "batch_sizes": [32],  # Using batch_size 32
    "conv_layers": [3, 4],  # Conv layers (number of GCN layers)
    "model_dims": [512],  # Model dimension, fixed to 512
    "dropout_rate": [0.0, 0.1],  # Dropout rates
    "fc_hidden_dim": [128, 256],  # FC hidden layers dimensions
    "num_fc_layers": [1, 2]  # Number of FC layers
}

PARAM_COMBINATIONS = list(itertools.product(*PARAM_GRID.values()))
PARAM_KEYS = list(PARAM_GRID.keys())

def train(model, train_loader, optimizer, criterion, threshold=0.5):
    model.train()
    total_loss = 0
    all_labels, all_predictions = [], []

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
    
    roc_auc = roc_auc_score(all_labels, all_predictions)

    return total_loss / len(train_loader), roc_auc

def test(model, loader, threshold=0.5):
    model.eval()
    all_labels, all_predictions = [], []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data).squeeze()
            probabilities = torch.sigmoid(out)
            all_labels.extend(data.label.cpu().numpy())
            all_predictions.extend(probabilities.detach().cpu().numpy().flatten())
    
    roc_auc = roc_auc_score(all_labels, all_predictions)

    return roc_auc

def load_data(dataset, label):
    with open(f'../data/{dataset}/graph_data_{label}.p', 'rb') as f:
        data_list = pickle.load(f)

    train_data, val_data, test_data = [], [], []

    for data in data_list:
        if data.split == 'train':
            train_data.append(data)
        elif data.split == 'val':
            val_data.append(data)
        elif data.split == 'test':
            test_data.append(data)

    return train_data, val_data, test_data

def run_training(dataset, concat_conv_layers, label):
    train_data, val_data, test_data = load_data(dataset, label)
    train_labels = [e.label for e in train_data]
    num_positive = sum(train_labels)
    num_negative = len(train_labels) - num_positive
    pos_weight = num_negative / num_positive

    best_models_dir = f'best_models/{dataset}/class/gcn'
    os.makedirs(best_models_dir, exist_ok=True)
    best_val_roc = -1

    for lr, batch_size, n_gcn_layers, model_dim, dropout_rate, fc_hidden_dim, num_fc_layers in PARAM_COMBINATIONS:
        logging.info(f"Training model with lr={lr}, batch_size={batch_size}, n_layers={n_gcn_layers}, "
                     f"model_dim={model_dim}, dropout_rate={dropout_rate}, concat_conv_layers={concat_conv_layers}")

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
        epochs_without_improvement = 0

        for epoch in range(100):
            train_loss, train_roc_auc = train(model, train_loader, optimizer, criterion, threshold=0.5)
            val_roc_auc = test(model, val_loader)

            logging.info(f'Epoch {epoch + 1}: Train Loss {train_loss:.4f}, Train ROC AUC {train_roc_auc:.4f}')

            if val_roc_auc > best_val_roc:
                best_val_roc = val_roc_auc
                best_model_path = os.path.join(best_models_dir, f'best_model.pth')
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
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logging.info(f'Early stopping at epoch {epoch + 1} due to no improvement in validation ROC AUC.')
                break

    logging.info("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training a GCN model with hyperparameter tuning.")
    parser.add_argument("--concat_conv_layers", type=int, default=1, help="Enable or disable concatenation of conv layers")
    parser.add_argument("--dataset", choices=["cdk2"], default="cdk2", required=False, help="Dataset choice")
    parser.add_argument("--label", choices=['class', 'activity'], default='class', required=False, help="Label column")
    args = parser.parse_args()

    run_training(args.dataset, args.concat_conv_layers, args.label)


# nohup python train_gcn.py --dataset 'cdk2' --label 'class' > "logs/cdk2_gcn/train_graph_cdk2_gcn.log" 2>&1 &
