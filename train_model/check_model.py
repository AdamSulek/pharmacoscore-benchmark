import os
import sys

import torch
import joblib
import numpy as np
import pandas as pd
import logging
import argparse
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
import shap
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'train_model')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gcn_model import GCN
from mlp_model import MLP

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fingerprint_col='ECFP_2'

def evaluate_model(y_label, y_proba, y_pred):
    roc_auc = roc_auc_score(y_label, y_proba)
    accuracy = accuracy_score(y_label, y_pred)
    precision = precision_score(y_label, y_pred, zero_division=0)
    recall = recall_score(y_label, y_pred, zero_division=0)
    f1 = f1_score(y_label, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_label, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()

    logging.info(f"  ROC AUC: {roc_auc:.4f}")
    logging.info(f"  Accuracy: {accuracy:.4f}")
    logging.info(f"  Precision: {precision:.4f}")
    logging.info(f"  Recall: {recall:.4f}")
    logging.info(f"  F1 Score: {f1:.4f}")
    logging.info(f"    TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

def load_gcn_model(dataset, use_hooks):
    if use_hooks:
        model = GCN(input_dim=42, model_dim=512, dropout_rate=0.0, n_layers=3,
                    num_fc_layers=2, fc_hidden_dim=128,
                    concat_conv_layers=1, use_hooks=True).to(device)
    else:
        model = GCN(input_dim=42, model_dim=512, dropout_rate=0.0, n_layers=3,
                    num_fc_layers=2, fc_hidden_dim=128, concat_conv_layers=1).to(device)
    model.load_state_dict(torch.load(f"best_models/{dataset}/gcn/best_model.pth"))

    return model

def load_mlp_model(dataset, label):
    checkpoint = torch.load(f"best_models/{dataset}/{label}/mlp/best_model.pth")

    hidden_dim = checkpoint['hidden_dim']
    num_hidden_layers = checkpoint['num_hidden_layers']
    dropout_rate = checkpoint['dropout_rate']

    model = MLP(hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers, dropout_rate=dropout_rate)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def load_rf_model(dataset, label):
    return joblib.load(f"best_models/{dataset}/{label}/random_forest/best_model.joblib")

def load_xgb_model(dataset, label):
    return joblib.load(f"best_models/{dataset}/{label}/xgboost/best_model.joblib")

def check_model(df, model, model_name, model_hooks=None, label='class'):
    if model_name == 'GCN':
        # TODO: Implement this
        with open('../data/cdk2/graph_data.p', 'rb') as f:
            data_list = pickle.load(f)
        test_data = [data for data in data_list if data.split == 'test']

        y_proba_list = []
        y_proba_mask_list = []
        node_importances = []

        for i, test_sample in enumerate(test_data):
            test_sample = test_sample.to(device)
            modified_sample = test_sample.clone()

            with torch.no_grad():
                original_prediction = torch.sigmoid(model(test_sample)).cpu().item()

            with torch.no_grad():
                modified_prediction = torch.sigmoid(model(modified_sample)).cpu().item()

            y_proba_list.append(original_prediction)
            y_proba_mask_list.append(modified_prediction)

    elif model_name == 'MLP':
        X_test = np.stack(df[fingerprint_col].values).astype(np.float32)
        y_label = df[label].values
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = model(X_test_tensor)
            y_proba = torch.sigmoid(logits).cpu().numpy()
        y_pred = (y_proba > 0.5).astype(int)

        evaluate_model(y_label, y_proba, y_pred)

    elif model_name in ['RF', 'XGB']:
        X_test = np.stack(df[fingerprint_col].values).astype(np.float32)
        y_label = df[label].values
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        evaluate_model(y_label, y_proba, y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check model metrics on test split from dataset.")
    parser.add_argument("--model", type=str,
                        choices=['GCN', 'MLP', 'RF', 'XGB'],
                        required=False,
                        default='GCN',
                        help="Model type to load and generate predictions.")
    parser.add_argument("--model_dataset", choices=["cdk2"], default="cdk2", required=False, help="Dataset model was trained on.")
    parser.add_argument("--validate_dataset", choices=["cdk2", "cdktest", "decoy"], default="cdk2", required=False, help="Dataset with check model label.")
    parser.add_argument("--model_label", choices=['class', 'activity'], default='class', required=False, help="Label model was trained on.")
    parser.add_argument("--validate_label", choices=['class', 'activity', 'y'], default='class', required=False, help="Label to check model on.")

    args = parser.parse_args()
    c_model, validate_dataset, model_dataset, validate_label, model_label =\
        args.model, args.validate_dataset, args.model_dataset, args.validate_label, args.model_label

    df = pd.read_parquet(f"../data/{validate_dataset}/raw.parquet")
    test_df = df
    if "split" in test_df.columns:
        test_df = df[df['split'] == 'test']

    model_hooks = None

    if c_model == 'GCN':
        load_model_fn = load_gcn_model
        model = load_model_fn(model_dataset, use_hooks=False)
        model_hooks = load_model_fn(model_dataset, use_hooks=True)  # Load a version with hooks for Grad-CAM

    elif c_model == 'MLP':
        model = load_mlp_model(model_dataset, model_label)

    elif c_model == 'RF':
        model = load_rf_model(model_dataset, model_label)

    elif c_model == "XGB":
        model = load_xgb_model(model_dataset, model_label)

    check_model(test_df, model, c_model, model_hooks, validate_label)

# python check_model.py --model 'MLP' --model_dataset 'cdk2' --validate_dataset 'decoy' --model_label "class" --validate_label 'y'
