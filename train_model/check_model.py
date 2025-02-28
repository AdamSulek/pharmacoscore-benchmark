import os
import sys

import torch
import joblib
import numpy as np
import pandas as pd
import logging
import argparse
import pickle
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

def load_gcn_model(dataset, label):
    checkpoint = torch.load(f"best_models/{dataset}/{label}/gcn/best_model.pth", weights_only=False)
    input_dim = checkpoint['input_dim']
    model_dim = checkpoint['model_dim']
    dropout_rate = checkpoint['dropout_rate']
    n_layers = checkpoint['n_layers']
    num_fc_layers = checkpoint['num_fc_layers']
    fc_hidden_dim = checkpoint['fc_hidden_dim']

    model = GCN(input_dim=input_dim, model_dim=model_dim, dropout_rate=dropout_rate, n_layers=n_layers,
                num_fc_layers=num_fc_layers, fc_hidden_dim=fc_hidden_dim,
                concat_conv_layers=1).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

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


def check_gcn_model(test_data, model):
    y_proba_list = []
    y_pred_list = []
    y_label_list = []

    for test_sample in test_data:
        test_sample = test_sample.to(device)

        with torch.no_grad():
            logits = model(test_sample)
            y_proba = torch.sigmoid(logits).cpu().numpy()
            y_pred = (y_proba > 0.5).astype(int)

        y_proba_list.append(y_proba)
        y_pred_list.append(y_pred)
        y_label_list.append(np.array(test_sample.label).flatten())

    y_proba_list = np.concatenate(y_proba_list)
    y_pred_list = np.concatenate(y_pred_list)
    y_label_list = np.concatenate(y_label_list)

    evaluate_model(y_label_list, y_proba_list, y_pred_list)

def check_model(df, model, model_name, label='class'):
    if model_name == 'GCN':
        with open(f'../data/cdk2/graph_data_{label}.p', 'rb') as f:
            data_list = pickle.load(f)
        test_data = [data for data in data_list if data.split == 'test']

        check_gcn_model(test_data, model)

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
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Check model metrics on test split from dataset.")
        parser.add_argument("--model", type=str, choices=['GCN', 'MLP', 'RF', 'XGB'], required=True,
                            help="Model type to load and generate predictions.")
        parser.add_argument("--model_dataset", choices=["cdk2"], default="cdk2", required=True,
                            help="Dataset model was trained on.")
        parser.add_argument("--validate_dataset", choices=["cdk2", "cdktest", "decoy"], default="cdk2", required=True,
                            help="Dataset to check model label.")
        parser.add_argument("--model_label", choices=['class', 'activity'], default='class', required=True,
                            help="Label model was trained on.")
        parser.add_argument("--validate_label", choices=['class', 'activity', 'y'], default='class', required=True,
                            help="Label to check model on.")

    args = parser.parse_args()
    c_model, validate_dataset, model_dataset, validate_label, model_label = args.model, args.validate_dataset, args.model_dataset, args.validate_label, args.model_label

    df = pd.read_parquet(f"../data/{validate_dataset}/raw.parquet")
    test_df = df[df['split'] == 'test'] if 'split' in df.columns else df


    if c_model == 'GCN':
        model = load_gcn_model(model_dataset, model_label)

    elif c_model == 'MLP':
        model = load_mlp_model(model_dataset, model_label)

    elif c_model == 'RF':
        model = load_rf_model(model_dataset, model_label)

    elif c_model == "XGB":
        model = load_xgb_model(model_dataset, model_label)

    check_model(test_df, model, c_model, validate_label)

# python check_model.py --model 'GCN' --model_dataset 'cdk2' --validate_dataset 'cdk2' --model_label "class" --validate_label 'class'
