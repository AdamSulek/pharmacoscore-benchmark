import os
import random
import sys

import torch
import joblib
import numpy as np
import pandas as pd
import logging
import argparse
import pickle
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, average_precision_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'train_model')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gcn_model import GCN
from mlp_model import MLP

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(y_label, y_proba, y_pred):
    print("Evaluate model")
    roc_auc = roc_auc_score(y_label, y_proba)
    pr_auc = average_precision_score(y_label, y_proba)
    accuracy = accuracy_score(y_label, y_pred)
    precision = precision_score(y_label, y_pred, zero_division=0)
    recall = recall_score(y_label, y_pred, zero_division=0)
    f1 = f1_score(y_label, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_label, y_pred)
    if len(conf_matrix.ravel()) == 4:
        tn, fp, fn, tp = conf_matrix.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    logging.info(f"  ROC AUC: {roc_auc:.4f}")
    logging.info(f"  PR AUC: {pr_auc:.4f}")
    logging.info(f"  Accuracy: {accuracy:.4f}")
    logging.info(f"  Precision: {precision:.4f}")
    logging.info(f"  Recall: {recall:.4f}")
    logging.info(f"  F1 Score: {f1:.4f}")
    logging.info(f"    TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

def load_gcn_model(dataset, label, fingerprint):
    checkpoint = torch.load(f"best_models/{dataset}/{fingerprint}/{label}/gcn/best_model.pth", weights_only=False)
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

def load_mlp_model(dataset, label, fingerprint, filename):
    checkpoint = torch.load(f"best_models/{dataset}/{fingerprint}/{label}/mlp/best_model_{filename}.pth")
    hidden_dim = checkpoint['hidden_dim']
    num_hidden_layers = checkpoint['num_hidden_layers']
    dropout_rate = checkpoint['dropout_rate']

    model = MLP(hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers, dropout_rate=dropout_rate, in_features=2048)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def load_rf_model(dataset, label, fingerprint, filename):
    return joblib.load(f"best_models/{dataset}/{fingerprint}/{label}/random_forest/best_model_{filename}.joblib")

def load_xgb_model(dataset, label, fingerprint, filename):
    return joblib.load(f"best_models/{dataset}/{fingerprint}/{label}/xgboost/best_model_{filename}.joblib")


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

def check_model(df, model, model_name, fingerprint, label='class', theshold="0.5"):
    if model_name == 'GCN':
        with open(f'../data/cdk2/graph_data_{label}.p', 'rb') as f:
            data_list = pickle.load(f)
        test_data = [data for data in data_list if data.split == 'test']

        check_gcn_model(test_data, model)

        return None, None, None

    elif model_name == 'MLP':
        X_test = np.stack(df[fingerprint].values).astype(np.float32)
        y_label = df[label].values
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        model = model.to(device)

        with torch.no_grad():
            logits = model(X_test_tensor)
            y_proba = torch.sigmoid(logits).cpu().numpy()
        y_pred = (y_proba > float(threshold)).astype(int)

        evaluate_model(y_label, y_proba, y_pred)
        return df["ID"].values, y_proba, y_label

    elif model_name in ['RF', 'XGB']:
        X_test = np.stack(df[fingerprint].values).astype(np.float32)
        y_label = df[label].values
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba > float(threshold)).astype(int)

        evaluate_model(y_label, y_proba, y_pred)
        return df["ID"].values, y_proba, y_label

def save_predictions_to_csv(ids, y_proba, y_label, output_path="predictions.csv"):
    df_out = pd.DataFrame({
        "ID": ids,
        "pred_proba": y_proba.flatten(),
        "label": y_label.flatten()
    })
    df_out.to_csv(output_path, index=False)
    logging.info(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check model metrics on test split from dataset.")
    parser.add_argument("--model", type=str, choices=['GCN', 'MLP', 'RF', 'XGB'], required=False,
                        default="RF", help="Model type to load and generate predictions.")
    parser.add_argument("--model_dataset", default="cdk4", required=False,
                        help="Dataset model was trained on.")
    parser.add_argument("--validate_dataset", default="cdk4", required=False,
                        help="Dataset to check model label.")
    parser.add_argument("--model_label", choices=['y', 'class', 'activity'], default='y', required=False,
                        help="Label model was trained on.")
    parser.add_argument("--validate_label", choices=['class', 'activity', 'y'], default='y', required=False,
                        help="Label to check model on.")
    parser.add_argument("--fingerprint", choices=['ECFP_2', 'ECFP_count_2', 'ECFP_3', 'ECFP_count_3', 'KLEKOTA',
                                               'KLEKOTA_count', 'MORDRED', "FGP"], required=False, default="ECFP_2", help="Fingerprint method.")
    parser.add_argument("--model_filename", required=False, default="raw", help="Dataset filename")
    parser.add_argument("--validate_filename", required=False, default="raw", help="Validate filename")
    parser.add_argument("--threshold", required=False, default="0.5", help="Threshold")

    args = parser.parse_args()
    c_model, validate_dataset, model_dataset, validate_label, model_label, fingerprint, model_filename, validate_filename, threshold\
        = args.model, args.validate_dataset, args.model_dataset, args.validate_label, args.model_label, args.fingerprint, args.model_filename, args.validate_filename, args.threshold

    seed_everything(123)

    df = pd.read_parquet(f"../data/{validate_dataset}/{validate_filename}.parquet")
    test_df = df[df['split'] == 'test'] if 'split' in df.columns else df

    if c_model == 'GCN':
        model = load_gcn_model(model_dataset, model_label, fingerprint)

    elif c_model == 'MLP':
        model = load_mlp_model(model_dataset, model_label, fingerprint, model_filename)

    elif c_model == 'RF':
        model = load_rf_model(model_dataset, model_label, fingerprint, model_filename)

    elif c_model == "XGB":
        model = load_xgb_model(model_dataset, model_label, fingerprint, model_filename)

    ids, y_proba, y_label = check_model(test_df, model, c_model, fingerprint, validate_label)
    os.makedirs("predictions", exist_ok=True)
    output_file = f"predictions/{c_model}_{model_dataset}_{fingerprint}_{model_filename}.csv"
    save_predictions_to_csv(ids, y_proba, y_label, output_file)

# python check_model.py --model 'XGB' --fingerprint 'ECFP_2' --model_dataset 'cdk2' --validate_dataset 'cdk2' --model_label "y" --validate_label 'y' --model_filename 'raw' --validate_filename 'raw'
