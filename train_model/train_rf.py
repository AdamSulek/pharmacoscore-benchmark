import argparse
import logging
import os
import random

import numpy as np
import pandas as pd
import time

from sklearn.metrics import confusion_matrix
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)

def load_scaffold_data(dataset, fingerprint, label):
    df = pd.read_parquet(f'../data/{dataset}/raw.parquet')

    df_train = df[df['split'] == 'train']
    df_test = df[df['split'] == 'test']
    df_val = df[df['split'] == 'val']

    X_train = np.stack(df_train[fingerprint].values)
    y_train = df_train[label].values
    X_val = np.stack(df_val[fingerprint].values)
    y_val = df_val[label].values
    X_test = np.stack(df_test[fingerprint].values)
    y_test = df_test[label].values

    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluate_model(model, X, y, dataset_name):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    roc_auc = roc_auc_score(y, y_proba)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()

    logging.info(f"Metrics for {dataset_name}:")
    logging.info(f"  ROC AUC: {roc_auc:.4f}")
    logging.info(f"  Accuracy: {accuracy:.4f}")
    logging.info(f"  Precision: {precision:.4f}")
    logging.info(f"  Recall: {recall:.4f}")
    logging.info(f"  F1 Score: {f1:.4f}")
    logging.info(f"    TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    return {
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "conf_matrix": conf_matrix
    }

def tune_hyperparameters(X_train, y_train, X_val, y_val, param_grid):
    best_model = None
    best_params = None
    best_roc_auc = -1

    logging.info("Starting manual hyperparameter tuning...")

    for n_estimators in param_grid['n_estimators']:
        for max_depth in param_grid['max_depth']:
            for min_samples_split in param_grid['min_samples_split']:
                for min_samples_leaf in param_grid['min_samples_leaf']:
                    for bootstrap in param_grid['bootstrap']:
                        model = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            bootstrap=bootstrap,
                            class_weight='balanced',
                            n_jobs=8,
                            random_state=42
                        )

                        model.fit(X_train, y_train)

                        y_proba_val = model.predict_proba(X_val)[:, 1]
                        roc_auc = roc_auc_score(y_val, y_proba_val)

                        logging.info(
                            f"Params: n_estimators={n_estimators}, max_depth={max_depth}, "
                            f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, "
                            f"bootstrap={bootstrap} | Validation ROC AUC: {roc_auc:.4f}"
                        )

                        if roc_auc > best_roc_auc:
                            best_roc_auc = roc_auc
                            best_model = model
                            best_params = {
                                'n_estimators': n_estimators,
                                'max_depth': max_depth,
                                'min_samples_split': min_samples_split,
                                'min_samples_leaf': min_samples_leaf,
                                'bootstrap': bootstrap
                            }

    logging.info(f"Best Validation ROC AUC: {best_roc_auc:.4f}")
    logging.info(f"Best Hyperparameters: {best_params}")

    return best_model, best_params, best_roc_auc

def train_model(dataset, label):
    seed_everything(123)

    param_grid = {
        'n_estimators': [10, 25, 50, 100, 200],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    fingerprint = "ECFP_2"
    X_train, y_train, X_val, y_val, X_test, y_test = load_scaffold_data(dataset, fingerprint, label)
    logging.info(f"Loaded Dataset: {dataset} using {label} label")

    start_time = time.time()

    best_model, best_params, best_roc_auc = tune_hyperparameters(X_train, y_train, X_val, y_val, param_grid)

    training_time = time.time() - start_time

    checkpoint_dir = f'best_models/{dataset}/{label}/random_forest'
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_model_path = os.path.join(checkpoint_dir, f'best_model.joblib')
    dump(best_model, best_model_path)

    y_proba = best_model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    logging.info(f"ROC AUC Score: {roc_auc:.4f}")

    logging.info("Evaluating on Test Set...")
    evaluate_model(best_model, X_test, y_test, "Test Set")

    logging.info("Scaffold Training completed.")
    logging.info(f"Best model saved with params: {best_params} in time: {training_time / 60:.2f} minutes")

    logging.info("Training completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Random Forest model with different handling methods for class imbalance.")
    parser.add_argument("--dataset", choices=["cdk2", "cdktest"], default = "cdk2", required=False, help="Dataset choice.")
    parser.add_argument("--label", choices=['class', 'activity'], default='class', required=False, help="Y label column")
    args = parser.parse_args()

    dataset, label = args.dataset, args.label
    train_model(dataset, label)

# nohup python train_random_forest.py --dataset 'ER' --split 'scaffold_split' > logs/train_rf_ER_scaffold_split.log 2>&1 &
# python train_random_forest.py --dataset 'cdk2' --label 'class'
