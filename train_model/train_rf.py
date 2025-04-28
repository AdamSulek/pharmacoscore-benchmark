import argparse
import logging
import os
import random
from itertools import product

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)

PARAM_GRID = {
    "n_estimators": [10, 25, 50, 100, 200],
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
}

PARAM_COMBINATIONS = list(product(*PARAM_GRID.values()))
PARAM_KEYS = list(PARAM_GRID.keys())

def load_dataset(dataset, fingerprint, label):
    df = pd.read_parquet(f'../data/{dataset}/raw.parquet')

    data_splits = {
        split: df[df["split"] == split]
        for split in ["train", "val", "test"]
    }

    return {
        split: (
            np.stack(data_splits[split][fingerprint].values),
            data_splits[split][label].values
        )
        for split in data_splits
    }

def train_and_evaluate(dataset, label, dataset_splits, fingerprint_col="ECFP_2"):
    best_model, best_params, best_roc_auc = None, None, -1

    X_train, y_train = dataset_splits["train"]
    X_val, y_val = dataset_splits["val"]

    checkpoint_dir = f"best_models/{dataset}/{label}/random_forest"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for param_values in PARAM_COMBINATIONS:
        params = dict(zip(PARAM_KEYS, param_values))

        model = RandomForestClassifier(
            **params, class_weight="balanced", n_jobs=8, random_state=42
        )

        model.fit(X_train, y_train)
        y_proba_val = model.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, y_proba_val)

        if roc_auc > best_roc_auc:
            best_roc_auc, best_model, best_params = roc_auc, model, params
            model_path = os.path.join(checkpoint_dir, "best_model.joblib")
            dump(best_model, model_path)
            logging.info(f"New best model saved! ROC-AUC: {roc_auc:.4f}, Params: {params}")

    X_test, y_test = dataset_splits["test"]
    y_proba_test = best_model.predict_proba(X_test)[:, 1]
    test_roc_auc = roc_auc_score(y_test, y_proba_test)

    logging.info("Final Model Evaluation:")
    logging.info(f"Test ROC-AUC: {test_roc_auc:.4f}")
    logging.info(f"Best Params: {best_params} | Validation ROC-AUC: {best_roc_auc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Random Forest model")
    parser.add_argument("--dataset", choices=["ampc", "cdk2"], default="cdk2", help="Dataset choice")
    parser.add_argument("--label", choices=["y", "class", "activity"], default="class", help="Target column")

    args = parser.parse_args()
    dataset, label = args.dataset, args.label

    seed_everything(123)

    dataset_splits = load_dataset(dataset, "ECFP_2", label)
    train_and_evaluate(dataset, label, dataset_splits)

    logging.info("Done!")

# nohup python train_rf.py --dataset 'ampc' --label 'y' > train_rf.log 2>&1 &
