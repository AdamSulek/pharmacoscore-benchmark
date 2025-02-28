import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score 
import logging
import argparse
import pandas as pd
import random
from itertools import product
import joblib
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)

PARAM_GRID = {
    'learning_rate': [0.05, 0.1],        
    'max_depth': [6, 8],                 
    'n_estimators': [100, 300],          
    'colsample_bytree': [0.7, 0.9],      
    'gamma': [0, 0.1],                   
    'reg_alpha': [0, 0.1],               
    'reg_lambda': [0.5, 1]               
}

PARAM_COMBINATIONS = list(product(*PARAM_GRID.values()))
PARAM_KEYS = list(PARAM_GRID.keys())


def train_and_evaluate(dataset, label, df, test_df, tree_method, fingerprint_col = "ECFP_2"):
    best_model, best_roc_auc, best_params = None, 0, None

    train_df, val_df = df[df['split'] == 'train'], df[df['split'] == 'val']
    checkpoint_dir = f"best_models/{dataset}/{label}/xgboost"
    os.makedirs(checkpoint_dir, exist_ok=True)

    X_train, y_train = np.stack(train_df[fingerprint_col].values), train_df[label].values
    X_val, y_val = np.stack(val_df[fingerprint_col].values), val_df[label].values

    for param_values in PARAM_COMBINATIONS:
        params = dict(zip(PARAM_KEYS, param_values))

        model = xgb.XGBClassifier(
            tree_method=tree_method,
            eval_metric='logloss',
            **params
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, y_val_proba)

        if roc_auc > best_roc_auc:
            best_roc_auc, best_model, best_params = roc_auc, model, params
            model_path = f"{checkpoint_dir}/best_model.joblib"
            joblib.dump(best_model, model_path)
            logging.info(f"New best model saved! ROC-AUC: {roc_auc:.4f}, Params: {params}")

    X_test, y_test = np.stack(test_df[fingerprint_col].values), test_df[label].values
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    test_roc_auc = roc_auc_score(y_test, y_test_proba)

    logging.info("Final Model Evaluation:")
    logging.info(f"Test ROC-AUC: {test_roc_auc:.4f}")
    logging.info(f"Best Params: {best_params} | Validation ROC-AUC: {best_roc_auc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train XGBoost model")
    parser.add_argument("--gpu", type=int, default=0, help="Use GPU (1) or CPU (0)")
    parser.add_argument("--dataset", choices=["cdk2"], default="cdk2", help="Dataset choice")
    parser.add_argument("--label", choices=['class', 'activity'], default='class', help="Target column")

    args = parser.parse_args()
    dataset, label, gpu = args.dataset, args.label, args.gpu

    seed_everything(123)
    
    df = pd.read_parquet(f'../data/{dataset}/raw.parquet')
    test_df = df[df['split'] == 'test']

    train_and_evaluate(dataset, label, df, test_df,
                       tree_method="gpu_hist" if gpu == 1 else "hist")
    logging.info("Done!")

# nohup python train_xgb.py --dataset 'cdk2' --label 'class' > train_xgb.log 2>&1 &



