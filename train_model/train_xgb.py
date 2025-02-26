import time
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
import time
from sklearn.preprocessing import label_binarize

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)

param_grid = {
    'learning_rate': [0.05, 0.1],        
    'max_depth': [6, 8],                 
    'n_estimators': [100, 300],          
    'colsample_bytree': [0.7, 0.9],      
    'gamma': [0, 0.1],                   
    'reg_alpha': [0, 0.1],               
    'reg_lambda': [0.5, 1]               
}

param_combinations = list(product(*param_grid.values()))
param_keys = list(param_grid.keys())

best_model = None
best_roc_auc = 0
best_params = None


def train_and_evaluate(dataset, label, df, test_df, device, tree_method):
    """
    Train and evaluate an XGBoost model for binary classification based on ROC-AUC.

    Args:
        df (DataFrame): Training and validation dataset.
        test_df (DataFrame): Test dataset.
        device (str): Device to use for training ('gpu' or 'cpu').
        tree_method (str): XGBoost tree method.
        fingerprint (str): Column name containing molecular fingerprints.
        activity_column (str): Column name for the binary target variable.

    Returns:
        None
    """

    global best_model, best_roc_auc, best_params

    # Split the dataset
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']

    checkpoint_dir = f"best_models/{dataset}/{label}/xgboost"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for param_values in param_combinations:
        params = dict(zip(param_keys, param_values))

        start_time = time.time()

        model = xgb.XGBClassifier(
            tree_method=tree_method,
            device=device,
            use_label_encoder=False,
            eval_metric='logloss',  # For binary classification
            **params
        )

        # Extract features and labels
        X_train = np.stack(train_df[f'ECFP_2'].values)
        y_train = train_df[label].values
        X_val = np.stack(val_df[f'ECFP_2'].values)
        y_val = val_df[label].values

        # Train the model
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Get predicted probabilities for the positive class (class 1)
        y_val_proba = model.predict_proba(X_val)[:, 1]

        # Compute ROC-AUC score
        roc_auc = roc_auc_score(y_val, y_val_proba)

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model = model
            best_params = params

            # Save the best model
            model_path = f"{checkpoint_dir}/best_model.joblib"
            joblib.dump(best_model, model_path)
            logging.info(f"New best model saved successfully at {model_path} with ROC-AUC: {roc_auc:.4f}")

        logging.info(f"Params: {params} | Validation ROC-AUC: {roc_auc:.4f}")

    # Load the best model
    best_model_path = f"{checkpoint_dir}/best_model.joblib"
    best_model = joblib.load(best_model_path)

    # Evaluate on the test set
    X_test = np.stack(test_df[f'ECFP_2'].values)
    y_test = test_df[label].values

    # Get predicted probabilities for the positive class (class 1)
    y_test_proba = best_model.predict_proba(X_test)[:, 1]

    # Compute ROC-AUC on test set
    test_roc_auc = roc_auc_score(y_test, y_test_proba)

    logging.info(f"Final Model Evaluation on Test Set:")
    logging.info(f"Test ROC-AUC: {test_roc_auc:.4f}")
    logging.info(f"Best Model Params: {best_params} | Best Validation ROC-AUC: {best_roc_auc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="choose CPU or GPU")
    parser.add_argument("--gpu", 
                        type=int, 
                        default=0,
                        help="choose GPU or CPU")
    parser.add_argument("--dataset", choices=["cdk2"], default="cdk2", required=False, help="Dataset choice.")
    parser.add_argument("--label", choices=['class', 'activity'], default='class', required=False,
                        help="Y label column")

    args = parser.parse_args()
    dataset, label, gpu = args.dataset, args.label, args.gpu

    seed_everything(123)
    
    df = pd.read_parquet(f'../data/{dataset}/raw.parquet')
    test_df = df[df['split'] == 'test']
    #param_grid['scale_pos_weight'] = [scale_pos_weight] 
    
    if args.gpu == 0:
        train_and_evaluate(dataset, label, df=df, test_df=test_df,
                           device="cpu", tree_method="hist")
    elif args.gpu == 1:
        train_and_evaluate(dataset, label, df=df, test_df=test_df,
                           device="cuda", tree_method="gpu_hist")

    logging.info("Done!")

 
# python train_xgb.py > train_xgb.log 2>&1 &
# python train_xgb.py --dataset 'cdk2' --label 'activity'



