import argparse
import logging
import os
import random
import sys
from time import time
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from skfp.fingerprints import MordredFingerprint, ECFPFingerprint, KlekotaRothFingerprint
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)


def get_fingerprint_method(method, count, radius):
    if method == 'ecfp':
        return ECFPFingerprint(count=count, radius=radius, verbose=3)
    elif method == 'klekota':
        return KlekotaRothFingerprint(count=count, verbose=3)
    elif method == 'mordred':
        return MordredFingerprint(verbose=3)
    else:
        raise ValueError(f"Unknown fingerprint method: {method}")


def modify_parquet(fingerprint, df):
    log_path = f"output_logs/{fingerprint}/"
    log_file = log_path + f"/modify_fingerprint_{fingerprint}.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

    seed_everything(123)

    parameters = fingerprint.split('_')
    method = parameters[1]
    count, radius = True, 2
    if method == 'ecfp':
        if len(parameters) == 3:
            count = False
            radius = int(parameters[2])
        if len(parameters) == 4:
            count = True
            radius = int(parameters[3])
    if method == 'klekota':
        if len(parameters) == 2:
            count = False
        if len(parameters) == 3:
            count = True

    fp = get_fingerprint_method(method, count, radius)
    X_fingerprint = fp.transform(df['smiles'].tolist())

    df[f"{fingerprint}"] = list(X_fingerprint)

    return df


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Modify parquet file with different fingerprint smiles representation.")
    # parser.add_argument("--fingerprint", choices=['X', 'X_ecfp_2', 'X_ecfp_count_2', 'X_ecfp_3', 'X_ecfp_count_3', 'X_klekota',
    #                                                'X_klekota_count', 'X_mordred'], required=True, help="Fingerprint method.")
    # args = parser.parse_args()
    # file_path = 'data/assay_and_VS/raw.parquet'
    # df = pd.read_parquet(file_path)

    file_path = '../data/ampc/raw.csv'
    df = pd.read_csv(file_path)

    for fp_str in ['X_ecfp_2']:  # ['X_ecfp_count_2', 'X_ecfp_3', 'X_ecfp_count_3']:#, 'X_mordred']:
        df = modify_parquet(fp_str, df)
        logging.info(f"fingerprint {fp_str} done")

    df.to_parquet('data/assay_and_VS/df_new_split.parquet')  # file_path)

    logging.info("Done!")

# nohup python data_featurizer.py > data_featurizer_even_newer_dataset.log 2>&1 &

