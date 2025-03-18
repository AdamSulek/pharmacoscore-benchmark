import argparse
import logging
import random
from time import time

import numpy as np
import pandas as pd
from skfp.fingerprints import ECFPFingerprint

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)


def modify_parquet(dataset):
    logging.info(f"Start ECFP fingerprint generation")

    seed_everything(123)

    file_path = f'../data/{dataset}/raw.parquet'
    df = pd.read_parquet(file_path)
    logging.info(f"Read raw.parquet file from {dataset} with {len(df)} rows and {df.shape[1]} columns")

    fp = ECFPFingerprint(count=False, radius=2, verbose=3)

    start_time = time()

    # Przechowywanie wyników
    fingerprints = []
    valid_indices = []
    invalid_count = 0

    for idx, smiles in enumerate(df['smiles']):
        try:
            fingerprint = fp.transform([smiles])[0]  # Pobieramy pierwszy element, bo transform zwraca listę
            valid_indices.append(idx)  # Zachowujemy pozycje (indeksy)
            fingerprints.append(fingerprint)
        except Exception as e:
            logging.warning(f"Could not process SMILES at index {idx}: {smiles} | Error: {e}")
            invalid_count += 1

    logging.info(f"Elapsed Time: {time() - start_time:.2f} seconds")
    logging.info(f"Number of failed fingerprints: {invalid_count}")

    # 📌 **Zachowanie oryginalnych kolumn i dodanie fingerprintów**
    df_clean = df.iloc[valid_indices].copy()  # Używamy `iloc` zamiast `loc`!
    df_clean["ECFP_2"] = fingerprints  # Dodajemy fingerprinty

    # 📌 **Zapisanie jako Parquet**
    df_clean.to_parquet(file_path, engine='pyarrow')
    logging.info(f"Saved cleaned dataset as Parquet with {len(df_clean)} molecules and {df_clean.shape[1]} columns")

    # 📌 **Zapisanie jako CSV**
    csv_path = file_path.replace(".parquet", ".csv")
    df_clean.to_csv(csv_path, index=False)
    logging.info(f"Saved cleaned dataset as CSV: {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Modify parquet file with different fingerprint smiles representation.")
    parser.add_argument("--dataset", choices=['ampc', 'cdk2', 'cdktest', 'decoy'], default="cdk2", required=False,
                        help="Dataset choice.")
    args = parser.parse_args()
    dataset = args.dataset
    modify_parquet(dataset)

#  python fingerprint_gen.py --dataset 'ampc'
