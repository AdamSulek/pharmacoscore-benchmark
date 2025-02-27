import argparse
import logging
import os
import pickle
import sys

import joblib
import numpy as np
import pandas as pd
import shap
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'train_model')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gcn_model import GCN
from mlp_model import MLP

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_gcn_model(dataset):
    model = GCN(input_dim=42, model_dim=512, dropout_rate=0.0, n_layers=3,
                num_fc_layers=2, fc_hidden_dim=128, concat_conv_layers=1, use_hooks=True).to(device)
    model.load_state_dict(torch.load(f"../train_model/best_models/{dataset}/class/gcn/best_model.pth", map_location=torch.device('cpu')))
    return model

def load_mlp_model(dataset):
    checkpoint = torch.load(f"../train_model/best_models/{dataset}/class/mlp/best_model.pth")

    hidden_dim = checkpoint['hidden_dim']
    num_hidden_layers = checkpoint['num_hidden_layers']
    dropout_rate = checkpoint['dropout_rate']

    model = MLP(hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers, dropout_rate=dropout_rate)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def load_rf_model(dataset):
    return joblib.load(f"../train_model/best_models/{dataset}/class/random_forest/best_model.joblib")

def load_xgb_model(dataset):
    return joblib.load(f"../train_model/best_models/{dataset}/class/xgboost/best_model.joblib")

def get_fingerprint_with_bit_info(mol, radius=2, n_bits=2048):
    bit_info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, bitInfo=bit_info)
    return np.array(fp), bit_info


def get_unique_atoms_from_top_features(top_features, bit_info):
    unique_atoms = set()

    for feature in top_features:
        if feature in bit_info:  # Ensure feature exists in bit_info
            atoms_in_feature = bit_info[feature]
            for atom_tuple in atoms_in_feature:  # Some features map to multiple atoms
                unique_atoms.update(atom_tuple)  # Add atoms to the set
    return list(unique_atoms)


def score_match(mol, top_nodes):
    feature_counts = {"HBA": 0, "HBD": 0, "aromatic_1": 0, "aromatic_2": 0, "hydrophobic": 0}

    for node in top_nodes:
        if node == mol['HBA_label']:
            feature_counts["HBA"] = 1
        if node == mol['HBD_label']:
            feature_counts["HBD"] = 1
        if node in mol['aromatic_1_label']:
            feature_counts["aromatic_1"] = 1
        if node in mol['aromatic_2_label']:
            feature_counts["aromatic_2"] = 1
        if node in mol['hydrophobic_label_modified']:
            feature_counts["hydrophobic"] = 1
    return feature_counts

def compute_saliency_map_mlp(model, data):
    model.eval()
    data = data.to(next(model.parameters()).device)

    data.requires_grad = True

    output = model(data)
    loss = output.sum()
    loss.backward()

    gradients = data.grad
    feature_importance = gradients.detach().cpu().numpy()
    return feature_importance

def identify_influential_nodes_gcn_gradcam(model, data, top_k=5):
    model.eval()
    data = data.to(device)
    output = model(data)
    model.zero_grad()
    output.backward()
    gradients = model.final_conv_grads
    node_activations = model.final_conv_acts
    node_importance = (gradients * node_activations).sum(dim=1)
    _, top_indices = torch.topk(node_importance, top_k)
    top_nodes = top_indices.cpu().numpy()
    return top_nodes, node_importance.cpu().detach().numpy()

def identify_influential_nodes_gcn_vg(model, data, top_k=5):
    model.eval()
    data = data.to(next(model.parameters()).device)
    data.x.requires_grad_(True)

    output = model(data)

    model.zero_grad()
    output.sum().backward()

    saliency = data.x.grad.abs().sum(dim=1)

    _, top_indices = torch.topk(saliency, top_k)
    top_nodes = top_indices.cpu().numpy()

    return top_nodes, saliency.cpu().detach().numpy()

def analyze_mol(matching_row, important_atoms, total_feature_counts, found_cases):

    mol = matching_row.iloc[0]
    feature_counts = score_match(mol, important_atoms)

    for key in total_feature_counts:
        total_feature_counts[key] += feature_counts[key]

    found_cases += 1
    return total_feature_counts, found_cases

def analyze_rf(rf_model, df, pharm_labels):
    total_feature_counts = {"HBA": 0, "HBD": 0, "aromatic_1": 0, "aromatic_2": 0, "hydrophobic": 0}

    explainer = shap.Explainer(rf_model, feature_perturbation="tree_path_dependent")
    found_cases = 0
    unique_atoms, all_atoms = 0, 0

    for _, row in df.iterrows():
        chembl_id = row['chembl_id']
        mol = Chem.MolFromSmiles(row['smiles'])

        fingerprint, bit_info = get_fingerprint_with_bit_info(mol)
        fingerprint = fingerprint.reshape(1, -1)

        shap_values = explainer(fingerprint)
        shap_values_class_1 = shap_values.values[0, :, 1]
        top_bits = np.argsort(np.abs(shap_values_class_1))[::-1][:5]

        important_atoms = [bit_info[bit][0][0] for bit in top_bits if bit in bit_info]

        matching_row = pharm_labels[pharm_labels["chembl_id"] == chembl_id].iloc[:1]

        if not matching_row.empty:
            important_atoms_unique = get_unique_atoms_from_top_features(top_bits, bit_info)
            all_atoms += mol.GetNumAtoms()
            unique_atoms += len(important_atoms_unique)
            total_feature_counts, found_cases = analyze_mol(matching_row, important_atoms, total_feature_counts, found_cases)

    feature_means = {feature: count / found_cases for feature, count in total_feature_counts.items()}

    return feature_means, unique_atoms / all_atoms

def analyze_mlp(mlp_model, df, pharm_labels):
    total_feature_counts = {"HBA": 0, "HBD": 0, "aromatic_1": 0, "aromatic_2": 0, "hydrophobic": 0}

    X_test = np.stack(df['ECFP_2'].values).astype(np.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    background = X_test_tensor
    explainer = shap.DeepExplainer(mlp_model, background)

    found_cases = 0
    unique_atoms, all_atoms = 0, 0

    for _, row in df.iterrows():
        chembl_id = row['chembl_id']
        mol = Chem.MolFromSmiles(row['smiles'])

        fingerprint, bit_info = get_fingerprint_with_bit_info(mol)
        fingerprint = fingerprint.reshape(1, -1)

        fingerprint_tensor = torch.tensor(fingerprint, dtype=torch.float32).to(device)

        shap_values = explainer.shap_values(fingerprint_tensor, check_additivity=False)
        shap_values_class_1 = shap_values[0, :]
        top_bits = np.argsort(np.abs(shap_values_class_1))[::-1][:5]

        important_atoms = [bit_info[bit][0][0] for bit in top_bits if bit in bit_info]

        matching_row = pharm_labels[pharm_labels["chembl_id"] == chembl_id].iloc[:1]

        if not matching_row.empty:
            important_atoms_unique = get_unique_atoms_from_top_features(top_bits, bit_info)
            all_atoms += mol.GetNumAtoms()
            unique_atoms += len(important_atoms_unique)
            total_feature_counts, found_cases = analyze_mol(matching_row, important_atoms, total_feature_counts, found_cases)

    feature_means = {feature: count / found_cases for feature, count in total_feature_counts.items()}

    return feature_means, unique_atoms / all_atoms

def analyze_mlp_vg(mlp_model, df, pharm_labels):
    total_feature_counts = {"HBA": 0, "HBD": 0, "aromatic_1": 0, "aromatic_2": 0, "hydrophobic": 0}

    found_cases = 0
    unique_atoms, all_atoms = 0, 0

    for _, row in df.iterrows():
        chembl_id = row['chembl_id']
        mol = Chem.MolFromSmiles(row['smiles'])

        fingerprint, bit_info = get_fingerprint_with_bit_info(mol)
        fingerprint = fingerprint.reshape(1, -1)

        fingerprint_tensor = torch.tensor(fingerprint, dtype=torch.float32).to(device)
        saliency_values = compute_saliency_map_mlp(mlp_model, fingerprint_tensor)
        molecule_siliency_values = saliency_values[0]

        indices_with_ones = np.where(fingerprint[0] == 1)[0]
        saliency_values_for_ones = molecule_siliency_values[indices_with_ones]

        sorted_indices = np.argsort(np.abs(saliency_values_for_ones))[::-1][:5]
        top_bits = indices_with_ones[sorted_indices]

        important_atoms = [bit_info[bit][0][0] for bit in top_bits if bit in bit_info]

        matching_row = pharm_labels[pharm_labels["chembl_id"] == chembl_id].iloc[:1]

        if not matching_row.empty:
            important_atoms_unique = get_unique_atoms_from_top_features(top_bits, bit_info)
            all_atoms += mol.GetNumAtoms()
            unique_atoms += len(important_atoms_unique)
            total_feature_counts, found_cases = analyze_mol(matching_row, important_atoms, total_feature_counts, found_cases)

    feature_means = {feature: count / found_cases for feature, count in total_feature_counts.items()}

    return feature_means, unique_atoms / all_atoms

def analyze_xgb(xgb_model, df, pharm_labels):
    total_feature_counts = {"HBA": 0, "HBD": 0, "aromatic_1": 0, "aromatic_2": 0, "hydrophobic": 0}

    explainer = shap.Explainer(xgb_model)
    found_cases = 0
    unique_atoms, all_atoms = 0, 0

    for _, row in df.iterrows():
        chembl_id = row['chembl_id']
        mol = Chem.MolFromSmiles(row['smiles'])

        fingerprint, bit_info = get_fingerprint_with_bit_info(mol)
        fingerprint = fingerprint.reshape(1, -1)

        shap_values = explainer(fingerprint)
        shap_values_class_1 = shap_values.values[0, :]
        top_bits = np.argsort(np.abs(shap_values_class_1))[::-1][:5]

        important_atoms = [bit_info[bit][0][0] for bit in top_bits if bit in bit_info]

        matching_row = pharm_labels[pharm_labels["chembl_id"] == chembl_id].iloc[:1]

        if not matching_row.empty:
            important_atoms_unique = get_unique_atoms_from_top_features(top_bits, bit_info)
            all_atoms += mol.GetNumAtoms()
            unique_atoms += len(important_atoms_unique)
            total_feature_counts, found_cases = analyze_mol(matching_row, important_atoms, total_feature_counts, found_cases)

    feature_means = {feature: count / found_cases for feature, count in total_feature_counts.items()}

    return feature_means, unique_atoms / all_atoms


def analyze_gcn(model, df, pharm_labels):
    total_feature_counts = {"HBA": 0, "HBD": 0, "aromatic_1": 0, "aromatic_2": 0, "hydrophobic": 0}
    found_cases = 0

    for test_sample in df:
        label = test_sample.label
        if label == 1:
            chembl_id = test_sample.molecule_id
            test_sample = test_sample.to(device)

            top_nodes, _ = identify_influential_nodes_gcn_gradcam(model, test_sample)

            matching_row = pharm_labels[pharm_labels["chembl_id"] == chembl_id].iloc[:1]

            if not matching_row.empty:
                # important_atoms_unique = get_unique_atoms_from_top_features(top_bits, bit_info)
                # all_atoms += mol.GetNumAtoms()
                # unique_atoms += len(important_atoms_unique)
                total_feature_counts, found_cases = analyze_mol(matching_row, top_nodes, total_feature_counts,
                                                                found_cases)
    feature_means = {feature: count / found_cases for feature, count in total_feature_counts.items()}

    return feature_means, 0

def analyze_gcn_vg(model, df, pharm_labels):
    total_feature_counts = {"HBA": 0, "HBD": 0, "aromatic_1": 0, "aromatic_2": 0, "hydrophobic": 0}
    found_cases = 0

    for test_sample in df:
        label = test_sample.label
        if label == 1:
            chembl_id = test_sample.molecule_id
            test_sample = test_sample.to(device)

            top_nodes, _ = identify_influential_nodes_gcn_vg(model, test_sample)

            matching_row = pharm_labels[pharm_labels["chembl_id"] == chembl_id].iloc[:1]

            if not matching_row.empty:
                # important_atoms_unique = get_unique_atoms_from_top_features(top_bits, bit_info)
                # all_atoms += mol.GetNumAtoms()
                # unique_atoms += len(important_atoms_unique)
                total_feature_counts, found_cases = analyze_mol(matching_row, top_nodes, total_feature_counts,
                                                                found_cases)
    feature_means = {feature: count / found_cases for feature, count in total_feature_counts.items()}

    return feature_means, 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check model interpretability in terms of pharmacophore alignment and sparsity.")
    parser.add_argument("--model" ,type=str, choices=['RF', 'MLP', 'GCN', 'GCN_VG', 'XGB', 'MLP_VG'])
    parser.add_argument("--dataset", choices=["cdk2"], default="cdk2", required=False, help="Dataset choice.")

    args = parser.parse_args()
    c_model, dataset = args.model, args.dataset

    pharm_labels = pd.read_parquet(f"../data/{dataset}/pharmacophore_labels.parquet")

    os.makedirs(f"results/{dataset}", exist_ok=True)

    if c_model.startswith('GCN'):
        with open(f'../data/{dataset}/graph_data.p', 'rb') as f:
            df = pickle.load(f)
        test_data = [data for data in df if data.split == 'test']
    else:
        df = pd.read_parquet(f"../data/{dataset}/raw.parquet")
        test_data = df[(df['split'] == 'test') & (df['class'] == 1)].reset_index(drop=True)

    if args.model == 'MLP':
        mlp_model = load_mlp_model(dataset)
        feature_means, sparsity = analyze_mlp(mlp_model, test_data, pharm_labels)

    elif args.model == 'MLP_VG':
        mlp_model = load_mlp_model(dataset)
        feature_means, sparsity = analyze_mlp_vg(mlp_model, test_data, pharm_labels)

    elif args.model == 'RF':
        rf_model = load_rf_model(dataset)
        feature_means, sparsity = analyze_rf(rf_model, test_data, pharm_labels)

    elif args.model == 'XGB':
        xgb_model = load_xgb_model(dataset)
        feature_means, sparsity = analyze_xgb(xgb_model, test_data, pharm_labels)

    elif args.model == 'GCN':
        gcn_model = load_gcn_model(dataset)
        feature_means, sparsity = analyze_gcn(gcn_model, test_data, pharm_labels)

    elif args.model == 'GCN_VG':
        gcn_model = load_gcn_model(dataset)
        feature_means, sparsity = analyze_gcn_vg(gcn_model, test_data, pharm_labels)

    logging.info(f"Checking {args.model} \n Feature Mean Scores: {feature_means}")
    logging.info(f"Sparsity: {sparsity}")

# python check_pharmacophore_allignment_and_sparsity.py --model 'RF' --dataset 'cdk2'
