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
    checkpoint = torch.load(f"../train_model/best_models/{dataset}/class/gcn/best_model.pth", weights_only=False)
    input_dim = checkpoint['input_dim']
    model_dim = checkpoint['model_dim']
    dropout_rate = checkpoint['dropout_rate']
    n_layers = checkpoint['n_layers']
    num_fc_layers = checkpoint['num_fc_layers']
    fc_hidden_dim = checkpoint['fc_hidden_dim']

    model = GCN(input_dim=input_dim, model_dim=model_dim, dropout_rate=dropout_rate, n_layers=n_layers,
                num_fc_layers=num_fc_layers, fc_hidden_dim=fc_hidden_dim,
                concat_conv_layers=1, use_hooks=True).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def load_mlp_model(dataset, label, filename):
    checkpoint = torch.load(f"../train_model/best_models/{dataset}/{label}/mlp/best_model_{filename}.pth")
    hidden_dim = checkpoint['hidden_dim']
    num_hidden_layers = checkpoint['num_hidden_layers']
    dropout_rate = checkpoint['dropout_rate']

    model = MLP(hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers, dropout_rate=dropout_rate)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def load_rf_model(dataset, label, filename):
    return joblib.load(f"../train_model/best_models/{dataset}/{label}/random_forest/best_model_{filename}.joblib")

def load_xgb_model(dataset, label, filename):
    return joblib.load(f"../train_model/best_models/{dataset}/{label}/xgboost/best_model_{filename}.joblib")


def get_fingerprint_with_bit_info(mol, radius=2, n_bits=2048):
    bit_info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, bitInfo=bit_info)
    return np.array(fp), bit_info

def get_unique_atoms_from_top_features(top_features, bit_info):
    unique_atoms = set()
    for feature in top_features:
        if feature in bit_info:
            atoms_in_feature = bit_info[feature]
            for atom_tuple in atoms_in_feature:
                unique_atoms.update(atom_tuple)
    return list(unique_atoms)

def score_match(mol, top_nodes):
    feature_counts = {"HBA": 0, "HBD": 0, "aromatic_1": 0, "aromatic_2": 0, "hydrophobic": 0}

    hba_label = mol.get('HBA_label')
    hbd_label = mol.get('HBD_label')
    aromatic_1_label = mol.get('aromatic_1_label')
    aromatic_2_label = mol.get('aromatic_2_label')
    hydrophobic_label_modified = mol.get('hydrophobic_label')

    for node in top_nodes:
        if hba_label is not None and node == hba_label:
            feature_counts["HBA"] = 1
        if hbd_label is not None and node == hbd_label:
            feature_counts["HBD"] = 1
        if aromatic_1_label is not None and node in aromatic_1_label:
            feature_counts["aromatic_1"] = 1
        if aromatic_2_label is not None and node in aromatic_2_label:
            feature_counts["aromatic_2"] = 1
        if hydrophobic_label_modified is not None and node in hydrophobic_label_modified:
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

def analyze_rf(rf_model, df, pharm_labels, output_list):
    total_feature_counts = {"HBA": 0, "HBD": 0, "aromatic_1": 0, "aromatic_2": 0, "hydrophobic": 0}

    explainer = shap.Explainer(rf_model, feature_perturbation="tree_path_dependent")
    found_cases = 0
    unique_atoms, all_atoms = 0, 0

    for _, row in df.iterrows():
        chembl_id = row["ID"]
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)

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
            output_list.append({'chembl_id': chembl_id, 'top_nodes': list(important_atoms_unique), 'smiles': smiles})

    feature_means = {feature: count / found_cases for feature, count in total_feature_counts.items()}

    return feature_means, unique_atoms / all_atoms

def analyze_mlp(mlp_model, df, pharm_labels, output_list):
    total_feature_counts = {"HBA": 0, "HBD": 0, "aromatic_1": 0, "aromatic_2": 0, "hydrophobic": 0}

    X_test = np.stack(df['ECFP_2'].values).astype(np.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    background = X_test_tensor
    explainer = shap.DeepExplainer(mlp_model, background)

    found_cases = 0
    unique_atoms, all_atoms = 0, 0

    for _, row in df.iterrows():
        chembl_id = row["ID"]
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)

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
            output_list.append({'chembl_id': chembl_id, 'top_nodes': list(important_atoms_unique), 'smiles': smiles})

    feature_means = {feature: count / found_cases for feature, count in total_feature_counts.items()}

    return feature_means, unique_atoms / all_atoms

def analyze_mlp_vg(mlp_model, df, pharm_labels, output_list):
    total_feature_counts = {"HBA": 0, "HBD": 0, "aromatic_1": 0, "aromatic_2": 0, "hydrophobic": 0}

    found_cases = 0
    unique_atoms, all_atoms = 0, 0

    for _, row in df.iterrows():
        chembl_id = row["ID"]
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)

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
            output_list.append({'chembl_id': chembl_id, 'top_nodes': list(important_atoms_unique), 'smiles': smiles})

    feature_means = {feature: count / found_cases for feature, count in total_feature_counts.items()}

    return feature_means, unique_atoms / all_atoms

def analyze_xgb(xgb_model, df, pharm_labels, output_list):
    total_feature_counts = {"HBA": 0, "HBD": 0, "aromatic_1": 0, "aromatic_2": 0, "hydrophobic": 0}

    explainer = shap.Explainer(xgb_model)
    found_cases = 0
    unique_atoms, all_atoms = 0, 0

    for _, row in df.iterrows():
        chembl_id = row["ID"]
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)

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
            output_list.append({'chembl_id': chembl_id, 'top_nodes': list(important_atoms_unique), 'smiles': smiles})

    feature_means = {feature: count / found_cases for feature, count in total_feature_counts.items()}

    return feature_means, unique_atoms / all_atoms


def analyze_gcn(model, df, pharm_labels, output_list):
    total_feature_counts = {"HBA": 0, "HBD": 0, "aromatic_1": 0, "aromatic_2": 0, "hydrophobic": 0}
    found_cases = 0

    for test_sample in df:
        label = test_sample.label
        if label == 1:
            chembl_id = test_sample.molecule_id
            smiles = test_sample.smiles
            test_sample = test_sample.to(device)

            top_nodes, _ = identify_influential_nodes_gcn_gradcam(model, test_sample)

            matching_row = pharm_labels[pharm_labels["chembl_id"] == chembl_id].iloc[:1]

            if not matching_row.empty:
                # important_atoms_unique = get_unique_atoms_from_top_features(top_bits, bit_info)
                # all_atoms += mol.GetNumAtoms()
                # unique_atoms += len(important_atoms_unique)
                total_feature_counts, found_cases = analyze_mol(matching_row, top_nodes, total_feature_counts,
                                                                         found_cases)
                output_list.append({'chembl_id': chembl_id, 'top_nodes': str(top_nodes), 'smiles': smiles})
    feature_means = {feature: count / found_cases for feature, count in total_feature_counts.items()}

    return feature_means, 0

def analyze_gcn_vg(model, df, pharm_labels, output_list):
    total_feature_counts = {"HBA": 0, "HBD": 0, "aromatic_1": 0, "aromatic_2": 0, "hydrophobic": 0}
    found_cases = 0

    for test_sample in df:
        label = test_sample.label
        if label == 1:
            chembl_id = test_sample.molecule_id
            smiles = test_sample.smiles
            test_sample = test_sample.to(device)

            top_nodes, _ = identify_influential_nodes_gcn_vg(model, test_sample)

            matching_row = pharm_labels[pharm_labels["chembl_id"] == chembl_id].iloc[:1]

            if not matching_row.empty:
                # important_atoms_unique = get_unique_atoms_from_top_features(top_bits, bit_info)
                # all_atoms += mol.GetNumAtoms()
                # unique_atoms += len(important_atoms_unique)
                total_feature_counts, found_cases = analyze_mol(matching_row, top_nodes, total_feature_counts,
                                                                         found_cases)
                output_list.append({'chembl_id': chembl_id, 'top_nodes': str(top_nodes), 'smiles': smiles})
    feature_means = {feature: count / found_cases for feature, count in total_feature_counts.items()}

    return feature_means, 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check model interpretability in terms of pharmacophore alignment and sparsity.")
    parser.add_argument("--model", default="RF", choices=['RF', 'MLP', 'GCN', 'GCN_VG', 'XGB', 'MLP_VG'], required=False)
    parser.add_argument("--dataset", default="cdk2", required=False, help="Dataset choice.")
    parser.add_argument("--label", choices=["y", "class", "activity"], default="y", required=False, help="Y label column.")
    parser.add_argument("--filename", default="raw", required=False, help="Dataset filename")

    args = parser.parse_args()
    c_model, dataset, label, filename = args.model, args.dataset, args.label, args.filename

    os.makedirs("results", exist_ok=True)
    output_file = f"results/{dataset}_{c_model}_{label}_top_nodes_activity.parquet"
    pharm_labels = pd.read_parquet(f"../data/{dataset}/pharmacophore_labels.parquet")
    output_data = []

    if c_model.startswith('GCN'):
        with open(f'../data/{dataset}/graph_data_class.p', 'rb') as f:
            df = pickle.load(f)
        test_data = [data for data in df if data.split == 'test']
    else:
        df = pd.read_parquet(f"../data/{dataset}/{filename}.parquet")
        test_data = df[(df['split'] == 'test') & (df[label] == 1)].reset_index(drop=True)

    if args.model == 'MLP':
        mlp_model = load_mlp_model(dataset, label, filename)
        feature_means, sparsity = analyze_mlp(mlp_model, test_data, pharm_labels, output_data)

    elif args.model == 'MLP_VG':
        mlp_model = load_mlp_model(dataset, label, filename)
        feature_means, sparsity = analyze_mlp_vg(mlp_model, test_data, pharm_labels, output_data)

    elif args.model == 'RF':
        rf_model = load_rf_model(dataset, label, filename)
        feature_means, sparsity = analyze_rf(rf_model, test_data, pharm_labels, output_data)

    elif args.model == 'XGB':
        xgb_model = load_xgb_model(dataset, label, filename)
        feature_means, sparsity = analyze_xgb(xgb_model, test_data, pharm_labels, output_data)

    elif args.model == 'GCN':
        gcn_model = load_gcn_model(dataset)
        feature_means, sparsity = analyze_gcn(gcn_model, test_data, pharm_labels, output_data)

    elif args.model == 'GCN_VG':
        gcn_model = load_gcn_model(dataset)
        feature_means, sparsity = analyze_gcn_vg(gcn_model, test_data, pharm_labels, output_data)

    logging.info(f"Checking {args.model} \n Feature Mean Scores: {feature_means}")
    logging.info(f"Sparsity: {sparsity}")

    output_df = pd.DataFrame(output_data)
    output_df.to_parquet(output_file, index=False)


# python check_pharmacophore_allignment_and_sparsity.py --model 'RF' --dataset 'cdk2' --label 'y' --filename 'raw'
