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

from sklearn.metrics import accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'train_model')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gcn_model import GCN
from mlp_model import MLP

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fingerprint_col = 'ECFP_2'

def count_fidelity(fidelity_df):
    y_true = fidelity_df[f"label"].values
    y_pred = (fidelity_df[f"pred_proba"] >= 0.5).astype(int)  # Binarize predictions
    y_pred_mask = (fidelity_df[f"pred_proba_mask"] >= 0.5).astype(int)  # Binarize masked predictions

    accuracy = accuracy_score(y_true, y_pred)
    masked_accuracy = accuracy_score(y_true, y_pred_mask)

    fidelity = accuracy - masked_accuracy

    return fidelity, accuracy, masked_accuracy

def get_top_5_indices(shap_values):
    top_5_indices = np.argsort(shap_values)[-5:]
    return top_5_indices

def generate_fingerprint(smiles, fingerprint_size=2048, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    bit_info = {}
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=fingerprint_size, bitInfo=bit_info)
    return np.array(fingerprint), bit_info

def gen_X_fingerprint(df, radius, fingerprint_size=2048):
    smiles_list = df['smiles'].tolist()
    X_array = np.zeros((df.shape[0], fingerprint_size))
    bit_info_list = []

    for i, smiles_value in enumerate(smiles_list):
        X, bit_info = generate_fingerprint(smiles_value, fingerprint_size=fingerprint_size, radius=radius)
        X_array[i, :len(X)] = X
        bit_info_list.append(bit_info)

    return X_array, bit_info_list

def load_gcn_model(use_hooks):
    checkpoint = torch.load(f"../train_model/best_models/{dataset}/{model_label}/mlp/best_model.pth")
    input_dim = checkpoint['input_dim']
    model_dim = checkpoint['model_dim']
    dropout_rate = checkpoint['dropout_rate']
    n_layers = checkpoint['n_layers']
    num_fc_layers = checkpoint['num_fc_layers']
    fc_hidden_dim = checkpoint['fc_hidden_dim']

    if use_hooks:
        model = GCN(input_dim=input_dim, model_dim=model_dim, dropout_rate=dropout_rate, n_layers=n_layers,
                    num_fc_layers=num_fc_layers, fc_hidden_dim=fc_hidden_dim,
                    concat_conv_layers=1, use_hooks=True).to(device)
    else:
        model = GCN(input_dim=input_dim, model_dim=model_dim, dropout_rate=dropout_rate, n_layers=n_layers,
                    num_fc_layers=num_fc_layers, fc_hidden_dim=fc_hidden_dim, concat_conv_layers=1).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def load_mlp_model(dataset, model_label):
    checkpoint = torch.load(f"../train_model/best_models/{dataset}/{model_label}/mlp/best_model.pth")

    hidden_dim = checkpoint['hidden_dim']
    num_hidden_layers = checkpoint['num_hidden_layers']
    dropout_rate = checkpoint['dropout_rate']

    model = MLP(hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers, dropout_rate=dropout_rate)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def load_rf_model(dataset, model_label):
    return joblib.load(f"../train_model/best_models/{dataset}/{model_label}/random_forest/best_model.joblib")

def load_xgb_model(dataset, model_label):
    return joblib.load(f"../train_model/best_models/{dataset}/{model_label}/xgboost/best_model.joblib")

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

def add_model_predictions_and_atom_importance_to_df(df, model, model_name, label='class'):
    chembl_id_list = df['chembl_id'].tolist()
    y_list = df[label].tolist()
    df_result_fidelity = pd.DataFrame({'chembl_id': chembl_id_list, 'label': y_list})
    df_result_atom_importance = pd.DataFrame({'chembl_id': chembl_id_list})
    y_proba_list = []
    y_proba_mask_list = []
    df_result_data = []

    if model_name == 'GCN':
        for i, test_sample in enumerate(df):
            test_sample = test_sample.to(device)
            modified_sample = test_sample.clone()

            with torch.no_grad():
                original_prediction = torch.sigmoid(model(test_sample)).cpu().item()

            top_nodes, node_importance = identify_influential_nodes_gcn_gradcam(model, modified_sample)

            df_result_data.append(node_importance)

            for node in top_nodes:
                modified_sample.x[node] = torch.zeros_like(modified_sample.x[node])

            with torch.no_grad():
                modified_prediction = torch.sigmoid(model(modified_sample)).cpu().item()

            y_proba_list.append(original_prediction)
            y_proba_mask_list.append(modified_prediction)

        df_result_atom_importance['node_importance'] = df_result_data


    elif model_name == 'MLP':
        X_test = np.stack(df[fingerprint_col].values).astype(np.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

        background = X_test_tensor
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(torch.tensor(X_test, dtype=torch.float32).to(device),
                                            check_additivity=False)

        for i in range(shap_values.shape[0]):
            row = df.iloc[i]
            chembl_id = row["chembl_id"]
            bit_info = row["bit_info"]
            molecule_shap_values = shap_values[i]

            atom_data = {}

            for bit, atom_radius_list in bit_info.items():
                for (atom_idx, radius) in atom_radius_list:
                    if atom_idx not in atom_data:
                        atom_data[atom_idx] = {"shap_values": [], "radiuses": []}
                    atom_data[atom_idx]["shap_values"].append(molecule_shap_values[bit])
                    atom_data[atom_idx]["radiuses"].append(radius)

            for atom_idx, data in atom_data.items():
                df_result_data.append({
                    "chembl_id": chembl_id,
                    "atom_index": atom_idx,
                    "shap_values": data["shap_values"],
                    "radiuses": data["radiuses"]
                })

        df_result_atom_importance = pd.DataFrame(df_result_data)

        for i in range(X_test_tensor.shape[0]):
            sample = X_test_tensor[i].clone()
            with torch.no_grad():
                original_prediction = torch.sigmoid(model(sample.unsqueeze(0))).cpu().item()

            sample_shap_values = shap_values[i]
            top_features = get_top_5_indices(sample_shap_values)

            if i == 0:
                print(sample_shap_values)
                print(top_features)
                print(sample_shap_values[top_features])

            modified_sample = sample.clone()
            for feature in top_features:
                modified_sample[feature] = 1 - modified_sample[feature]
            with torch.no_grad():
                modified_prediction = torch.sigmoid(model(modified_sample.unsqueeze(0))).cpu().item()

            y_proba_list.append(original_prediction)
            y_proba_mask_list.append(modified_prediction)

    elif model_name == 'MLP_VG':
        X_test = np.stack(df[fingerprint_col].values).astype(np.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

        df_result_data = []

        for i in range(X_test.shape[0]):
            row = df.iloc[i]
            chembl_id = row["chembl_id"]
            bit_info = row["bit_info"]

            fingerprint = X_test[i]
            fingerprint_tensor = torch.tensor(fingerprint, dtype=torch.float32).to(device)
            saliency_values = compute_saliency_map_mlp(model, fingerprint_tensor)
            molecule_siliency_values = saliency_values
            indices_with_ones = np.where(fingerprint == 1)[0]
            saliency_values_for_ones = molecule_siliency_values[indices_with_ones]

            sorted_indices = np.argsort(np.abs(saliency_values_for_ones))[::-1][:5]
            top_bits = indices_with_ones[sorted_indices]

            atom_data = {}

            for bit, atom_radius_list in bit_info.items():
                for (atom_idx, radius) in atom_radius_list:
                    if atom_idx not in atom_data:
                        atom_data[atom_idx] = {"shap_values": [], "radiuses": []}
                    atom_data[atom_idx]["shap_values"].append(molecule_siliency_values[bit])
                    atom_data[atom_idx]["radiuses"].append(radius)

            for atom_idx, data in atom_data.items():
                df_result_data.append({
                    "chembl_id": chembl_id,
                    "atom_index": atom_idx,
                    "shap_values": data["shap_values"],
                    "radiuses": data["radiuses"]
                })

        df_result_atom_importance = pd.DataFrame(df_result_data)

        for i in range(X_test_tensor.shape[0]):

            fingerprint = X_test[i]
            fingerprint_tensor = torch.tensor(fingerprint, dtype=torch.float32).to(device)
            saliency_values = compute_saliency_map_mlp(model, fingerprint_tensor)
            molecule_siliency_values = saliency_values
            indices_with_ones = np.where(fingerprint == 1)[0]

            saliency_values_for_ones = molecule_siliency_values[indices_with_ones]

            sorted_indices = np.argsort(np.abs(saliency_values_for_ones))[::-1][:5]
            top_bits = indices_with_ones[sorted_indices]

            sample = X_test_tensor[i].clone()

            if i == 0:
                print(top_bits)
                print(molecule_siliency_values[top_bits])

            modified_sample = sample.clone()
            for feature in top_bits:
                modified_sample[feature] = 1 - modified_sample[feature]

    elif model_name in ['RF', 'XGB']:
        explainer = shap.TreeExplainer(model)
        X_test = np.stack(df[fingerprint_col].values).astype(np.float32)

        y_proba = model.predict_proba(X_test)[:, 1]

        shap_values = explainer.shap_values(X_test, check_additivity=False)
        if model_name == 'RF':
            shap_values = shap_values[1]

        df_result_data = []

        for i in range(shap_values.shape[0]):
            row = df.iloc[i]
            chembl_id = row["chembl_id"]
            bit_info = row["bit_info"]
            molecule_shap_values = shap_values[i]

            atom_data = {}

            for bit, atom_radius_list in bit_info.items():
                for (atom_idx, radius) in atom_radius_list:
                    if atom_idx not in atom_data:
                        atom_data[atom_idx] = {"shap_values": [], "radiuses": []}
                    atom_data[atom_idx]["shap_values"].append(molecule_shap_values[bit])
                    atom_data[atom_idx]["radiuses"].append(radius)

            for atom_idx, data in atom_data.items():
                df_result_data.append({
                    "chembl_id": chembl_id,
                    "atom_index": atom_idx,
                    "shap_values": data["shap_values"],
                    "radiuses": data["radiuses"]
                })

        top_positive_indices = [get_top_5_indices(shap_element) for shap_element in shap_values]

        X_test_masked = X_test.copy()
        logging.info(f"X_test_masked: {X_test_masked.shape}")
        for i, indices in enumerate(top_positive_indices):
            X_test_masked[i, indices] = 1 - X_test_masked[i, indices]  # Flip bits

        y_proba_masked = model.predict_proba(X_test_masked)[:, 1]

        y_proba_list.extend(y_proba.tolist())
        y_proba_mask_list.extend(y_proba_masked.tolist())

        df_result_atom_importance = pd.DataFrame(df_result_data)

    df_result_fidelity[f'pred_proba'] = np.array(y_proba_list)
    df_result_fidelity[f'pred_proba_mask'] = np.array(y_proba_mask_list)
    return df_result_fidelity, df_result_atom_importance

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate model predictions and atom importance to "
                                                 "further analysis of model fidelity and pharmacophore score.")
    parser.add_argument("--model", type=str,
                        choices=['GCN', 'MLP', 'RF', 'XGB', 'MLP_VG', 'GCN_VG'],
                        required=False,
                        default='GCN',
                        help="Model type to load and generate predictions.")
    parser.add_argument("--dataset", choices=["cdk2"], default="cdk2", required=False, help="Dataset choice.")
    parser.add_argument("--model_label", choices=['class', 'activity'], default='class', required=False, help="Y label column")
    parser.add_argument("--label", choices=['class', 'activity'], default='class', required=False, help="Y label column")

    args = parser.parse_args()
    c_model, dataset, model_label, label = args.model, args.dataset, args.model_label, args.label

    if c_model.startswith('GCN'):
        with open(f'../data/{dataset}/graph_data_{model_label}.p', 'rb') as f:
            df = pickle.load(f)
        test_df = [data for data in df if data.split == 'test']
    else:
        df = pd.read_parquet(f"../data/{dataset}/raw.parquet")
        test_df = df[(df['split'] == 'test')].reset_index(drop=True)

    if c_model == 'GCN':
        model = load_gcn_model(use_hooks=True)  # Load a version with hooks for Grad-CAM

    elif c_model == 'GCN_VG':
        model = load_gcn_model(use_hooks=False)

    elif c_model.startswith('MLP'):
        X_array, bit_info_list = gen_X_fingerprint(test_df, radius=2)
        test_df['bit_info'] = bit_info_list
        model = load_mlp_model(dataset, model_label)

    elif c_model == 'RF':
        X_array, bit_info_list = gen_X_fingerprint(test_df, 2)
        test_df['bit_info'] = bit_info_list
        model = load_rf_model(dataset, model_label)

    elif c_model in 'XGB':
        X_array, bit_info_list = gen_X_fingerprint(test_df, 2)
        test_df['bit_info'] = bit_info_list
        model = load_xgb_model(dataset, model_label)


    fidelity_results, atom_importance_results = (
        add_model_predictions_and_atom_importance_to_df(test_df, model, c_model, label))

    fidelity, accuracy, masked_accuracy = count_fidelity(fidelity_results)
    logging.info(f"Fidelity: {fidelity}")
    logging.info(f"Accuracy: {accuracy}")
    logging.info(f"Masked Accuracy: {masked_accuracy}")

    output_dir = f"results/{dataset}/prediction"
    os.makedirs(output_dir, exist_ok=True)
    output_file_name = f"{output_dir}/{c_model}_{model_label}_{label}_pred_mask"

    fidelity_results.to_csv(f"{output_file_name}_fidelity.csv", index=False)
    atom_importance_results.to_parquet(f"{output_file_name}_atom_importance.parquet")
    logging.info(f"Test dataset with {c_model} predictions saved to {output_file_name}")
    logging.info(f"Count fidelity: {output_file_name}")


# python check_fidelity_and_pharmacoscore.py --model "RF" --model_label 'class' --label 'class'
