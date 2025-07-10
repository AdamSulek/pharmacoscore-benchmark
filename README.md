# pharmacoscore-benchmark

This repository accompanies our ICCS 2025 paper, *"Explainable Artificial Intelligence for Bioactivity Prediction: Unveiling the Challenges with Curated CDK2/4/6 Breast Cancer Dataset"* ([PDF](https://www.iccs-meeting.org/archive/iccs2025/papers/159030021.pdf)).

## Table of Contents
- [Run Instructions](#run-instructions)
  - [1. Creating the Conda Environment](#1-creating-the-conda-environment)
  - [2. Activating the Environment](#2-activating-the-environment)
  - [3. Alternative: Using requirements.txt](#3-alternative-using-requirementstxt)
- [Scripts](#scripts)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
  - [Pharmacophore Alignment](#pharmacophore-alignment)
  - [Fidelity Analysis](#fidelity-analysis)
- [Datasets](#datasets)


## Run Instructions

### 1. Creating the Conda Environment

If you have an environment configuration file (`env.yml` for linux and `env-windows.yml` for windows system), you can create the environment using:

```sh
conda env create -f env.yml
```

### 2. Activating the Environment

Once the installation is complete, activate the environment with:

```sh
conda activate myenv
```

### 3. Alternative: Using requirements.txt

If creating the Conda environment fails, you can install the required dependencies using requirements.txt:

```sh
pip install -r requirements.txt
```

## Scripts

### Model Training
_train{model}.py_

The train_{model}.py scripts are used to train different models on a given dataset.
It supports various models such as GCN, MLP, RF, and XGB. The script preprocesses
the data, trains the model, and saves the best model based on validation performance.

```sh
python train_{model}.py --dataset 'cdk2' --label 'class'
```

Arguments:
* dataset: The dataset choice. Default is cdk2.
* label: The Y label column. Choices are class, activity, y. Default is y.
* filename: Filename of dataset, for example _raw_ or _decoy_.

### Model Evaluation
_check_model.py_

The check_model.py script is used to evaluate the performance of different
models on a test dataset. It supports various models such as GCN, MLP, RF, and XGB.
The script loads the appropriate model, makes predictions on the test data, and evaluates
the model's performance using metrics like ROC AUC, accuracy, precision, recall, F1 score,
and confusion matrix.

```sh
python check_model.py --model 'RF' --model_dataset 'cdk2' --validate_dataset 'cdk2' --model_label 'class' --validate_label 'class'
```

Arguments:
* model: The model type to load and generate predictions. Choices are GCN, MLP, RF, XGB.
* model_dataset: The dataset used to train the model.
* validate_dataset: The dataset used for validation.
* model_label: The model label column. Choices are class, activity, y. Default is y.
* validate_label: The Y label column for validation. Choices are class, activity, y. Default is y.
* model_filename: Filename of training dataset, for example raw.
* validate_filename: Filename of validate dataset, for example decoy.
* threshold: threshold to count metrics. By default it is 0.5.

### Pharmacophore Alignment
_check_pharmacophore_allignment_and_sparsity.py_

This script checks the model interpretability in terms of pharmacophore alignment and sparsity.

```sh
python check_pharmacophore_allignment_and_sparsity.py --model <MODEL> --dataset <DATASET>
```
Arguments:  
* model: The model type to load and generate predictions. Choices are RF, MLP, GCN, GCN_VG, XGB, MLP_VG.
* dataset: The dataset choice. Default is cdk2.
* label: The training label column. Choices are class, activity, y. Default is y.
* filename: Filename of dataset, for example raw.


### Fidelity Analysis
_check_fidelity_and_pharmacoscore.py_

This script calculates model predictions and atom importance to further analyze model fidelity and pharmacophore score.

```sh
python fidelity/check_fidelity_and_pharmacoscore.py --model <MODEL> --dataset <DATASET> --model_label <MODEL_LABEL> --label <LABEL>
```

Arguments:
* model: The model type to load and generate predictions. Choices are GCN, MLP, RF, XGB, MLP_VG, GCN_VG.
* dataset: The dataset choice. Default is cdk2.
* model_label: The model label column. Choices are class, activity, y. Default is y.
* label: The Y label column. Choices are class, activity, y. Default is y.
* filename: Filename of dataset, for example raw.

## Datasets

The datasets contains the following files:
* pharmacophore_labels.parquet: Contains the pharmacophore labels for checking pharmacophore allignment.
* raw.parquet: Contains the raw data used for training and validating models, with labels: class, activity, y. Training is based 
on ECFP_2 column, based on Morgan fingerprint without count and with radius 2.
* graph_data_class.p: Contains the graph data for the class model used by GCN.
* graph_data_activity.p: Contains the graph data for the activity model used by GCN.

## Benchmark Dataset 🧪

The benchmark dataset is located at `data/{cdk_index}/pharmacophores_labels.parquet`.  
Each pharmacophore type is represented by an atom index or a list of atom indices. Atoms at these positions are labeled as `1`, while all others are labeled as `0`, in case of PharmacoScore calculation.

Medicinal chemists can evaluate our labeling protocol using the 2D plots using plot_labels.py script via:

```sh
python plot_labels.py --dataset 'cdk6'
```

These plots visually highlight labeled pharmacophore atoms, allowing assessment of atom type and interaction type:

- **Hydrophobic** – yellow  
- **Aromatic** – pink  
- **Hydrogen Donor** – orange  
- **Hydrogen Acceptor (Nitrogen)** – green  

### Notes on labeling

- In **fused ring systems**, the entire aromatic system is labeled as aromatic, not just the nearest atoms selected by **Phase (Schrödinger)**.  
- For **hydrophobic interactions**, the entire hydrophobic surface is marked, rather than just the closest atom.
