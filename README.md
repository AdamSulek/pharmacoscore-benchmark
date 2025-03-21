# pharmacoscore-benchmark

## Table of Contents
- [Run Instructions](#run-instructions)
  - [1. Creating the Conda Environment](#1-creating-the-conda-environment)
  - [2. Activating the Environment](#2-activating-the-environment)
  - [3. Alternative: Using requirements.txt](#3-alternative-using-requirementstxt)
- [Scripts](#scripts)
  - [Pharmacophore Alignment](#pharmacophore-alignment)
  - [Fidelity Analysis](#fidelity-analysis)
  - [Model Evaluation](#model-evaluation)
  - [Model Training](#model-training)
- [Datasets](#datasets)


## Run Instructions

### 1. Creating the Conda Environment

If you have an environment configuration file (`env.yml`), you can create the environment using:

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

### Pharmacophore Alignment
_check_pharmacophore_allignment_and_sparsity.py_

This script checks the model interpretability in terms of pharmacophore alignment and sparsity.

```sh
python check_pharmacophore_allignment_and_sparsity.py --model <MODEL> --dataset <DATASET>
```
Arguments:  
* model: The model type to load and generate predictions. Choices are RF, MLP, GCN, GCN_VG, XGB, MLP_VG.
* dataset: The dataset choice. Default is cdk2.


### Fidelity Analysis
_check_fidelity_and_pharmacoscore.py_

This script calculates model predictions and atom importance to further analyze model fidelity and pharmacophore score.

```sh
python fidelity/check_fidelity_and_pharmacoscore.py --model <MODEL> --dataset <DATASET> --model_label <MODEL_LABEL> --label <LABEL>
```

Arguments:
* model: The model type to load and generate predictions. Choices are GCN, MLP, RF, XGB, MLP_VG, GCN_VG.
* dataset: The dataset choice. Default is cdk2.
* model_label: The model label column. Choices are class, activity. Default is class.
* label: The Y label column. Choices are class, activity. Default is class.


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
* model_label: The model label column. Choices are class, activity. Default is class.
* validate_label: The Y label column for validation. Choices are class, activity. Default is class.

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
* label: The Y label column. Choices are class, activity. Default is class.

## Datasets

The datasets contains the following files:
* pharmacophore_labels.parquet: Contains the pharmacophore labels for checking pharmacophore allignment.
* raw.parquet: Contains the raw data used for training and validating models, with labels: class, activity.
* graph_data_class.p: Contains the graph data for the class model used by GCN.
* graph_data_activity.p: Contains the graph data for the activity model used by GCN.