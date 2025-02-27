# pharmacoscore-benchmark

## Creating and Managing a Conda Environment from env.yml file

## 1. Create the Environment

If you have an environment configuration file (`env.yml`), you can create the environment using:

```sh
conda env create -f env.yml
```

## 2. Activate the Environment

Once the installation is complete, activate the environment with:

```sh
conda activate myenv
```

## 3. If env conda failed then try to use requirements.txt

n requirements.txt are all required libraries and to create env just use:

```sh
pip install -r requirements.txt
```
