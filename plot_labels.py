import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import ast
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def plot_molecule_with_highlights(smi, atom_labels):
    """
    Plots a molecule with specific atoms and bonds highlighted in different colors.
    
    Parameters:
        smi (str): SMILES representation of the molecule.
        atom_labels (dict): Dictionary containing lists of atom indices for different labels.
    """
    colors = {
        "HBD_label": (1, 0.647, 0),  # orange
        "HBA_label": (0, 1, 0),  # Green
        "hydrophobic_label": (1, 1, 0),  # Yellow
        "aromatic_label": (1, 0.75, 0.8)  # pink
    }
    
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        raise ValueError("Invalid SMILES string")
    
    Chem.rdDepictor.Compute2DCoords(mol)
    
    for label in ["aromatic_1_label", "aromatic_2_label"]:
        if label in atom_labels and isinstance(atom_labels[label], str):
            atom_labels[label] = list(map(int, ast.literal_eval(atom_labels[label])))
    
    highlight_atoms = []
    highlight_colors = {}
    highlight_bonds = []
    bond_colors = {}
    
    colored_atoms = set()
    
    # Assign unique colors to HBD, HBA, and hydrophobic first
    for label in ["HBD_label", "HBA_label", "hydrophobic_label"]:
        for atom in atom_labels.get(label, []):
            atom = int(atom)  
            highlight_atoms.append(atom)
            highlight_colors[atom] = colors[label]
            colored_atoms.add(atom)
    
    # Assign aromatic color only if not previously colored
    for label in ["aromatic_1_label"]:
        for atoms in atom_labels.get(label, []):  
            if isinstance(atoms, tuple):  
                for atom in atoms: 
                    atom = int(atom)  
                    if atom not in colored_atoms:
                        highlight_atoms.append(atom)
                        highlight_colors[atom] = colors["aromatic_label"]
                        colored_atoms.add(atom)
            else:  # Single atom case
                atom = int(atoms)  
                if atom not in colored_atoms:
                    highlight_atoms.append(atom)
                    highlight_colors[atom] = colors["aromatic_label"]
                    colored_atoms.add(atom)
    
    # Highlight bonds for aromatic system (pink)
    aromatic_atoms = set()
    for label in ["aromatic_1_label"]:
        for atoms in atom_labels.get(label, []):
            if isinstance(atoms, tuple):
                aromatic_atoms.update(atoms)
            else:
                aromatic_atoms.add(atoms)
    
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        if begin_atom in aromatic_atoms and end_atom in aromatic_atoms:
            highlight_bonds.append(bond.GetIdx())
            bond_colors[bond.GetIdx()] = colors["aromatic_label"]
    
    # Highlight bonds for hydrophobic group (yellow)
    hydrophobic_atoms = atom_labels.get("hydrophobic_label", [])
    hydrophobic_atoms = [int(atom) for atom in hydrophobic_atoms]  
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        if begin_atom in hydrophobic_atoms and end_atom in hydrophobic_atoms:
            highlight_bonds.append(bond.GetIdx())
            bond_colors[bond.GetIdx()] = colors["hydrophobic_label"]
    
    drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors
    )
    drawer.FinishDrawing()
    
    return drawer

if __name__ == "__main__":
    import argparse 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cdk2', 'cdk4', 'cdk6'], required=True)
    args = parser.parse_args()
    
    df_labels = pd.read_parquet(f"data/{args.dataset}/pharmacophore_labels.parquet")
    
    for ind, row in df_labels.iterrows():
        chembl_id = df_labels.iloc[ind]['chembl_id']
        smi = df_labels.iloc[ind]['smiles']
        
        atom_labels = {
            "hydrophobic_label": df_labels.iloc[ind]['hydrophobic'],
            "aromatic_1_label": df_labels.iloc[ind]['aromatic'],
        }

        if args.dataset in ['cdk2', 'cdk6']:
            atom_labels["HBD_label"] = [int(df_labels.iloc[ind]['HBD'])]

        if args.dataset in ['cdk2', 'cdk4']:
            atom_labels["HBA_label"] = [int(df_labels.iloc[ind]['HBA'])]

        d = plot_molecule_with_highlights(smi, atom_labels)
        os.makedirs(f"labels_plot/{args.dataset}", exist_ok=True)
        with open(f"labels_plot/{args.dataset}/label_{df_labels.iloc[ind]['chembl_id']}.png", 'wb') as f:
            f.write(d.GetDrawingText())
    
    logging.info("Labels plotted")  
            
# nohup python plot_labels.py --dataset 'cdk6' > plot_labels_cdk6.log &