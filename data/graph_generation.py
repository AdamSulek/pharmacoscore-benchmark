import numpy as np
from rdkit import Chem
import torch 


def one_of_k_encoding(x, allowable_set, default=None):
    if x not in allowable_set:
        if default is None:
            raise ValueError(f"{x} not in allowable set {allowable_set}:")
        else:
            x = default
    return list(map(lambda s: x == s, allowable_set))


def get_atom_features(atom):
    """
    Extract atom data and return an array of numeric features.

    Args:
        atom (rdkit.Chem.rdchem.Atom): Atom to extract features from.

    Returns:
        list: List of numbers that represents attributes of an atom.
    """
    ELEMENT_SYMBOLS = ["Br", "C", "Cl", "F", "H", "I", "N", "O", "P", "S", "Unknown"]
    HYBRIDIZATION_TYPES = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]

    result = []
    result += one_of_k_encoding(atom.GetSymbol(), ELEMENT_SYMBOLS, default="Unknown")
    result += one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result += one_of_k_encoding(
        atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6], default=6
    )
    result += [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
    result += one_of_k_encoding(
        atom.GetHybridization(),
        HYBRIDIZATION_TYPES,
        default=Chem.rdchem.HybridizationType.SP3D2,
    )
    result += [atom.GetIsAromatic()]
    result += one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4], default=4)
    return result


def get_bond_features(bond):
    """
    Extract bond data and return an array of numeric features.

    Args:
        bond (rdkit.Chem.rdchem.Bond): Bond to extract features from.

    Returns:
        list: List of numbers that represents attributes of a bond.
    """
    BOND_TYPES = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    return one_of_k_encoding(
        bond.GetBondType(), BOND_TYPES, default=Chem.rdchem.BondType.AROMATIC
    )


def get_edge(bond):
    # edge as adjacency list entry
    return [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]


def make_ligand_graphs(mol):
    """
    Make graphs from molecules. Each graph is set of data on connectivity, nodes, edges and spatial positions.

    Args:
        molecules ([rdkit.Chem.rdchem.Mol]): List of molecules.

    Returns:
        list: List of graphs. Each graph is tuple of node features, positions, edges and edge features.
    """
    edge_features = []
    edges = []

    for bond in mol.GetBonds():
        edge_feature = get_bond_features(bond)
        edge = get_edge(bond)
        # edge has to be added twice because graph is directed
        edge_features.append(edge_feature)
        edges.append(edge)
        edge_features.append(edge_feature)
        edges.append(edge[::-1])

    node_features = [get_atom_features(atom) for atom in mol.GetAtoms()]

    edges = np.array(edges, dtype=np.int8)
    edge_features = np.array(edge_features, dtype=np.int8)
    node_features = np.array(node_features, dtype=np.int8)

    x = torch.tensor(node_features, dtype=torch.float)
    edges_t = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_features_t = torch.tensor(edge_features, dtype=torch.long)
    
    result = (x, edges_t, edge_features_t)
    
    return result
