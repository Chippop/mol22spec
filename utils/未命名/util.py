# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from hashlib import md5
import logging
import os

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
def pad_1d_tokens(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
    ):
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res


def pad_2d(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
    ):
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size, size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :, size - len(v) :] if left_pad else res[i][: len(v), : len(v)])
    return res


def pad_coords(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
    ):
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size, 3).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :, :] if left_pad else res[i][: len(v),:])
    return res

from module.data.Dictionary import Dictionary

def batch_collate_fn(samples):
        device = "cuda"
        weight_dir = "/vepfs/fs_users/yftc/code/mol2spec_git/module/weights"
        dictionary = Dictionary.load(os.path.join(weight_dir,"mol.dict.txt"))
        padding_idx = dictionary.pad()

        batch = {}
        for k in samples[0][0].keys():
            if k == 'src_coord':
                v = pad_coords([torch.tensor(s[0][k],dtype=torch.float32).to(device) for s in samples], pad_idx=0.0)
            elif k == 'src_edge_type':
                v = pad_2d([torch.tensor(s[0][k],dtype=torch.long).to(device) for s in samples], pad_idx=padding_idx)
            elif k == 'src_distance':
                v = pad_2d([torch.tensor(s[0][k],dtype=torch.float32).to(device) for s in samples], pad_idx=0.0)
            elif k == 'src_tokens':
                v = pad_1d_tokens([torch.tensor(s[0][k],dtype=torch.long).to(device) for s in samples], pad_idx=padding_idx)
            batch[k] = v
        try:
            label = torch.tensor([s[1] for s in samples],dtype=torch.float32).to(device)
        except:
            label = None
        return batch, label
def padding_smi_Search(samples):
        device = "cuda"
        weight_dir = "/vepfs/fs_users/yftc/code/mol2spec_git/module/weights"
        dictionary = Dictionary.load(os.path.join(weight_dir,"mol.dict.txt"))
        padding_idx = dictionary.pad()

        batch = {}
        for k in samples.keys():
            if k == 'src_coord':
                v = pad_coords([torch.tensor(samples[k],dtype=torch.float32).to(device)], pad_idx=0.0)
            elif k == 'src_edge_type':
                v = pad_2d([torch.tensor(samples[k],dtype=torch.long).to(device)], pad_idx=padding_idx)
            elif k == 'src_distance':
                v = pad_2d([torch.tensor(samples[k],dtype=torch.float32).to(device)], pad_idx=0.0)
            elif k == 'src_tokens':
                v = pad_1d_tokens([torch.tensor(samples[k],dtype=torch.long).to(device)], pad_idx=padding_idx)
            batch[k] = v
        return batch


def formatsmi_one(smi):
    weigth_path = "/vepfs/fs_users/yftc/code/mol2spec_git/module/weights"
    atoms, coordinates = inner_smi2coords(smi = smi, seed = 42, mode = "fast", remove_hs = True)
    dictionary = Dictionary.load(os.path.join(weigth_path,"mol.dict.txt"))
    dictionary.add_symbol("[MASK]", is_special=True)
    reaslut = coords2unimol(atoms=atoms, coordinates=coordinates, dictionary=dictionary)
    
    
    return reaslut


def inner_coords(atoms, coordinates, remove_hs=True):
    
    assert len(atoms) == len(coordinates), "coordinates shape is not align atoms"
    coordinates = np.array(coordinates).astype(np.float32)
    if remove_hs:
        idx = [i for i, atom in enumerate(atoms) if atom != 'H']
        atoms_no_h = [atom for atom in atoms if atom != 'H']
        coordinates_no_h = coordinates[idx]
        assert len(atoms_no_h) == len(coordinates_no_h), "coordinates shape is not align with atoms"
        return atoms_no_h, coordinates_no_h
    else:
        return atoms, coordinates
    
def coords2unimol(atoms, coordinates, dictionary, max_atoms=256, remove_hs=True, **params):
    
    atoms, coordinates = inner_coords(atoms, coordinates, remove_hs=remove_hs)
    atoms = np.array(atoms)
    coordinates = np.array(coordinates).astype(np.float32)
    # cropping atoms and coordinates
    if len(atoms) > max_atoms:
        idx = np.random.choice(len(atoms), max_atoms, replace=False)
        atoms = atoms[idx]
        coordinates = coordinates[idx]
    # tokens padding
    src_tokens = np.array([dictionary.bos()] + [dictionary.index(atom) for atom in atoms] + [dictionary.eos()])
    src_distance = np.zeros((len(src_tokens), len(src_tokens)))
    # coordinates normalize & padding
    src_coord = coordinates - coordinates.mean(axis=0)
    src_coord = np.concatenate([np.zeros((1,3)), src_coord, np.zeros((1,3))], axis=0)
    # distance matrix
    src_distance = distance_matrix(src_coord, src_coord)
    # edge type
    src_edge_type = src_tokens.reshape(-1, 1) * len(dictionary) + src_tokens.reshape(1, -1)

    return {
            'src_tokens': src_tokens.astype(int), 
            'src_distance': src_distance.astype(np.float32), 
            'src_coord': src_coord.astype(np.float32), 
            'src_edge_type': src_edge_type.astype(int),
            }

def distance_matrix(x, y, p=2, threshold=1000000):
    x = np.asarray(x)
    m, k = x.shape
    y = np.asarray(y)
    n, kk = y.shape

    if k != kk:
        raise ValueError(f"x contains {k}-dimensional vectors but y contains "
                         f"{kk}-dimensional vectors")

    if m*n*k <= threshold:
        return minkowski_distance(x[:,np.newaxis,:],y[np.newaxis,:,:],p)
    else:
        result = np.empty((m,n),dtype=float)  # FIXME: figure out the best dtype
        if m < n:
            for i in range(m):
                result[i,:] = minkowski_distance(x[i],y,p)
        else:
            for j in range(n):
                result[:,j] = minkowski_distance(x,y[j],p)
        return result
def minkowski_distance_p(x, y, p=2):
    
    x = np.asarray(x)
    y = np.asarray(y)

    # Find smallest common datatype with float64 (return type of this
    # function) - addresses #10262.
    # Don't just cast to float64 for complex input case.
    common_datatype = np.promote_types(np.promote_types(x.dtype, y.dtype),
                                       'float64')

    # Make sure x and y are NumPy arrays of correct datatype.
    x = x.astype(common_datatype)
    y = y.astype(common_datatype)

    if p == np.inf:
        return np.amax(np.abs(y-x), axis=-1)
    elif p == 1:
        return np.sum(np.abs(y-x), axis=-1)
    else:
        return np.sum(np.abs(y-x)**p, axis=-1)


def minkowski_distance(x, y, p=2):
    x = np.asarray(x)
    y = np.asarray(y)
    if p == np.inf or p == 1:
        return minkowski_distance_p(x, y, p)
    else:
        return minkowski_distance_p(x, y, p)**(1./p)
    

def inner_smi2coords(smi, mode='fast', remove_hs=True, seed = 42):
    logger = logging.getLogger()
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:  # Check if RDKit was able to parse the SMILES string
            return None, None
        mol = AllChem.AddHs(mol)
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        if len(atoms) == 0:
            return None, None
        res = AllChem.EmbedMolecule(mol, randomSeed=seed)
        if res == 0 or (res == -1 and mode == 'heavy'):
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except:
                pass
            coordinates = mol.GetConformer().GetPositions().astype(np.float32)
        else:
            AllChem.Compute2DCoords(mol)
            coordinates = mol.GetConformer().GetPositions().astype(np.float32)

        if remove_hs:
            idx = [i for i, atom in enumerate(atoms) if atom != 'H']
            atoms = [atom for i, atom in enumerate(atoms) if i in idx]
            coordinates = coordinates[idx]

        return atoms, coordinates
    except Exception as e:
        logger.error(f"Failed to process SMILES: {smi} with error: {e}")
        return None, None
