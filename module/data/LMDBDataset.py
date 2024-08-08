import pickle
import lmdb
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os
import sys
sys.path.append("/home/code/mol2sepc")
class SmilesPeaksDataset(Dataset):
    def __init__(self, lmdb_file) -> None:
        self.env = lmdb.open(
            lmdb_file,
            readonly = True,
            lock = False,
            readahead = False,
            meminit = False
        )
        with self.env.begin(write = False) as txn:
            self.length = txn.stat()["entries"]
        
    def __getitem__(self, index):
        key = str(index).encode("utf-8")
        with self.env.begin(write = False) as txn:
            value = txn.get(key)

        smiles, peaks = pickle.loads(value)
        return smiles, peaks
    
    def __len__(self):
        return self.length
    