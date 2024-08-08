import torch
import pandas as pd
import numpy as np
from torch import optim
import os
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm
sys.path.append("/vepfs/fs_users/yftc/code/mol2spec_git")
# from code.mol2sepc.module.Multimodel import Multimodel

from module.data.LMDBDataset import SmilesPeaksDataset
from module.data.datanorm import PeakProcessor
from module.Multimodel import Multimodel
from utils.util import batch_collate_fn
from module.losses.Multimodel_AlignCrossLoss import MultiModelAlignLoss

def contrastive_loss(smiles_embedding, peaks_embedding):
    print(smiles_embedding.shape,peaks_embedding.shape)
    cosine_sim = nn.functional.cosine_similarity(smiles_embedding, peaks_embedding)
    loss = 1 - cosine_sim.mean()
    return loss





if __name__ == "__main__":
    lmdb_file = "/vepfs/fs_users/yftc/code/mol2spec_git/data/test.lmdb"

    input_dim = 1
    embd_dim = 256
    num_head = 8
    ff_dim = 512
    num_layer = 6

    lr = 0.01
    num_epochs = 50
    temperature = 0.07
    batch_size = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Multimodel(input_dim=input_dim, embd_dim=embd_dim, num_head=num_head, ff_dim=ff_dim, num_layer=num_layer).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    dataset = SmilesPeaksDataset(lmdb_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,collate_fn=batch_collate_fn)

    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_cosine_smi = 0
        for smiles, peaks in tqdm(dataloader):
            smiles, peaks = smiles,peaks.to(device)
            optimizer.zero_grad()
            smiles_embd,peaks_embd = model(smiles,peaks)
            MLoss = MultiModelAlignLoss(temperature=temperature)
            loss = MLoss(smiles_embd, peaks_embd)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss}")

        

