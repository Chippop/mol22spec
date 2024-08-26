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
    lmdb_file = "/vepfs/fs_users/yftc/code/mol2spec_git/data/nega_test.lmdb"
    checkpoint_dir = "/vepfs/fs_ckps/yftc/mol2spec"
    input_dim = 1
    embd_dim = 256
    num_head = 8
    ff_dim = 512
    num_layer = [4 ,5 ,6 ,7 ,8, 9 ,10 ,11 ,12 ,13 ,14 ,15]

    lr = [0.00001,0.00005,0.0001 ,0.0003 ,0.0005 ,0.001 ,0.005]
    num_epochs = 100
    temperature = 0.07
    batch_size = [8,16,32]

    data_path = "data/nega_train.pkl"
    df = pd.read_pickle(data_path)

    data = df[:1000]
    pd.to_pickle(data,"data/nega_train1000.pkl")


    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = Multimodel(input_dim=input_dim, embd_dim=embd_dim, num_head=num_head, ff_dim=ff_dim, num_layer=num_layer, output_dim=).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # dataset = SmilesPeaksDataset(lmdb_file)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,collate_fn=batch_collate_fn)
    # best_loss = float('inf')
    
    # for epoch in range(num_epochs):
    #     model.train()
    #     total_loss = 0
    #     total_cosine_smi = 0
    #     for smiles, peaks in tqdm(dataloader):
    #         smiles, peaks = smiles,peaks.to(device)
    #         optimizer.zero_grad()
    #         smiles_embd,peaks_embd = model(smiles,peaks)
    #         MLoss = MultiModelAlignLoss(temperature=temperature)
    #         loss = MLoss(smiles_embd, peaks_embd)
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
    #         total_loss += loss.item()
    #     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")
    #     avg_loss = total_loss / len(dataloader)
    #     if (avg_loss < best_loss) & (epoch % 10 == 0):  # 仅保存更好的模型
    #         best_loss = avg_loss
    #         checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch_{epoch+1}.pt")
    #         torch.save({
    #             'epoch': epoch + 1,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'loss': avg_loss,
    #         }, checkpoint_path)
    #         print(f"Checkpoint saved at {checkpoint_path}")
    # torch.save(model.state_dict(),"/vepfs/fs_users/yftc/code/mol2spec_git/data/weight/fi_bin200_bs16.pt")
        

