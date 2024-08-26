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
from torch.utils.tensorboard import SummaryWriter

def contrastive_loss(smiles_embedding, peaks_embedding):
    print(smiles_embedding.shape,peaks_embedding.shape)
    cosine_sim = nn.functional.cosine_similarity(smiles_embedding, peaks_embedding)
    loss = 1 - cosine_sim.mean()
    return loss





if __name__ == "__main__":
    lmdb_file = "/vepfs/fs_users/yftc/code/mol2spec_git/data/nega_train_bins2000.lmdb"
    checkpoint_dir = "/vepfs/fs_ckps/yftc/mol2spec"
    input_dim = 1
    output_dim = 512
    embd_dim = 512
    num_head = 8
    ff_dim = 2048
    num_layer = [6]
    lr = [1e-5]
    num_epochs = 300
    print("300")
    temperature = 0.07
    batch_size = [64]

#     lr_values=(0.00001 0.00005 0.0001 0.0003 0.0005 0.001 0.005)
# local_batch_size_values=(8 16 32)
# dropout_values=(0 0.1 0.2)
# warmup_values=(0.06 0.1)
# encoder_layers_values=(4 5 6 7 8 9 10 11 12 13 14 15)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = Multimodel(input_dim=input_dim, embd_dim=embd_dim, num_head=num_head, ff_dim=ff_dim, num_layer=num_layer)
    # model = nn.DataParallel(model)
    # model.to(device)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # dataset = SmilesPeaksDataset(lmdb_file)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,collate_fn=batch_collate_fn)
    # best_loss = float('inf')

    for p_lr in lr:
        for p_num_layer in num_layer:
            for p_batch_size in batch_size:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = Multimodel(input_dim=input_dim, embd_dim=embd_dim, num_head=num_head, ff_dim=ff_dim, num_layer=p_num_layer,output_dim=output_dim)
                model = nn.DataParallel(model)
                # checkpoint = torch.load("/vepfs/fs_ckps/yftc/mol2spec/1e-05_8_32/bins1000_best_model_epoch_1e-05_bs32_nl8.pt")
                # model.load_state_dict(checkpoint["model_state_dict"])
                model.to(device)
                optimizer = optim.Adam(model.parameters(), lr=p_lr)
                # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                dataset = SmilesPeaksDataset(lmdb_file)
                dataloader = DataLoader(dataset, batch_size=p_batch_size, shuffle=True,collate_fn=batch_collate_fn)
                best_loss = float('inf')
                # best_loss = checkpoint["loss"]
                tailstr = str(p_lr) + "_" + str(p_num_layer) + "_" + str(p_batch_size) + "_bins2000"
                Writerpath = os.path.join(checkpoint_dir,tailstr)
                # 创建文件夹
                try:
                    os.makedirs(Writerpath)
                    print(f"文件夹 '{Writerpath}' 创建成功!")
                except FileExistsError:
                    print(f"文件夹 '{Writerpath}' 已经存在。")
                except Exception as e:
                    print(f"创建文件夹时出错: {e}")
                writer = SummaryWriter(os.path.join(Writerpath,"runs/simple_model_experiment_1"))
            
                for epoch in range(num_epochs):
                    model.train()
                    total_loss = 0
                    num = 0
                    for smiles, peaks in tqdm(dataloader):
                        smiles, peaks = smiles,peaks.to(device)
                        optimizer.zero_grad()
                        smiles_embd,peaks_embd = model(smiles,peaks)
                        MLoss = MultiModelAlignLoss(temperature=temperature)
                        loss = MLoss(smiles_embd, peaks_embd)
                        loss.backward()
                        optimizer.step()
                        # for name, param in model.named_parameters():
                        #     if param.grad is not None:
                        #         print(f"Layer {name} | Gradient Norm: {param.grad.norm().item()}")
                        total_loss += loss.item()
                        
                        if num % 10 == 0:
                            writer.add_scalar('training loss',
                              loss.item(),
                              epoch * len(dataloader) + num)
                        num = 1 + num
                    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")
                    avg_loss = total_loss / len(dataloader)
                    if (avg_loss < best_loss):  # 仅保存更好的模型
                        best_loss = avg_loss
                        checkpoint_path = os.path.join(Writerpath, f"bins2000_best_model_epoch_{p_lr}_bs{p_batch_size}_nl{p_num_layer}.pt")
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': avg_loss,
                        }, checkpoint_path)
                        print(f"Checkpoint saved at {checkpoint_path}")
                
                    writer.add_scalar('epoch loss', avg_loss, epoch)
                writer.close()
                # torch.save(model.state_dict(),"/vepfs/fs_users/yftc/code/mol2spec_git/data/weight/fi_bin200_bs16.pt")