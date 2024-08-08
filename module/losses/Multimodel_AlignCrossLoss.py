import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from module.smiles2peaksDim import smiles2peaksDim

class MultiModelAlignLoss(nn.Module):
    def __init__(self, temperature = 0.07) -> None:
        super(MultiModelAlignLoss,self).__init__()
        self.temperature = temperature
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, smiles_embd, peaks_embd):
        smiles_embd = F.normalize(smiles_embd, p = 2, dim=-1).float()
        peaks_embd = F.normalize(peaks_embd, p = 2,dim=-1).float()
        # peaks_embd = F.normalize(smiles_embd, p = 2,dim=-1).float()

        # log_scale = nn.Parameter(torch.ones[1] * np.log(1/self.temperature)).exp().float()
       
        s2pD = smiles2peaksDim(smiles_embd.shape[1], peaks_embd.shape[1]).to("cuda")
        smiles_embd = s2pD(smiles_embd)
        logits_per_smiles = torch.matmul(smiles_embd, peaks_embd.T) / self.temperature
        logits_per_peaks = torch.matmul(peaks_embd, smiles_embd.T) / self.temperature

        batch_size = smiles_embd.shape[0]
        labels = torch.arange(batch_size, device=smiles_embd.device)

        loss_smiles2peaks = self.cross_entropy_loss(logits_per_smiles, labels)
        loss_peaks2smiles = self.cross_entropy_loss(logits_per_peaks, labels)

        loss = (loss_peaks2smiles + loss_smiles2peaks) / 2 

        return loss

