import torch.nn as nn

class smiles2peaksDim(nn.Module):
    def __init__(self,smiles_embddim,peaks_embddim) -> None:
        super(smiles2peaksDim,self).__init__()
        self.embd = nn.Linear(smiles_embddim, peaks_embddim)

    def forward(self, smiles_embd):
        return self.embd(smiles_embd)
