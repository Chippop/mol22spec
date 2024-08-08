import math
import torch.nn as nn
import torch


class PeaksPositionalEncode(nn.Module):
    def __init__(self, embd_dim, dropout = 0.1,max_len = 5000):
        super(PeaksPositionalEncode, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, embd_dim)
        position = torch.arange(0., max_len).unsqueeze(1)
        para = torch.exp(torch.arange(0., embd_dim, 2) *
                             -(math.log(10000.0) / embd_dim))
        
        pe[:, 0::2] = torch.sin(position * para).to("cuda")
        pe[:, 1::2] = torch.cos(position * para).to("cuda")
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer("pe",pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
        