import os

import torch.nn as nn
import torch
import numpy as np
from module.PeaksPositionalEncode import PeaksPositionalEncode
class _TransformerEncode(nn.Module):
    def __init__(self, input_dim, embd_dim, ff_dim, num_head, num_layer) -> None:
        super(_TransformerEncode,self).__init__()
        self.peak_mlp = nn.Sequential(
            nn.Linear(1, embd_dim >> 1),
            nn.ReLU(),
            nn.Linear(embd_dim >> 1,embd_dim >> 1)
        )
        self.embd = nn.Linear(input_dim, embd_dim >> 1)
        self.pos_encoder = PeaksPositionalEncode(embd_dim = embd_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model= embd_dim, nhead= num_head, dim_feedforward= ff_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers = num_layer)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(embd_dim, embd_dim),
            nn.ReLU(),
            nn.Linear(embd_dim, 128)
        )

    def forward(self,peak_input):
        if isinstance(peak_input, np.ndarray):
            peak_input = torch.tensor(peak_input, dtype=torch.float32)
        peak_x = peak_input[:,:, 0:1]
        intensity = peak_input[:,:, 1:2]

        intensity_embd = self.peak_mlp(intensity)
        peak_x_embd = self.embd(peak_x)

        combine = torch.cat((peak_x_embd, intensity_embd), dim = -1)
        combine = combine.permute(1, 0, 2)
        combine = self.pos_encoder(combine)
        combine = self.transformer_encoder(combine)
        combine = combine.permute(1, 0, 2)

        encode_peaks = combine.mean(dim = 1)
        encode_peaks = self.fc(encode_peaks)

        return encode_peaks
