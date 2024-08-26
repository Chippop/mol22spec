import os
import faiss
import torch
import numpy as np
import sys
sys.path.append("/vepfs/fs_users/yftc/code/mol2spec_git")
from task_module.PreprocessorAndEncoder import PreprocessorAndEncoder



if __name__ == "__main__":
    input_dim = 1
    embd_dim = 256
    device = 'cuda'
    num_head = 8
    ff_dim = 512
    num_layer = 8
    output_dim = 128

    encode_peaks = '/vepfs/fs_users/yftc/code/mol2spec_git/data/task/nega_test/encoded_peak.npy'
    peak_map = '/vepfs/fs_users/yftc/code/mol2spec_git/data/task/nega_test/peak_map.npy'
    model_path = "/vepfs/fs_ckps/yftc/mol2spec/1e-05_8_64_bins2000/bins2000_best_model_epoch_1e-05_bs64_nl8.pt"

    encode_peaks = np.load(encode_peaks)
    peak_map = np.load(peak_map, allow_pickle = True)
    encode_peaks = encode_peaks.squeeze(1)
    index = faiss.IndexFlatL2(encode_peaks.shape[1])
    index.add(encode_peaks)

    PAE = PreprocessorAndEncoder(model_path=model_path, device=device,input_dim=input_dim, output_dim=output_dim, embd_dim=embd_dim,ff_dim=ff_dim,num_layer=num_layer,num_head=num_head)
    smi = "Cc1[nH]c2ccc(Br)cc2c1CCN=C(O)c1cc2ccccc2oc1=O"
    smi_pad = PAE._smiles_preprocessing_logic(smi)
    smi_embd = PAE.encoder_smiles(smi_pad)
    D, I = index.search(smi_embd.cpu().numpy(), k=5)
    # 获取 top K 个 PEAKS
    top_k_peaks = [peak_map[i][1] for i in I[0]] 
    print(top_k_peaks)



