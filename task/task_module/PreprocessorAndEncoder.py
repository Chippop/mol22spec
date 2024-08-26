


import os
import numpy as np
import pandas as pd
import torch
import sys
sys.path.append("/vepfs/fs_users/yftc/code/mol2spec_git")
from module.Multimodel import Multimodel
from module.data.datanorm import PeakProcessor
from utils.util import formatsmi_one, padding_smi_Search

class PreprocessorAndEncoder:
    def __init__(self, model_path, input_dim, output_dim, embd_dim, ff_dim, num_layer, num_head,device = "cuda"):
        self.model = Multimodel(input_dim=input_dim, embd_dim=embd_dim, num_head=num_head, ff_dim=ff_dim, num_layer=num_layer,output_dim=output_dim)
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.","")
            new_state_dict[new_key] = v
        self.model.load_state_dict(new_state_dict)
        self.model.to(device=device)
        self.model.eval()
        self.output_dim = output_dim
    
    def _peaks_preprocessing_logic(self, allPeaks, mass_range = 1000, bins = 2000):
        peakp = PeakProcessor(peaks_ndarray=allPeaks, mass_range=mass_range, bins=bins)
        np.set_printoptions(threshold=np.inf)
        peaks_feature = peakp.get_peaks_feature()
        return peaks_feature

    def _smiles_preprocessing_logic(self, smi):
        smiFormat = formatsmi_one(smi)
        smiles_pad = padding_smi_Search(smiFormat)
        return smiles_pad
    
    def encoder_peaks(self, peaks_input):
        with torch.no_grad():
            peaks_embd = self.model.peakEncoder(peaks_input)
        return peaks_embd
    
    def encoder_smiles(self, smiles_input):
        with torch.no_grad():
            smiles_embd = self.model.smilesEncoder(**smiles_input, return_repr=True,)['cls_repr']
            # smiles_embd = self.model.s2pD(smiles_embd)
        return smiles_embd
        

if __name__ == "__main__":
    # input_dim = 1
    # embd_dim = 256
    device = 'cuda'
    # num_head = 8
    # ff_dim = 512
    # num_layer = 8
    # output_dim = 128

    input_dim = 1
    output_dim = 512
    embd_dim = 512
    num_head = 8
    ff_dim = 2048
    num_layer = 6
    
    dataSearch = "/vepfs/fs_users/yftc/code/mol2spec_git/data/task/output512_nl6_ffdim2048_embdim512/bins2000"
    # model_path = "/vepfs/fs_ckps/yftc/mol2spec/1e-05_8_64_bins2000/bins2000_best_model_epoch_1e-05_bs64_nl8.pt"
    # model_path = '/vepfs/fs_ckps/yftc/mol2spec/1e-05_8_128_bins1000/bins1000_best_model_epoch_1e-05_bs128_nl8.pt'
    model_path = '/vepfs/fs_ckps/yftc/mol2spec/1e-05_6_64_bins2000/bins2000_best_model_epoch_1e-05_bs64_nl6.pt'
    # data_path = "/vepfs/fs_users/yftc/code/mol2spec_git/data/nega_test_smi_format.npy"
    data_path = "/vepfs/fs_users/yftc/code/mol2spec_git/data/nega_test.pkl"

    df = pd.read_pickle(data_path)
    smiles_feature = df["cleaned_smiles"].values
    peaks_ndarray = df["peaks"].values
    encoder_smi = []
    smi_map = []
    encoder_peak = []
    peak_map = []


    PAE = PreprocessorAndEncoder(model_path=model_path, device=device,input_dim=input_dim, output_dim=output_dim, embd_dim=embd_dim,ff_dim=ff_dim,num_layer=num_layer,num_head=num_head)
    peaks_feature = PAE._peaks_preprocessing_logic(allPeaks=peaks_ndarray)

    
    try:
        os.makedirs(dataSearch)
        print(f"文件夹 '{dataSearch}' 创建成功!")
    except FileExistsError:
        print(f"文件夹 '{dataSearch}' 已经存在。")
    except Exception as e:
        print(f"创建文件夹时出错: {e}")

    for smi in smiles_feature:
        smi_pad = PAE._smiles_preprocessing_logic(smi)
        # print(smi_pad["src_tokens"].device)
        # print(PAE.model.smilesEncoder.device)
        smi_embd = PAE.encoder_smiles(smi_pad)
        smi_embd = smi_embd.cpu().numpy()
        encoder_smi.append(smi_embd)
        smi_map.append((smi_embd, smi))
    
  
    np.save(os.path.join(dataSearch, 'encoded_smi.npy'), np.array(encoder_smi))
    np.save(os.path.join(dataSearch, 'smi_map.npy'), np.array(smi_map, dtype=object))

    for peak in peaks_feature:
        peak_embd = PAE.encoder_peaks(peak)
        peak_embd = peak_embd.cpu().numpy()
        encoder_peak.append(peak_embd)
        peak_map.append((peak_embd,peak))

    np.save(os.path.join(dataSearch, 'encoded_peak.npy'), np.array(encoder_peak))
    np.save(os.path.join(dataSearch, 'peak_map.npy'), np.array(peak_map, dtype=object))
