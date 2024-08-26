
import os
import sys
sys.path.append("/vepfs/fs_users/yftc/code/mol2spec_git")
import numpy as np
import pandas as pd
from task.task_module import PreprocessorAndEncoder


if __name__ == "__main__":
    input_dim = 1
    embd_dim = 256
    device = 'cuda'
    num_head = 8
    ff_dim = 512
    num_layer = 8
    output_dim = 128
    dataSearch = "/vepfs/fs_users/yftc/code/mol2spec_git/data/task/nega_test"
    model_path = "/vepfs/fs_ckps/yftc/mol2spec/1e-05_8_64_bins2000/bins2000_best_model_epoch_1e-05_bs64_nl8.pt"
    # data_path = "/vepfs/fs_users/yftc/code/mol2spec_git/data/nega_test_smi_format.npy"
    data_path = "/vepfs/fs_users/yftc/code/mol2spec_git/data/nega_test1.pkl"

    df = pd.read_pickle(data_path)
    smiles_feature = df["cleaned_smiles"].values
    
    encoder_smi = []
    smi_map = []
    encoder_peak = []
    peak_map = []

    test = "/vepfs/fs_users/yftc/code/mol2spec_git/data/task/nega_test/encoded_smi.npy"
    test1 = "/vepfs/fs_users/yftc/code/mol2spec_git/data/task/nega_test/encode"


    PAE = PreprocessorAndEncoder(model_path=model_path, device=device,input_dim=input_dim, output_dim=output_dim, embd_dim=embd_dim,ff_dim=ff_dim,num_layer=num_layer,num_head=num_head)
    peaks_feature = PAE._peaks_preprocessing_logic(allPeaks_path=data_path)

    dataSearch = "/vepfs/fs_users/yftc/code/mol2spec_git/data/task/nega_test"
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
        encoder_peak.append(peak_embd)
        peak_embd = peak_embd.cpu().numpy()
        peak_map.append((peak_embd,peak))

    np.save(os.path.join(dataSearch, 'encoded_peak.npy'), np.array(encoder_smi))
    np.save(os.path.join(dataSearch, 'peak_map.npy'), np.array(smi_map, dtype=object))