import faiss
import torch
import numpy as np
import sys
sys.path.append("/vepfs/fs_users/yftc/code/mol2spec_git")
from task_module.PreprocessorAndEncoder import PreprocessorAndEncoder
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import TanimotoSimilarity
from rdkit import Chem
import torch.nn.functional as F



def evaluate(ori_smi, smi_list):
    ori_mol = Chem.MolFromSmiles(ori_smi)
    rdkit_gen = rdFingerprintGenerator.GetMorganGenerator(includeChirality=True, radius=2, fpSize=1024)
    ori_finger = rdkit_gen.GetFingerprint(ori_mol)
    ans = []
    for smi in smi_list:
        mol_tmp = Chem.MolFromSmiles(smi)
        finger_tmp = rdkit_gen.GetFingerprint(mol_tmp)
        similarity = TanimotoSimilarity(ori_finger, finger_tmp)
        ans.append((smi, similarity))

    return ans


if __name__ == "__main__":
    # input_dim = 1
    # embd_dim = 256
    device = 'cuda'
    # num_head = 8
    # ff_dim = 512
    # num_layer = 8
    # output_dim = 128
    # bins = 1000

    input_dim = 1
    output_dim = 512
    embd_dim = 512
    num_head = 8
    ff_dim = 2048
    num_layer = 6
    bins = 2000

    encode_smi = '/vepfs/fs_users/yftc/code/mol2spec_git/data/task/output512_nl6_ffdim2048_embdim512/bins2000/encoded_smi.npy'
    smi_map = '/vepfs/fs_users/yftc/code/mol2spec_git/data/task/output512_nl6_ffdim2048_embdim512/bins2000/smi_map.npy'
    # model_path = "/vepfs/fs_ckps/yftc/mol2spec/1e-05_8_64_bins2000/bins2000_best_model_epoch_1e-05_bs64_nl8.pt"
    # model_path = "/vepfs/fs_ckps/yftc/mol2spec/1e-05_8_128_bins1000/bins1000_best_model_epoch_1e-05_bs128_nl8.pt"
    model_path = '/vepfs/fs_ckps/yftc/mol2spec/1e-05_6_64_bins2000/bins2000_best_model_epoch_1e-05_bs64_nl6.pt'
    peaks_path = "/vepfs/fs_users/yftc/code/mol2spec_git/data/nega_test.pkl"


    df = np.load(peaks_path, allow_pickle=True)
    df = df.reset_index(drop=True)
    encode_smi = np.load(encode_smi)
    smi_map = np.load(smi_map, allow_pickle = True)
    encode_smi = encode_smi.squeeze(1)
    # index = faiss.IndexFlatL2(encode_smi.shape[1])
    index = faiss.IndexFlatIP(encode_smi.shape[1])
    index.add(encode_smi)
    PAE = PreprocessorAndEncoder(model_path=model_path, device=device,input_dim=input_dim, output_dim=output_dim, embd_dim=embd_dim,ff_dim=ff_dim,num_layer=num_layer,num_head=num_head)    
    


    # smi_pad = PAE._smiles_preprocessing_logic(ori_smi)
    # smi_embd = PAE.encoder_smiles(smi_pad)
    accNum = 0
    err = 0
    for row in df.index:
        peak = df.loc[row]["peaks"]
        smi = df.loc[row]["cleaned_smiles"]
        peak = np.array([peak])
        try:
             peaks_pad = PAE._peaks_preprocessing_logic(peak, bins=bins)
        except:
           err = err + 1
           continue
        peaks_embd = PAE.encoder_peaks(peaks_pad)
        # peaks_embd = F.normalize(peaks_embd, p=2, dim=-1)
        # smi_pad = PAE._smiles_preprocessing_logic(smi)
        # smi_embd = PAE.encoder_smiles(smi_pad)
        D, I = index.search(peaks_embd.cpu().numpy(), k=100)
        top_k_smi = [smi_map[i][1] for i in I[0]]
        if smi in top_k_smi:
            accNum = 1 + accNum
        # for s in top_k_smi:
        #     smi_pad = PAE._smiles_preprocessing_logic(s)
        #     smi_embd = PAE.encoder_smiles(smi_pad)
        #     cosine_similarity = F.cosine_similarity(peaks_embd, smi_embd)
        #     print("余弦相似度:", cosine_similarity.item())
        # if row > 1:
        #     break
    print(accNum,err)
    print(float(accNum/(20000 - err)))
    # # index.hnsw.efSearch = 70
    # D, I = index.search(peaks_embd.cpu().numpy(), k=10)
    # # 获取 top K 个 PEAKS
    # top_k_smi = [smi_map[i][1] for i in I[0]] 
    # ori_smi = "CCCCCC=CCC=CCCCCCCCC(=O)OCC(O)COP(=O)([O-])OCC[N+](C)(C)C"
    # ans = evaluate(ori_smi, top_k_smi)

    # for i in ans:
    #     print(i)


#test
#CCCCCCCCOC(=O)c1cc(O)c(O)c(O)c1
#Nc1ccccc1C(=O)C[C@H](N)C(=O)O

#train
#Cc1[nH]c2ccc(Br)cc2c1CCN=C(O)c1cc2ccccc2oc1=O top5
#CCc1cccc(CC)c1N(COC)C(=O)CS(=O)(=O)O NAN
#CCCCCC=CCC=CCCCCCCCC(=O)OCC(O)COP(=O)([O-])OCC[N+](C)(C)C