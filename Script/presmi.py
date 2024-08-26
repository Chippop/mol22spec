
import numpy as np
import pandas as pd
import sys
sys.path.append("/vepfs/fs_users/yftc/code/mol2spec_git")
from module.data.Dictionary import Dictionary
import os
from utils.util import coords2unimol,inner_smi2coords
weigth_path = "/vepfs/fs_users/yftc/code/mol2spec_git/module/weights"
def formatsmi(smi_list):
    smiles_formatList = []
    for smiles in smi_list:
        atoms, coordinates = inner_smi2coords(smi = smiles, seed = 42, mode = "fast", remove_hs = True)
        dictionary = Dictionary.load(os.path.join(weigth_path,"mol.dict.txt"))
        dictionary.add_symbol("[MASK]", is_special=True)
        reaslut = coords2unimol(atoms=atoms, coordinates=coordinates, dictionary=dictionary)
        smiles_formatList.append(reaslut)
        # print(reaslut["src_distance"])
        # print(type(reaslut["src_distance"]))
    return smiles_formatList

data_path = "/vepfs/fs_users/yftc/code/mol2spec_git/data/nega_test.pkl"
df = pd.read_pickle(data_path)
# smiles_list = ["C1=CC=C(C=C1)C=O","CCCCC/C(=C/C1=CC=CC=C1)/C=O"]
smiles_feature = df["cleaned_smiles"].values
smiles_feature = smiles_feature[:10]
smiles_formatList = formatsmi(smiles_feature)


print(smiles_formatList[0]["src_tokens"])
# import numpy as np
# a = np.array(smiles_formatList)
# np.save("/vepfs/fs_users/yftc/code/mol2sepc/data/1111111.npy",a)

# file = "/vepfs/fs_users/yftc/code/mol2sepc/data/1111.npy"
# test = np.load(file,allow_pickle=True)
# print(len(test))
