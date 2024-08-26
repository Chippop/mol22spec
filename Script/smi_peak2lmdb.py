import csv
import numpy as np
import pickle
import sys
import lmdb
sys.path.append("/vepfs/fs_users/yftc/code/mol2spec_git")
from module.data.datanorm import PeakProcessor




data_path = "/vepfs/fs_users/yftc/code/mol2spec_git/data/nega_train.pkl"
smiles_feature_path = "/vepfs/fs_users/yftc/code/mol2spec_git/data/nega_train_smi_format.npy"#this file need Script/presmi.py
mass_range = 1000
bins = 2000

peakp = PeakProcessor(mass_range, bins ,data_path)
np.set_printoptions(threshold=np.inf)
peaks_feature = peakp.get_peaks_feature()
smiles_feature = np.load(smiles_feature_path, allow_pickle=True)


lmdb_file = "/vepfs/fs_users/yftc/code/mol2spec_git/data/nega_train_bins2000.lmdb"
# csv_file = "/vepfs/fs_users/yftc/code/mol2spec_git/data/nega_train_bins1000.csv"
env = lmdb.open(lmdb_file,map_size = int(1e12))
# txn = env.begin()
# cursor = txn.cursor()
# with open(csv_file, 'w', newline='') as f:
#     writer = csv.writer(f)
#     for key, value in cursor:
#         writer.writerow([key, value])
 
# # 关闭事务和环境
# cursor.close()
# txn.commit()
# env.close()

with env.begin(write = True) as txn:
    for i, (s,p) in enumerate(zip(smiles_feature,peaks_feature)):
        key = str(i).encode("utf-8")
        value = pickle.dumps((s,p))
        txn.put(key,value)
env.close()