import gc
import numpy as np
import pandas as pd 

from preprocessing.helpers_preproc import get_svd_features 
from preprocessing.config_preproc import PreprocConfig as CFG    

n_components = 12

print('Reading and concatenating data')

flat_cat_subsets, flat_cat_cols = [], []
for c in CFG.cat_features:

    flat_cat_list = [f'{c}_{i}' for i in range(0, CFG.max_n_statement)] 
    flat_cat_subsets.append(flat_cat_list) 
    flat_cat_cols += flat_cat_list

use_cols = ['customer_ID'] + flat_cat_cols

input_fnames = [CFG.output_dir + '{}_flattened_full.parquet'.format(f) 
                for f in ['train','test']]

df_train = pd.read_parquet(input_fnames[0], columns=use_cols)
df_train['dset'] = 'train'

df_test = pd.read_parquet(input_fnames[1], columns=use_cols)
df_test['dset'] = 'test'

df_comb = pd.concat([df_train, df_test]).reset_index(drop=True)
df_outs = []

for i, c in enumerate(CFG.cat_features):
    print(f"Running SVD on {c} history")
    ohe_cols = flat_cat_subsets[i] 
    df_out = get_svd_features(pd.get_dummies(df_comb[ohe_cols], columns=ohe_cols),
                              n_components, f'{c}_history_svd')
    df_outs.append(df_out)

df_out = pd.concat(df_outs, axis=1)
del df_outs
gc.collect()

df_out['customer_ID'] = df_comb['customer_ID']
df_out['dset'] = df_comb['dset']

print(df_out.shape)
print(df_out.head(10))

df_out[df_out['dset'] == 'train'].drop(columns='dset') \
    .to_parquet(CFG.output_dir + 'train_cat_time_svd_features.parquet')

df_out[df_out['dset'] == 'test'].drop(columns='dset') \
    .to_parquet(CFG.output_dir + 'test_cat_time_svd_features.parquet')