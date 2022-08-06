import gc
import numpy as np
import pandas as pd 

from preprocessing.config_preproc import PreprocConfig as CFG 

cat_features = CFG.cat_features 
use_cols = ['customer_ID', 'P_2'] + cat_features 
print('Reading and concatenating data to calc last cat means')

df_train = pd.read_parquet(CFG.train_feature_file, columns=use_cols)
df_train['dset'] = 'train'
df_train = df_train.groupby('customer_ID').tail(1)

df_test = pd.read_parquet(CFG.test_feature_file, columns=use_cols)
df_test['dset'] = 'test'
df_test = df_test.groupby('customer_ID').tail(1)

# concatenation to extract additional P2 information from test
df_comb = pd.concat([df_train, df_test]).reset_index(drop=True)
print(f'Combined shape: {df_comb.shape}')

del df_train, df_test
gc.collect()

cat_P2_means = {}

for cat in cat_features:
    cat_P2_means[cat] = df_comb.groupby(cat)['P_2'].mean()

del df_comb
gc.collect()

use_cols = ['customer_ID']
for cat in cat_features:
    use_cols += [f'{cat}_{i}' for i in range(0, CFG.max_n_statement)] 

print('Mapping last cat means to full flattened cats')
df_train = pd.read_parquet(CFG.output_dir + 'train_flattened_full.parquet', columns=use_cols)
df_test = pd.read_parquet(CFG.output_dir + 'test_flattened_full.parquet', columns=use_cols)

for cat in cat_features:
    for i in range(0, CFG.max_n_statement):
        df_train[f'{cat}_{i}_P2_mean'] = df_train[f'{cat}_{i}'].map(cat_P2_means[cat])
        df_train = df_train.drop(columns=[f'{cat}_{i}'])
        df_test[f'{cat}_{i}_P2_mean'] = df_test[f'{cat}_{i}'].map(cat_P2_means[cat])
        df_test = df_test.drop(columns=[f'{cat}_{i}'])

df_train.to_parquet(CFG.output_dir + 'train_full_flat_cat_p2_encoded_features.parquet')
df_test.to_parquet(CFG.output_dir + 'test_full_flat_cat_p2_encoded_features.parquet')