import gc
import numpy as np
import pandas as pd 

from preprocessing.config_preproc import PreprocConfig as CFG 

cat_features = CFG.cat_features 
use_cols = ['customer_ID', 'S_2', 'P_2'] + cat_features 
print('Reading and concatenating data')

df_train = pd.read_parquet(CFG.train_feature_file, columns=use_cols)
df_train['dset'] = 'train'

df_test = pd.read_parquet(CFG.test_feature_file, columns=use_cols)
df_test['dset'] = 'test'

# concatenation to extract additional P2 information from test
df_comb = pd.concat([df_train, df_test]).reset_index(drop=True)
print(f'Combined shape: {df_comb.shape}')

del df_train, df_test
gc.collect()

for cat in cat_features:
    cat_P2_means = df_comb.groupby(cat)['P_2'].mean()
    df_comb[f'{cat}_P2_mean'] = df_comb[cat].map(cat_P2_means) 

df_comb = df_comb.drop(columns=cat_features + ['P_2'])

print(df_comb[df_comb['dset'] == 'train'].drop(columns='dset').shape)

df_comb[df_comb['dset'] == 'train'].drop(columns='dset') \
    .to_parquet(CFG.output_dir + 'train_cat_all_p2_encoded_features.parquet')

df_comb[df_comb['dset'] == 'test'].drop(columns='dset') \
    .to_parquet(CFG.output_dir + 'test_cat_all_p2_encoded_features.parquet')