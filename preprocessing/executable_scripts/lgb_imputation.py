import gc
import numpy as np
import pandas as pd 
import lightgbm as lgb

from preprocessing.config_preproc import PreprocConfig as CFG 

lgb_params = {
    'objective': 'regression',
    'metric': 'mae',
    'seed': CFG.seed,
    'num_leaves': 64, 
    'learning_rate': 0.1, #.2
    'feature_fraction': 0.75,
    'bagging_freq': 10,
    'bagging_fraction': 0.95,
    'lambda_l2': 2,
    'n_jobs': -1,
    'min_data_in_leaf': 40,
    'force_col_wise': True
}

lgb_kwargs = {
    'params' : lgb_params,
    'num_boost_round' : 400, #100
    'verbose_eval' : 100,
}

df_train = pd.read_parquet(CFG.train_feature_file)
df_train['dset'] = 'train'

df_test = pd.read_parquet(CFG.test_feature_file)
df_test['dset'] = 'test'

# concatenation to extract additional information from test
df_comb = pd.concat([df_train, df_test]).reset_index(drop=True)
print(f'Combined shape: {df_comb.shape}')

df_comb[df_comb == -1] = np.nan

del df_train, df_test
gc.collect()

cat_features = CFG.cat_features 
num_features = [f for f in df_comb.columns if f not in (CFG.non_features + cat_features + ['dset'])]
features = cat_features + num_features

null_grid = df_comb[num_features].isnull() 
null_feats = list(null_grid.columns[null_grid.any()])
print(f'{len(null_feats)} to impute')

df_comb_imputed = df_comb.copy()
print(df_comb_imputed.isnull().sum())

for i, f in enumerate(null_feats):
    print(f'Imputing {f}, the {i}th feature of {len(null_feats)}')

    # FE - add 1 lag feature for each customer on the fly
    # lag_feat = f + '_lag'
    # df_comb[lag_feat] = df_comb.groupby('customer_ID')[f].shift()

    X_train = df_comb[~null_grid[f]][features].drop(columns=[f])  
    y_train = df_comb[~null_grid[f]][f] 

    X_impute = df_comb[null_grid[f]][features].drop(columns=[f])

    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature = cat_features)
    
    model = lgb.train(train_set = lgb_train, valid_sets = [lgb_train],
                      **lgb_kwargs)

    del lgb_train
    gc.collect()

    df_comb_imputed.loc[null_grid[f],f] = model.predict(X_impute)

print(df_comb_imputed.isnull().sum())
print(df_comb_imputed.shape)

df_comb_imputed[df_comb_imputed['dset'] == 'train'].drop(columns='dset') \
    .to_parquet(CFG.output_dir + 'train_lgb_400_imputed.parquet')

df_comb_imputed[df_comb_imputed['dset'] == 'test'].drop(columns='dset') \
    .to_parquet(CFG.output_dir + 'test_lgb_400_imputed.parquet')