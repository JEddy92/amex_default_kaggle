import gc
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from preprocessing.config_preproc import PreprocConfig as CFG_P
from training import helpers_flat_training as helpers_tr

def auto_log_transform(df, th=5):
    """
    全部をチェックするのは難しい時
    歪度が一定以上であれば、log変換を行う
    """

    log_cols = []
    log_reverse_cols = []

    for col in tqdm(df.columns):
        try:
            skew_org = df[col].skew()
            if skew_org > 0:
                skew_log = np.log1p(df[col] - df[col].min()).skew()
                diff = np.abs(skew_org) - np.abs(skew_log)
                if diff > th:
                    log_cols.append(col)
            else:
                skew_log = np.log1p(-1 * df[col] + df[col].max()).skew()
                diff = np.abs(skew_org) - np.abs(skew_log)
                if diff > th:
                    log_reverse_cols.append(col) 
        except:
            pass

    for col in tqdm(log_cols):
        df[col] = np.log1p(df[col] - df[col].min())

    for col in tqdm(log_reverse_cols):
        df[col] = np.log1p(-1 * df[col] + df[col].max())

    return df

tabnet_kwargs = {}

tabnet_kwargs['pretrain_constructor_kwargs'] = {
    'n_d': 64,
    'n_a': 64,
    'n_steps': 3,
    'gamma': 1.3,
    'n_independent': 3,
    'n_shared': 2,
    'lambda_sparse': 1e-3,
    'optimizer_fn': torch.optim.Adam,
    'optimizer_params': dict(lr=2e-2),# 1e-2
    'scheduler_fn': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'scheduler_params': dict(T_0=200, T_mult=1, eta_min=1e-5, last_epoch=-1, verbose=False),
    'mask_type': 'entmax',
    'verbose': 5,
    'seed': CFG_P.seed
}

tabnet_kwargs['pretrain_fit_kwargs'] = {
    'max_epochs' : 50, #20
    'patience' : 5,
    'pretraining_ratio' : 0.2,
    'batch_size' : 2048
}

tabnet_kwargs['train_constructor_kwargs'] = {
    'n_d': 64,
    'n_a': 64,
    'n_steps': 3,
    'gamma': 1.3,
    'n_independent': 3,
    'n_shared': 2,
    'lambda_sparse': 1e-3,
    'optimizer_fn': torch.optim.Adam,
    'optimizer_params': dict(lr=1e-3, weight_decay=1e-3),
    'scheduler_fn': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'scheduler_params': dict(T_0=200, T_mult=1, eta_min=1e-7, last_epoch=-1, verbose=False),
    'mask_type': 'entmax',
    'verbose': 5,
    'seed': CFG_P.seed,
}

tabnet_kwargs['train_fit_kwargs'] = {
    'max_epochs' : 100,
    'patience' : 10,
    'batch_size' : 2048
}

feature_fnames = ['{}_NA_agg_features.parquet',
                  '{}_refined_agg_features.parquet',
                  #'{}_cat_p2_encoded_features.parquet',
                  #'{}_refined_agg_features_med_3_7.parquet',
                  #'{}_cat_word_count_features.parquet',
                  #'{}_date_features.parquet',
                  #'{}_diff_agg_features.parquet',
                  '{}_diff2_features.parquet']
                  #'{}_cat_time_svd_features.parquet']           
tr_feature_files = [CFG_P.output_dir + f.format('train') for f in feature_fnames]
te_feature_files = [CFG_P.output_dir + f.format('test') for f in feature_fnames]

df_train = helpers_tr.load_flat_features(tr_feature_files)
df_train = pd.merge(df_train, pd.read_parquet(CFG_P.output_dir + 'train_labels_w_10_p2_strat_folds.parquet'),
                    on='customer_ID')
df_train = helpers_tr.filter_out_features(df_train, 
                                          CFG_P.model_output_dir + 'lgbm_dart_big_cat_p2_na_streak/model_log.json',
                                          imp_threshold = 4000) 
df_train = df_train.replace([np.inf, -np.inf], 0).fillna(0)

df_test = helpers_tr.load_flat_features(te_feature_files)
df_test = helpers_tr.filter_out_features(df_test, 
                                         CFG_P.model_output_dir + 'lgbm_dart_big_cat_p2_na_streak/model_log.json',
                                         imp_threshold = 4000) 
df_test = df_test.replace([np.inf, -np.inf], 0).fillna(0)

# df_train = df_train.iloc[:10000]
# df_test = df_test.iloc[:10000]

# log transform
train_cut = df_train.shape[0]
data_feat = pd.concat([df_train, df_test], axis=0)
del df_train, df_test
gc.collect()

data_feat = auto_log_transform(data_feat, th=5)
df_train = data_feat.iloc[:train_cut].reset_index()
df_test = data_feat.iloc[train_cut:].reset_index()
del data_feat
gc.collect()

features = [f for f in df_train.columns if f not in CFG_P.non_features]

scaler = StandardScaler()
print('partial scaler fitting train')
scaler.partial_fit(df_train[features].iloc[:(df_train.shape[0] // 2)])
scaler.partial_fit(df_train[features].iloc[(df_train.shape[0] // 2):])

print('partial scaler fitting test')
chunk_size = (df_test.shape[0] // 4) + 1
for i in range(4): 
    print(f' test chunk {i}')
    low, high = i * chunk_size, (i + 1) * chunk_size
    scaler.partial_fit(df_test[features].iloc[low:high])

print('Transforming data')
df_test[features] = scaler.transform(df_test[features])
gc.collect()
df_train[features] = scaler.transform(df_train[features])
gc.collect()

helpers_tr.train_save_flat_model(df_train, df_test,
                                 helpers_tr.get_tabnet_model, tabnet_kwargs,
                                 helpers_tr.get_tabnet_imp, 'tabnet_base',
                                 cat_mode='ohe')