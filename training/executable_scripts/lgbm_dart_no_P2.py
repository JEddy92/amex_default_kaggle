import pandas as pd

from preprocessing.config_preproc import PreprocConfig as CFG_P
from training import helpers_flat_training as helpers_tr

lgb_params = {
    'objective': 'binary',
    'metric': "binary_logloss",
    'boosting': 'dart',
    'seed': CFG_P.seed + 2,
    'max_depth': 7,
    'num_leaves': 120,
    'learning_rate': 0.01,
    'colsample_bytree': 0.45,
    'bagging_freq': 10,
    'bagging_fraction': 0.50,
    'reg_alpha': 2,
    'reg_lambda': 2,
    'n_jobs': -1,
    'min_data_in_leaf': 40
}

lgb_kwargs = {
    'params' : lgb_params,
    'num_boost_round' : 10000,
    'callbacks' : [helpers_tr.get_lgb_dart_callback()],
}

feature_fnames = ['{}_diff_agg_features.parquet',
                  '{}_basic_agg_features.parquet',
                  '{}_diff_features.parquet']           
tr_feature_files = [CFG_P.output_dir + f.format('train') for f in feature_fnames]
te_feature_files = [CFG_P.output_dir + f.format('test') for f in feature_fnames]

df_train = helpers_tr.load_flat_features(tr_feature_files)
p_2_cols = [c for c in df_train.columns if 'P_2_' in c] 
print(df_train.shape)
df_train = df_train.drop(columns=p_2_cols)
print(df_train.shape)
df_train = pd.merge(df_train, pd.read_parquet(CFG_P.output_dir + 'train_labels_w_folds.parquet'),
                    on='customer_ID')

df_test = helpers_tr.load_flat_features(te_feature_files)
df_test = df_test.drop(columns=p_2_cols)

helpers_tr.train_save_flat_model(df_train, df_test,
                                 helpers_tr.get_lgb_dart_model, lgb_kwargs,
                                 helpers_tr.get_lgb_imp, 'lgbm_dart_no_P2')