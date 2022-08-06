import pandas as pd

from preprocessing.config_preproc import PreprocConfig as CFG_P
from training import helpers_flat_training as helpers_tr

lgb_params = {
    'objective': 'binary',
    'metric': "binary_logloss",
    'boosting': 'dart',
    'seed': CFG_P.seed,
    'max_depth': 7,
    'num_leaves': 120,
    'learning_rate': 0.04,
    'colsample_bytree': 0.80,
    'bagging_freq': 10,
    'bagging_fraction': 0.95,
    'reg_alpha': 1,
    'reg_lambda': 1,
    'n_jobs': -1,
    'min_data_in_leaf': 40,
}

lgb_kwargs = {
    'params' : lgb_params,
    'num_boost_round' : 5000,
    'callbacks' : [helpers_tr.get_lgb_dart_callback()],
}
            
'{}_diff_features.parquet','{}_cat_time_svd_features.parquet'

feature_fnames = ['{}_diff_features.parquet','{}_cat_time_svd_features.parquet']    
tr_feature_files = [CFG_P.output_dir + f.format('train') for f in feature_fnames]
te_feature_files = [CFG_P.output_dir + f.format('test') for f in feature_fnames]

tr_base = [CFG_P.output_dir + 'train_2_statements.parquet']
te_base = [CFG_P.output_dir + 'test_1_statements.parquet']

df_train = helpers_tr.load_flat_features(tr_base)
for f in tr_feature_files:
    df_train = pd.merge(df_train, pd.read_parquet(f), on='customer_ID')
df_train = pd.merge(df_train, pd.read_parquet(CFG_P.output_dir + 'train_labels_w_folds.parquet'),
                    on='customer_ID')

df_test = helpers_tr.load_flat_features(te_base)
for f in te_feature_files:
    df_test = pd.merge(df_test, pd.read_parquet(f), on='customer_ID')

helpers_tr.train_save_flat_model(df_train, df_test,
                                 helpers_tr.get_lgb_dart_model, lgb_kwargs,
                                 helpers_tr.get_lgb_imp, 'lgbm_dart_2_statements_tr_diff_cat_time_svd')