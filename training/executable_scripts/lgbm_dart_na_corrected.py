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

feature_fnames = ['{}_null_handled_agg_features.parquet','{}_flattened_his_features.parquet',
                  '{}_diff_agg_features.parquet', '{}_diff_features.parquet',
                  '{}_cat_time_svd_features.parquet']           
tr_feature_files = [CFG_P.output_dir + f.format('train') for f in feature_fnames]
te_feature_files = [CFG_P.output_dir + f.format('test') for f in feature_fnames]

df_train = helpers_tr.load_flat_features(tr_feature_files)
df_train = helpers_tr.filter_out_features(df_train, CFG_P.model_output_dir + 'lgbm_dart_na_corrected/model_log.json')
df_train = pd.merge(df_train, pd.read_parquet(CFG_P.output_dir + 'train_labels_w_folds.parquet'),
                    on='customer_ID')
df_train = df_train.fillna(-1)

df_test = helpers_tr.load_flat_features(te_feature_files)
df_test = helpers_tr.filter_out_features(df_test, CFG_P.model_output_dir + 'lgbm_dart_na_corrected/model_log.json')
df_test = df_test.fillna(-1)

helpers_tr.train_save_flat_model(df_train, df_test,
                                 helpers_tr.get_lgb_dart_model, lgb_kwargs,
                                 helpers_tr.get_lgb_imp, 'lgbm_dart_diff_aggs')