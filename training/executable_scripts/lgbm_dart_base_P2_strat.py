import pandas as pd

from preprocessing.config_preproc import PreprocConfig as CFG_P
from training import helpers_flat_training as helpers_tr

lgb_params = {
    'objective': 'binary',
    'metric': "binary_logloss",
    'boosting': 'dart',
    'seed': CFG_P.seed,
    #'max_depth': 7,
    'num_leaves': 145, 
    'learning_rate': 0.01,
    'colsample_bytree': 0.28,
    'bagging_freq': 10,
    'bagging_fraction': 0.95,
    'reg_alpha': 2, 
    'reg_lambda': 2, 
    'n_jobs': -1,
    'min_data_in_leaf': 30
}

lgb_kwargs = {
    'params' : lgb_params,
    'num_boost_round' : 11000, #10000
    'callbacks' : [helpers_tr.get_lgb_dart_callback()],
}

feature_fnames = ['{}_NA_agg_features.parquet',
                  '{}_refined_agg_features.parquet',
                  '{}_cat_p2_encoded_features.parquet',
                  #'{}_refined_agg_features_med_3_7.parquet',
                  '{}_cat_word_count_features.parquet',
                  '{}_date_features.parquet',
                  '{}_diff_agg_features.parquet',
                  '{}_diff2_features.parquet',
                  '{}_cat_time_svd_features.parquet']           
tr_feature_files = [CFG_P.output_dir + f.format('train') for f in feature_fnames]
te_feature_files = [CFG_P.output_dir + f.format('test') for f in feature_fnames]

df_train = helpers_tr.load_flat_features(tr_feature_files)
df_train = pd.merge(df_train, pd.read_parquet(CFG_P.output_dir + 'train_labels_w_p2_strat_folds.parquet'),
                    on='customer_ID')

df_test = helpers_tr.load_flat_features(te_feature_files)

helpers_tr.train_save_flat_model(df_train, df_test,
                                 helpers_tr.get_lgb_dart_model, lgb_kwargs,
                                 helpers_tr.get_lgb_imp, 'lgbm_dart_big_cat_p2_na_streak')