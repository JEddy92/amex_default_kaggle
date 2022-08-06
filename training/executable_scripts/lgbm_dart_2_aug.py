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

tr1_features_files = [#'train_diff_agg_features.parquet',
                      'train_basic_agg_features.parquet', 
                      'train_diff2_features.parquet']
tr1_features_files = [CFG_P.output_dir + f for f in tr1_features_files]

tr2_features_files = [#'train_ex_last1_diff_agg_features.parquet',
                      'train_ex_last1_basic_agg_features.parquet', 
                      'train_ex1_diff2_features.parquet']
tr2_features_files = [CFG_P.output_dir + f for f in tr2_features_files]

test_features_files = [#'test_diff_agg_features.parquet',
                       'test_basic_agg_features.parquet', 
                       'test_diff2_features.parquet']
test_features_files = [CFG_P.output_dir + f for f in test_features_files]

df_train = pd.concat([helpers_tr.load_flat_features(tr1_features_files),
                      helpers_tr.load_flat_features(tr2_features_files)])

df_train = pd.merge(df_train, pd.read_parquet(CFG_P.output_dir + 'train_labels_w_folds.parquet'),
                    on='customer_ID')            

df_test = helpers_tr.load_flat_features(test_features_files)

helpers_tr.train_save_flat_model(df_train, df_test,
                                 helpers_tr.get_lgb_dart_model, lgb_kwargs,
                                 helpers_tr.get_lgb_imp, 'lgbm_dart_2_statements_aug')