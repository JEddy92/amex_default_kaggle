import pandas as pd

from preprocessing.config_preproc import PreprocConfig as CFG_P
from training import helpers_flat_training as helpers_tr

lgb_params = {
    'objective': 'binary',
    'metric': "binary_logloss",
    'boosting': 'dart',
    'seed': CFG_P.seed,
    'max_depth': 9,
    'num_leaves': 200,
    'learning_rate': 0.05,
    'colsample_bytree': 0.7,
    'bagging_freq': 10,
    'bagging_fraction': 0.95,
    'reg_alpha': 2,
    'reg_lambda': 2,
    'n_jobs': -1,
    'min_data_in_leaf': 20,
    'drop_rate': .30,
    'max_drop': 100, 
    'skip_drop': .40
}

lgb_kwargs = {
    'params' : lgb_params,
    'num_boost_round' : 3000,
    'callbacks' : [helpers_tr.get_lgb_dart_callback()],
}

feature_fnames = ['{}_basic_agg_features.parquet','{}_flattened_his_features.parquet',
                  '{}_NA_agg_features.parquet']            
tr_feature_files = [CFG_P.output_dir + f.format('train') for f in feature_fnames]
te_feature_files = [CFG_P.output_dir + f.format('test') for f in feature_fnames]

df_train = helpers_tr.load_flat_features(tr_feature_files)
df_train = pd.merge(df_train, pd.read_parquet(CFG_P.output_dir + 'train_labels_w_folds.parquet'),
                    on='customer_ID')

df_test = helpers_tr.load_flat_features(te_feature_files)

helpers_tr.train_save_flat_model(df_train, df_test,
                                 helpers_tr.get_lgb_dart_model, lgb_kwargs,
                                 'lgbm_dart_drop_params')