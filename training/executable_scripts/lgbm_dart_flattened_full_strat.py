import pandas as pd

from preprocessing.config_preproc import PreprocConfig as CFG_P
from training import helpers_flat_training as helpers_tr

lgb_params = {
    'objective': 'binary', 
    'metric': 'binary_logloss', 
    'boosting': 'dart', 
    'seed': CFG_P.seed, 
    #'max_depth': 5, 
    'num_leaves': 64, #31
    'learning_rate': 0.01, 
    'colsample_bytree': 0.15, 
    'bagging_freq': 10, 
    'bagging_fraction': 0.95, 
    'reg_alpha': 1, 
    'reg_lambda': 1, 
    'n_jobs': -1, 
    'min_data_in_leaf': 40
}

lgb_kwargs = {
    'params' : lgb_params,
    'num_boost_round' : 10500,
    'callbacks' : [helpers_tr.get_lgb_dart_callback()],
}

feature_fnames = ['{}_flattened_full.parquet', '{}_full_flat_cat_p2_encoded_features.parquet']            
tr_feature_files = [CFG_P.output_dir + f.format('train') for f in feature_fnames]
te_feature_files = [CFG_P.output_dir + f.format('test') for f in feature_fnames]

raw_cats = []
for cat in CFG_P.cat_features:
    raw_cats += [f'{cat}_{i}' for i in range(0,CFG_P.max_n_statement)]

df_train = helpers_tr.load_flat_features(tr_feature_files)
df_train = df_train.drop(columns=raw_cats)
print(df_train.info())
df_train = pd.merge(df_train, pd.read_parquet(CFG_P.output_dir + 'train_labels_w_p2_strat_folds.parquet'),
                    on='customer_ID')

df_test = helpers_tr.load_flat_features(te_feature_files)
df_test = df_test.drop(columns=raw_cats)
print(df_test.info())

helpers_tr.train_save_flat_model(df_train, df_test,
                                 helpers_tr.get_lgb_dart_model, lgb_kwargs,
                                 helpers_tr.get_lgb_imp, 'lgbm_dart_flattened_full_time_aligned_cat_p2_enc')