import pandas as pd

from preprocessing.config_preproc import PreprocConfig as CFG_P
from training import helpers_flat_training as helpers_tr

lgb_params = {
    'objective': 'binary',
    'metric': "binary_logloss",
    'seed': CFG_P.seed,
    'num_leaves': 250, #164
    'learning_rate': 0.01,
    'feature_fraction': 0.45,
    'bagging_freq': 10,
    'bagging_fraction': 0.90,
    'lambda_l2': 2,
    'n_jobs': -1,
    'min_data_in_leaf': 40
}

lgb_kwargs = {
    'params' : lgb_params,
    'num_boost_round' : 100000,
    'early_stopping_rounds' : 100,
    'verbose_eval' : 500,
}

feature_fnames = ['{}_basic_agg_features.parquet'] #, '{}_NA_agg_features.parquet']            
tr_feature_files = [CFG_P.output_dir + f.format('train') for f in feature_fnames]
te_feature_files = [CFG_P.output_dir + f.format('test') for f in feature_fnames]

df_train = helpers_tr.load_flat_features(tr_feature_files)
df_train = pd.merge(df_train, pd.read_parquet(CFG_P.output_dir + 'train_labels_w_folds.parquet'),
                    on='customer_ID')

df_test = helpers_tr.load_flat_features(te_feature_files)

helpers_tr.train_save_flat_model(df_train, df_test,
                                 helpers_tr.get_lgb_model, lgb_kwargs,
                                 'lgbm_base')