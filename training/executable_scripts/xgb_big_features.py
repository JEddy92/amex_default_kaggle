import numpy as np
import pandas as pd

from preprocessing.config_preproc import PreprocConfig as CFG_P
from training import helpers_flat_training as helpers_tr

xgb_params = {
            'objective' : 'binary:logistic', 
            'tree_method' : 'hist', 
            'random_state' : CFG_P.seed, 
            'n_jobs' : -1,
            'max_depth' : 7,
            'subsample' : 0.75, #.88
            'colsample_bytree' : 0.40, #.5
            'gamma' : 1.5,
            'min_child_weight' : 8,
            'lambda' : 70,
            'eta' : 0.006,
            'max_cat_to_onehot' : 5
    }

xgb_kwargs = {
    'params' : xgb_params,
    'num_boost_round' : 40000, #2600
    'early_stopping_rounds' : 600,
    'verbose_eval' : 100
}

feature_fnames = ['{}_after_pay_agg_features.parquet',
                  '{}_refined_agg_features.parquet',
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
df_train = df_train.fillna(-1)
df_train = df_train.replace([np.inf, -np.inf], -1)

df_test = helpers_tr.load_flat_features(te_feature_files)
df_test = df_test.fillna(-1)
df_test = df_test.replace([np.inf, -np.inf], -1)

helpers_tr.train_save_flat_model(df_train, df_test, 
                                 helpers_tr.get_xgb_model, xgb_kwargs,
                                 helpers_tr.get_xgb_imp, 'xgb_big_features',
                                 cat_mode = 'xgb_cat_mode')