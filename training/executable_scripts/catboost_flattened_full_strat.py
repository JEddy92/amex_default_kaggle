import pandas as pd

from preprocessing.config_preproc import PreprocConfig as CFG_P
from training import helpers_flat_training as helpers_tr

catboost_kwargs = {
    'iterations' : 40000, #12000
    'eval_metric' : 'AUC',
    'metric_period' : 10,
    'random_seed' : CFG_P.seed,
    'learning_rate' : 0.01, #.03
    'max_depth' : 9,
    'colsample_bylevel' : .30,
    'subsample' : .55,
    'reg_lambda' : 5,
    'early_stopping_rounds' : 500,
    'min_data_in_leaf' : 40
}

feature_fnames = ['{}_flattened_full.parquet']            
tr_feature_files = [CFG_P.output_dir + f.format('train') for f in feature_fnames]
te_feature_files = [CFG_P.output_dir + f.format('test') for f in feature_fnames]

df_train = helpers_tr.load_flat_features(tr_feature_files)
print(df_train.info())
df_train = pd.merge(df_train, pd.read_parquet(CFG_P.output_dir + 'train_labels_w_p2_strat_folds.parquet'),
                    on='customer_ID')

df_test = helpers_tr.load_flat_features(te_feature_files)
print(df_test.info())

helpers_tr.train_save_flat_model(df_train, df_test,
                                 helpers_tr.get_catboost_model, catboost_kwargs,
                                 helpers_tr.get_catboost_imp, 'catboost_flattened_full_time_aligned')