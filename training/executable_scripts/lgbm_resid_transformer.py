import pandas as pd

from preprocessing.config_preproc import PreprocConfig as CFG_P
from training import helpers_flat_training as helpers_tr

lgb_params = {
    'objective': 'regression',
    'metric': "rmse",
    'boosting': 'gbdt',
    'seed': CFG_P.seed,
    #'max_depth': 7,
    'num_leaves': 20, 
    'learning_rate': 0.01,
    'colsample_bytree': 0.50,
    'bagging_freq': 10,
    'bagging_fraction': 0.95,
    'reg_alpha': 1, 
    'reg_lambda': 1, 
    'n_jobs': -1,
    'min_data_in_leaf': 500
}

lgb_kwargs = {
    'params' : lgb_params,
    'num_boost_round' : 11000, #10000
    'early_stopping_rounds' : 50
}

feature_fnames = ['{}_refined_agg_features.parquet']           
tr_feature_files = [CFG_P.output_dir + f.format('train') for f in feature_fnames]
te_feature_files = [CFG_P.output_dir + f.format('test') for f in feature_fnames]

df_train = helpers_tr.load_flat_features(tr_feature_files)
df_train = pd.merge(df_train, pd.read_parquet(CFG_P.output_dir + 'train_labels_w_p2_strat_folds.parquet'),
                    on='customer_ID')
df_preds = pd.read_parquet(CFG_P.model_output_dir + 'keras_transformer_400_impute/oof_preds.parquet')
df_preds.columns = ['customer_ID','preds'] 
df_train = pd.merge(df_train, df_preds, on='customer_ID')

df_test = helpers_tr.load_flat_features(te_feature_files)
df_preds_test = pd.read_csv(CFG_P.model_output_dir + 'keras_transformer_400_impute/test_preds.csv')
df_preds_test.columns = ['customer_ID','preds'] 
df_test = pd.merge(df_test, df_preds_test, on='customer_ID')

helpers_tr.train_save_flat_model(df_train, df_test,
                                 helpers_tr.get_lgb_residual_learner, lgb_kwargs,
                                 helpers_tr.get_lgb_imp, 'lgbm_resid_transformer_400_impute')