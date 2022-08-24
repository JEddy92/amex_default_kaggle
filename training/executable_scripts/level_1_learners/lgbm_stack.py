import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from preprocessing.config_preproc import PreprocConfig as CFG_P
from training import helpers_flat_training as helpers_tr


# oof_pred_catboost_base_v3                                    6.781556e+06
# oof_pred_lgbm_dart_flattened_full_time_aligned_cat_p2_enc    1.655506e+05
# oof_pred_xgb_big_features                                    1.500066e+06
# oof_pred_lgbm_dart_big_cat_word_p2_strat                     4.908465e+06
# oof_pred_lgbm_dart_refined_aggs_large_fset_dates             6.561868e+06
# oof_pred_lgbm_dart_refined_aggs_large_fset                   3.962520e+06
# oof_pred_lgbm_dart_refined_aggs                              5.447968e+06
# oof_pred_lgbm_dart_2_statements_aug                          3.615451e+06
# oof_pred_xgb_base_afterpay                                   1.658261e+06
# oof_pred_xgb_base                                            1.873137e+05
# oof_pred_lgbm_dart_no_P2                                     3.856455e+06
# oof_pred_lgbm_dart_diff_aggs                                 1.447635e+06
# oof_pred_lgbm_dart_2_statements_tr_diff_cat_time_svd         2.189600e+04
# oof_pred_lgbm_dart_na_corrected                              1.914565e+06
# oof_pred_lgbm_dart_flattened_full                            1.139182e+05
# oof_pred_lgbm_dart_2_statements_tr                           1.961331e+05
# oof_pred_lgbm_dart_flattened_full_time_aligned               8.767898e+03
# dtype: float64
# Total out of fold exact Amex val score is 0.7773505831501056
# Average out of fold exact Amex val score is 0.8003166234404977
# All fold scores:
# [0.8037515020088386, 0.796322660886414, 0.7996054767449938, 0.802443845438527, 0.7994596321237153]

lgb_params = {
    'objective': 'binary',
    'metric': "binary_logloss",
    'boosting': 'dart',
    'seed': CFG_P.seed,
    #'max_depth': 3,
    'num_leaves': 7, #20 
    'learning_rate': 0.005,
    'colsample_bytree': 0.33,
    'bagging_freq': 10,
    'bagging_fraction': 0.95,
    'reg_alpha': 2, 
    'reg_lambda': 2, 
    'n_jobs': -1,
    'min_data_in_leaf': 30
}

lgb_kwargs = {
    'params' : lgb_params,
    'num_boost_round' : 4000,
    'callbacks' : [helpers_tr.get_lgb_dart_callback()],
}

model_names = ['lgbm_dart_group_rel',
               'lgbm_dart_big_alt_params',
               'keras_transformer_400_impute',
               #'keras_transformer_full_na_imputes',
               #'keras_transformer_cat_enc_cons_aug_8',
               #'keras_transformer_base_cat_enc_v2',
               #'keras_transformer_base_cat_enc',
               #'keras_transformer_base',
               #'keras_flat_full',
               'keras_flat_2aug',
               #'keras_flat_base_v5_10F_bags',
               #'keras_flat_base_v4_bags',
               #'keras_flat_base_v3_bags',
               #'keras_flat_base_v2',
               #'catboost_flattened_full_time_aligned',
               #'catboost_base_v4_na_streak',
               'lgbm_dart_big_cat_p2_na_streak',
               'catboost_base_v3',
               #'catboost_base_v2',
               #'catboost_base', #
               'lgbm_dart_flattened_full_time_aligned_cat_p2_enc',
               #'lgbm_dart_big_cat_p2', #
               #'lgbm_dart_big_med_3_7', #
               'xgb_big_features',
               'lgbm_dart_big_cat_word_p2_strat',
               'lgbm_dart_refined_aggs_large_fset_dates',
               'lgbm_dart_refined_aggs_large_fset',
               'lgbm_dart_refined_aggs', 
               'lgbm_dart_2_statements_aug',
               'xgb_base_afterpay',
               'xgb_base', 
               'lgbm_dart_no_P2',  
               'lgbm_dart_diff_aggs',
               'lgbm_dart_2_statements_tr_diff_cat_time_svd',
               'lgbm_dart_na_corrected', 
               #'lgbm_dart_round2_scale_pos_10', 
               #'lgbm_dart_round2_bagging', #
               #'lgbm_dart_round2_last_feats_fe_filt', 
               #'lgbm_dart_last_feats_fe_filt', #
               #'lgbm_dart_diff_minus_mean', #
               #'lgbm_dart_base_diff', #
               'lgbm_dart_flattened_full',
               'lgbm_dart_2_statements_tr', #,
               #'lgbm_dart_diff_&_cat_time_svd', #
               'lgbm_dart_flattened_full_time_aligned']#  
               #'lgbm_dart_base_2.8x_pos','lgbm_base_refined']
               #'lgbm_dart_base'
               #'lgbm_dart_diff_&_date',

# model_names = ['keras_flat_2aug',
#                'keras_flat_base_v2',
#                'lgbm_dart_big_cat_p2_na_streak',
#                'catboost_base_v3',
#                'lgbm_dart_flattened_full_time_aligned_cat_p2_enc',
#                'xgb_big_features',
#                'xgb_base_afterpay',
#                'lgbm_dart_big_cat_word_p2_strat',
#                'lgbm_dart_refined_aggs_large_fset_dates',
#                'lgbm_dart_2_statements_aug',
#                'lgbm_dart_2_statements_tr_diff_cat_time_svd',
#                'lgbm_dart_na_corrected', 
#                'lgbm_dart_flattened_full',
#                'lgbm_dart_2_statements_tr', 
#                'lgbm_dart_flattened_full_time_aligned']
          
# csv_model_names = []

tr_feature_files = [CFG_P.model_output_dir + f'{m}/oof_preds.parquet' for m in model_names]
te_feature_files = [CFG_P.model_output_dir + f'{m}/test_preds.csv' for m in model_names]

df_train = helpers_tr.load_flat_features(tr_feature_files, is_meta=True)

df_train = pd.concat([df_train.set_index('customer_ID'),
                      pd.read_csv(CFG_P.model_output_dir + f'ryota_0823/ryota_oof_0823.csv')
                                       .set_index('customer_ID'), 
                      pd.read_csv(CFG_P.model_output_dir + f'angus_0822_main/oof_ver2.csv')
                                       .set_index('customer_ID')], axis=1).reset_index()

# tr_csv_feature_files = [CFG_P.model_output_dir + f'{m}/oof_preds.csv' for m in csv_model_names] 
# df_train = pd.concat([df_train, 
#                       helpers_tr.load_flat_features(tr_csv_feature_files, parser=pd.read_csv, is_meta=True)], axis=1)

# df_train = pd.merge(df_train, pd.read_parquet(CFG_P.output_dir + 'train_labels_w_10_p2_strat_folds.parquet'),
#                     on='customer_ID')
df_train = pd.merge(df_train, pd.read_parquet(CFG_P.output_dir + 'train_labels_w_p2_strat_folds.parquet'),
                    on='customer_ID')
# df_train = pd.merge(df_train, pd.read_parquet(CFG_P.output_dir + 'train_labels_w_folds.parquet'),
#                     on='customer_ID')

features = [f for f in df_train.columns if f not in CFG_P.non_features]
for f in features:
    df_train[f] = df_train[f].clip(lower=.0001, upper=.9999)
df_train[features] = np.log(df_train[features] / (1-df_train[features]))
print(df_train[features].corr())

df_test = helpers_tr.load_flat_features(te_feature_files, parser=pd.read_csv, is_meta=True)
df_test = pd.concat([df_test.set_index('customer_ID'),
                     pd.read_csv(CFG_P.model_output_dir + f'ryota_0823/ryota_sub_0823.csv')
                                      .set_index('customer_ID'),
                     pd.read_csv(CFG_P.model_output_dir + f'angus_0822_main/sub_ver2.csv')
                                      .set_index('customer_ID')] , axis=1).reset_index()

for f in features:
    df_test[f] = df_test[f].clip(lower=.0001, upper=.9999)

helpers_tr.train_save_flat_model(df_train, df_test,
                                 helpers_tr.get_lgb_dart_model, lgb_kwargs,
                                 helpers_tr.get_lgb_imp, 'L1_lgbm_dart_base')

# full_lr, predict_func = helpers_tr.get_logreg_model(df_train[features], df_train['target'],
#                                                     None, None,
#                                                     None, logreg_kwargs)

# test_preds = df_test[['customer_ID']]
# test_preds['prediction'] = 0
# test_preds['prediction'] = predict_func(full_lr, df_test[features])

# out_path = CFG_P.model_output_dir + f'EXP_full_LR_stack'
# test_preds.to_csv(out_path + f'/test_preds.csv', index=False)