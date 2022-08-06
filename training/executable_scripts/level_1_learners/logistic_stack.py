import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from preprocessing.config_preproc import PreprocConfig as CFG_P
from training import helpers_flat_training as helpers_tr

# .085; P2 stratified folds; highest LB
# oof_pred_catboost_base_v2                                    0.386680
# oof_pred_lgbm_dart_flattened_full_time_aligned_cat_p2_enc    0.619860
# oof_pred_xgb_big_features                                    0.036201
# oof_pred_lgbm_dart_big_cat_word_p2_strat                     0.793739
# oof_pred_lgbm_dart_refined_aggs_large_fset_dates             0.425955
# oof_pred_lgbm_dart_refined_aggs_large_fset                   0.374608
# oof_pred_lgbm_dart_refined_aggs                             -0.164618
# oof_pred_lgbm_dart_2_statements_aug                          1.195759
# oof_pred_xgb_base_afterpay                                   0.146552
# oof_pred_xgb_base                                            0.178514
# oof_pred_lgbm_dart_no_P2                                     0.276101
# oof_pred_lgbm_dart_diff_aggs                                -0.286832
# oof_pred_lgbm_dart_2_statements_tr_diff_cat_time_svd         0.020284
# oof_pred_lgbm_dart_na_corrected                             -0.043434
# oof_pred_lgbm_dart_flattened_full                            0.517190
# oof_pred_lgbm_dart_2_statements_tr                           0.687836
# oof_pred_lgbm_dart_flattened_full_time_aligned              -1.477201
# dtype: float64
# Total out of fold exact Amex val score is 0.8004995157524442
# Average out of fold exact Amex val score is 0.800372059969383
# All fold scores:
# [0.8038650423653255, 0.7972439226868637, 0.8008612783463778, 0.8016766176378207, 0.7982134388105271]

# 1; .9999 clip; highest lb
# oof_pred_keras_flat_base_v2                                  0.253489
# oof_pred_catboost_base_v3                                    0.506703
# oof_pred_lgbm_dart_flattened_full_time_aligned_cat_p2_enc    0.742857
# oof_pred_xgb_big_features                                   -0.116302
# oof_pred_lgbm_dart_big_cat_word_p2_strat                     0.735184
# oof_pred_lgbm_dart_refined_aggs_large_fset_dates             0.383662
# oof_pred_lgbm_dart_refined_aggs_large_fset                   0.359864
# oof_pred_lgbm_dart_refined_aggs                             -0.222868
# oof_pred_lgbm_dart_2_statements_aug                          1.317359
# oof_pred_xgb_base_afterpay                                   0.223452
# oof_pred_xgb_base                                            0.169480
# oof_pred_lgbm_dart_no_P2                                     0.219816
# oof_pred_lgbm_dart_diff_aggs                                -0.407705
# oof_pred_lgbm_dart_2_statements_tr_diff_cat_time_svd         0.050621
# oof_pred_lgbm_dart_na_corrected                             -0.116052
# oof_pred_lgbm_dart_flattened_full                            0.624909
# oof_pred_lgbm_dart_2_statements_tr                           0.681689
# oof_pred_lgbm_dart_flattened_full_time_aligned              -1.834563
# dtype: float64
# Total out of fold exact Amex val score is 0.800466737256516
# Average out of fold exact Amex val score is 0.80063079209164
# All fold scores:
# [0.804267183068762, 0.7974950311791993, 0.8008644254587591, 0.8017920253525481, 0.7987352953989315]

logreg_kwargs = {
    'C' : .7, #. 035  
    'max_iter' : 200, 
    'random_state' : CFG_P.seed        
}

scaler = StandardScaler()
# scaler = MinMaxScaler()

model_names = ['keras_flat_full',
               'keras_flat_2aug',
               #'keras_flat_base_v5_10F_bags',
               #'keras_flat_base_v4_bags',
               #'keras_flat_base_v3_bags',
               #'keras_flat_base_v2',
               #'catboost_flattened_full_time_aligned',
               #'catboost_base_v4_na_streak',
               #'lgbm_dart_big_cat_p2_na_streak',
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
          
tr_feature_files = [CFG_P.model_output_dir + f'{m}/oof_preds.parquet' for m in model_names]
te_feature_files = [CFG_P.model_output_dir + f'{m}/test_preds.csv' for m in model_names]

df_train = helpers_tr.load_flat_features(tr_feature_files, is_meta=True)
df_train = pd.merge(df_train, pd.read_parquet(CFG_P.output_dir + 'train_labels_w_p2_strat_folds.parquet'),
                    on='customer_ID')
# df_train = pd.merge(df_train, pd.read_parquet(CFG_P.output_dir + 'train_labels_w_folds.parquet'),
#                     on='customer_ID')

features = [f for f in df_train.columns if f not in CFG_P.non_features]
for f in features:
    df_train[f] = df_train[f].clip(upper=.9999)
df_train[features] = np.log(df_train[features] / (1-df_train[features]))
print(df_train[features].corr())

df_test = helpers_tr.load_flat_features(te_feature_files, parser=pd.read_csv, is_meta=True)
for f in features:
    df_test[f] = df_test[f].clip(upper=.9999)
df_test[features] = np.log(df_test[features] / (1-df_test[features]))

scaler.fit(pd.concat([df_train[features], df_test[features]]))
df_train[features] = scaler.transform(df_train[features])
df_test[features] = scaler.transform(df_test[features]) 
print(df_test[features].corr())

helpers_tr.train_save_flat_model(df_train, df_test,
                                 helpers_tr.get_logreg_model, logreg_kwargs,
                                 helpers_tr.get_logreg_imp, 'L1_logistic_base')

full_lr, predict_func = helpers_tr.get_logreg_model(df_train[features], df_train['target'],
                                                    None, None,
                                                    None, logreg_kwargs)

test_preds = df_test[['customer_ID']]
test_preds['prediction'] = 0
test_preds['prediction'] = predict_func(full_lr, df_test[features])

out_path = CFG_P.model_output_dir + f'EXP_full_LR_stack'
test_preds.to_csv(out_path + f'/test_preds.csv', index=False)