import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from preprocessing.config_preproc import PreprocConfig as CFG_P
from training import helpers_flat_training as helpers_tr

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

# .05 ryota 0823 0.8029832635456325
# Below + my group rel is 0.8024368493351013

# .07; .0001 and .9999 clips; highest lb
# Angus0822 Ryota0822
# oof_pred_lgbm_dart_big_alt_params                            0.012694
# oof_pred_keras_transformer_400_impute                        0.298019
# oof_pred_keras_flat_2aug                                     0.020503
# oof_pred_lgbm_dart_big_cat_p2_na_streak                      0.347147
# oof_pred_catboost_base_v3                                    0.147570
# oof_pred_lgbm_dart_flattened_full_time_aligned_cat_p2_enc   -0.062417
# oof_pred_xgb_big_features                                   -0.197618
# oof_pred_lgbm_dart_big_cat_word_p2_strat                     0.162811
# oof_pred_lgbm_dart_refined_aggs_large_fset_dates             0.063412
# oof_pred_lgbm_dart_refined_aggs_large_fset                   0.100512
# oof_pred_lgbm_dart_refined_aggs                             -0.238521
# oof_pred_lgbm_dart_2_statements_aug                          0.380294
# oof_pred_xgb_base_afterpay                                   0.113988
# oof_pred_xgb_base                                            0.103299
# oof_pred_lgbm_dart_no_P2                                    -0.020824
# oof_pred_lgbm_dart_diff_aggs                                -0.227548
# oof_pred_lgbm_dart_2_statements_tr_diff_cat_time_svd         0.139066
# oof_pred_lgbm_dart_na_corrected                             -0.149349
# oof_pred_lgbm_dart_flattened_full                            0.256391
# oof_pred_lgbm_dart_2_statements_tr                           0.216742
# oof_pred_lgbm_dart_flattened_full_time_aligned              -0.842066
# oof_cat_clf_all                                             -0.373102
# oof_cat_clf_base                                             0.172716
# oof_cat_clf_large                                           -0.249063
# oof_lgb_dart_old                                             1.027916
# oof_lgb_gbdt_clf_all                                         0.109600
# oof_tabnet_all_005                                           0.171402
# oof_tabnet_large_001                                         0.096818
# oof_tabnet_large_002                                         0.177280
# oof_tabnet_large_003                                         0.038920
# oof_tabnet_ridge_004                                         0.071061
# oof_xgb_gbdt_clf_all                                         0.910327
# oof_xgb_gbdt_clf_large                                       0.472401
# angus_md_1                                                   0.289616
# angus_md_2                                                   0.083006
# angus_md_3                                                   0.114141
# angus_md_4                                                  -0.505614
# angus_md_5                                                   0.189341
# angus_md_6                                                   0.129166
# angus_md_7                                                   0.338978
# angus_md_8                                                  -0.062912
# angus_md_9                                                   0.051970
# ryota_md_1                                                  -0.180764
# Total out of fold exact Amex val score is 0.802528868035169
# Average out of fold exact Amex val score is 0.8023950036123969
# All fold scores:
# [0.804692429518813, 0.7981020287239073, 0.8020883758941699, 0.8049026484142494, 0.802189535510845]

# .05; .0001 and .9999 clips; highest lb! 
# oof_pred_lgbm_dart_group_rel               0.084846
# oof_pred_lgbm_dart_big_alt_params          0.017835
# oof_pred_keras_transformer_400_impute      0.306211
# oof_pred_keras_flat_2aug                   0.029272
# oof_pred_lgbm_dart_big_cat_p2_na_streak    0.312255
#                                              ...
# angus_md_6                                 0.114703
# angus_md_7                                 0.316577
# angus_md_8                                -0.056227
# angus_md_9                                 0.046467
# ryota_md_1                                -0.164117
# Length: 61, dtype: float64
# Total out of fold exact Amex val score is 0.8029108617551598
# Average out of fold exact Amex val score is 0.8029832635456325
# All fold scores:
# [0.8047207745104467, 0.7991759875933411, 0.8027931174309437, 0.8052413030594763, 0.8029851351339552]

logreg_kwargs = {
    'penalty' : 'elasticnet',
    'solver' : 'saga',
    'C' : .05, # .1 . 035  
    'l1_ratio' : .5,
    'max_iter' : 200, 
    'random_state' : CFG_P.seed        
}

scaler = StandardScaler()
# scaler = MinMaxScaler()

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