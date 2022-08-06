import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from preprocessing.config_preproc import PreprocConfig as CFG_P
from training import helpers_flat_training as helpers_tr

avgrank_kwargs = {}

model_names = ['lgbm_dart_2_statements_aug',
               #'xgb_base_afterpay',
               #'xgb_base',
               #'lgbm_dart_no_P2',
               'lgbm_dart_diff_aggs']
               #'lgbm_dart_2_statements_tr_diff_cat_time_svd',
               #'lgbm_dart_na_corrected',
               #'lgbm_dart_round2_scale_pos_10',
               #'lgbm_dart_round2_bagging',
               #'lgbm_dart_round2_last_feats_fe_filt',
               #'lgbm_dart_last_feats_fe_filt',
               #'lgbm_dart_diff_minus_mean',
               #'lgbm_dart_base_diff',
               #'lgbm_dart_flattened_full',
               #'lgbm_dart_2_statements_tr',
               #'lgbm_dart_diff_&_cat_time_svd',
               #'lgbm_dart_flattened_full_time_aligned']  
               #'lgbm_dart_base_2.8x_pos','lgbm_base_refined']
               #'lgbm_dart_base'
               #'lgbm_dart_diff_&_date',
          
tr_feature_files = [CFG_P.model_output_dir + f'{m}/oof_preds.parquet' for m in model_names]
te_feature_files = [CFG_P.model_output_dir + f'{m}/test_preds.csv' for m in model_names]

df_train = helpers_tr.load_flat_features(tr_feature_files, is_meta=True)
df_train = pd.merge(df_train, pd.read_parquet(CFG_P.output_dir + 'train_labels_w_folds.parquet'),
                    on='customer_ID')
features = [f for f in df_train.columns if f not in CFG_P.non_features]
print(df_train[features].corr())

df_test = helpers_tr.load_flat_features(te_feature_files, parser=pd.read_csv, is_meta=True)

helpers_tr.train_save_flat_model(df_train, df_test,
                                 helpers_tr.get_avgrank_model, avgrank_kwargs,
                                 helpers_tr.get_avgrank_imp, 'L1_avgrank_base')

# full_lr, predict_func = helpers_tr.get_logreg_model(df_train[features], df_train['target'],
#                                                     None, None,
#                                                     None, logreg_kwargs)

# test_preds = df_test[['customer_ID']]
# test_preds['prediction'] = 0
# test_preds['prediction'] = predict_func(full_lr, df_test[features])

# out_path = CFG_P.model_output_dir + f'EXP_full_LR_stack'
# test_preds.to_csv(out_path + f'/test_preds.csv', index=False)