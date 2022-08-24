import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from preprocessing.config_preproc import PreprocConfig as CFG_P
from training import helpers_flat_training as helpers_tr

logreg_kwargs = {
    'C' : 100000, 
    'max_iter' : 200, 
    'random_state' : CFG_P.seed        
}

scaler = StandardScaler()

model_names = ['L1_logistic_base',
               'L1_lgbm_dart_base']

tr_feature_files = [CFG_P.model_output_dir + f'{m}/oof_preds.parquet' for m in model_names]
te_feature_files = [CFG_P.model_output_dir + f'{m}/test_preds.csv' for m in model_names]

df_train = helpers_tr.load_flat_features(tr_feature_files, is_meta=True)
df_train = pd.merge(df_train, pd.read_parquet(CFG_P.output_dir + 'train_labels_w_p2_strat_folds.parquet'),
                    on='customer_ID')

features = [f for f in df_train.columns if f not in CFG_P.non_features]
for f in features:
    df_train[f] = df_train[f].clip(lower=.0001, upper=.9999)
df_train[features] = np.log(df_train[features] / (1-df_train[features]))
print(df_train[features].corr())

df_test = helpers_tr.load_flat_features(te_feature_files, parser=pd.read_csv, is_meta=True)

for f in features:
    df_test[f] = df_test[f].clip(lower=.0001, upper=.9999)
df_test[features] = np.log(df_test[features] / (1-df_test[features]))

scaler.fit(pd.concat([df_train[features], df_test[features]]))
df_train[features] = scaler.transform(df_train[features])
df_test[features] = scaler.transform(df_test[features]) 
print(df_test[features].corr())

helpers_tr.train_save_flat_model(df_train, df_test,
                                 helpers_tr.get_logreg_model, logreg_kwargs,
                                 helpers_tr.get_logreg_imp, 'L2_logistic_base')

full_lr, predict_func = helpers_tr.get_logreg_model(df_train[features], df_train['target'],
                                                    None, None,
                                                    None, logreg_kwargs)

test_preds = df_test[['customer_ID']]
test_preds['prediction'] = 0
test_preds['prediction'] = predict_func(full_lr, df_test[features])

out_path = CFG_P.model_output_dir + f'L2_EXP_full_LR_stack'
test_preds.to_csv(out_path + f'/test_preds.csv', index=False)