import json
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb

from preprocessing.config_preproc import PreprocConfig as CFG_P
from training import helpers_flat_training as helpers_tr

lgb_params = {
    'objective': 'binary',
    'metric': "binary_logloss",
    'seed': CFG_P.seed,
    'num_leaves': 127, #164
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
    'verbose_eval' : 100,
}

# load data and filter to last
df_train = pd.read_parquet(CFG_P.train_feature_file)
print(df_train.shape)

df_train = df_train.groupby('customer_ID').tail(1)

df_train = pd.merge(df_train, pd.read_parquet(CFG_P.output_dir + 'train_labels_w_folds.parquet'),
                    on='customer_ID')

print(df_train.shape)

features = [c for c in df_train.columns if c not in CFG_P.non_features]
cat_features = CFG_P.cat_features

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df_train[cat_features] = encoder.fit_transform(df_train[cat_features]).astype(int)

val_mask = df_train['val_fold_n'] == 0 
X_train, X_val = df_train.loc[~val_mask, features], df_train.loc[val_mask, features]
y_train, y_val = df_train.loc[~val_mask, 'target'], df_train.loc[val_mask, 'target']

model, _ = helpers_tr.get_lgb_model(X_train, y_train, X_val, y_val, 
                                    cat_features, lgb_kwargs)

# Extract bottom 2 and top 2 splits for all numerical features from model, save as json
# Split value is treated as lower bound of bin for lower clip,
# but upper bound of bin for higher clip
lgb_feature_splits = {}
for f in features:
    if f not in cat_features:
        split_bins = model.get_split_value_histogram(f)[1] 
        if f == 'P_2':
            print(split_bins)
        if len(split_bins) >= 6:
            lgb_feature_splits[f] = (split_bins[0], split_bins[2], split_bins[-3], split_bins[-1])  
        else:
            print(f'{f} does not have enough bins')

print(lgb_feature_splits['P_2'])
print(df_train['P_2'].min(), df_train['P_2'].max())

out_path = CFG_P.output_dir 
with open(out_path + "/lgb_capping_feature_splits.json", "w") as outfile:
    json.dump(lgb_feature_splits, outfile)