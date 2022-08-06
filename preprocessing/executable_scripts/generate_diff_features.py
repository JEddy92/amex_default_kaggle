import gc
import numpy as np
import pandas as pd 

from preprocessing.config_preproc import PreprocConfig as CFG    

def get_diff_features(df_in : pd.DataFrame, use_features : list,
                      periods : list) -> pd.DataFrame:
    """Maps customer df to diff features for specified raw features and periods
       
    Args:
        df_in (pd.DataFrame): raw input dataframe
        use_features (list): list of features to diff
        periods (list): list of diff periods to apply

    Returns:
        pd.DataFrame: aggregated output features aligned to last time step
    """

    df_in = df_in[['customer_ID'] + use_features]
    dtypes_dict = df_in[use_features].dtypes.to_dict() 
    df_in = df_in.replace(-1, np.nan) # assumes all num features some with -1 nulls
    df_last = df_in.groupby('customer_ID').nth([-1])
    
    df_periods = [] 
    
    for p in periods:
        print(f'Processing period {p}')
        df_periods.append(df_in.groupby('customer_ID').nth([-(p+1)]))

    for i, df_period in enumerate(df_periods):
        df_periods[i] = \
            (df_last - df_period).fillna(-1).astype(dtypes_dict).add_suffix(f'_diff_{periods[i]}') 
    
    del df_last
    gc.collect()
    
    return pd.concat(df_periods, axis=1).reset_index()

# MAKE SURE TO 1BASE THIS
periods = list(range(1,3)) # last-first is captured elsewhere

print('Processing train diffs')
df_train = pd.read_parquet(CFG.train_feature_file)
# df_train = pd.read_parquet(CFG.output_dir + 'train_ex_last1.parquet')
use_features = [c for c in df_train.columns if c not in CFG.cat_features
                and c not in CFG.non_features] 

df_train = get_diff_features(df_train, use_features, periods)
print(df_train.info())
print(df_train.head())

df_train.to_parquet(CFG.output_dir + 'train_diff2_features.parquet')
# df_train.to_parquet(CFG.output_dir + 'train_ex1_diff2_features.parquet')

del df_train
gc.collect()

print('Processing test diffs')
df_test = pd.read_parquet(CFG.test_feature_file)
# df_test = pd.read_parquet(CFG.output_dir + 'test_ex_last1.parquet')
df_test = get_diff_features(df_test, use_features, periods)

df_test.to_parquet(CFG.output_dir + 'test_diff2_features.parquet')
# df_test.to_parquet(CFG.output_dir + 'test_ex1_diff2_features.parquet')

del df_test
gc.collect()