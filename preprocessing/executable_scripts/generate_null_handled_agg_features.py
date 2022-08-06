import gc
import numpy as np
import pandas as pd 

from preprocessing.helpers_preproc import get_agg_features 
from preprocessing.config_preproc import PreprocConfig as CFG    

def get_null_handled_agg_features(df_in : pd.DataFrame) -> pd.DataFrame:
    """Maps customer df to aggregated features as in basic agg features, but with
    null handling on Raddar's dataset encoding and additional null count features
       
    Args:
        df_in (pd.DataFrame): raw input dataframe

    Returns:
        pd.DataFrame: aggregated output features aligned to last time step
    """

    num_features = [c for c in df_in.columns if c not in CFG.cat_features
                    and c not in CFG.non_features]

    num_agg_funcs = ['mean', 'std', 'min', 'max', 'last', 'first']
    cat_agg_funcs = [('mode', lambda x: pd.Series.mode(x).iloc[0]), 'last', 'first', 'nunique'] 
    na_agg_funcs = [('na_count', 'sum')]

    df_in[num_features] = df_in[num_features].replace(-1, np.nan) 

    df_in_na_agg = get_agg_features(df_in=df_in.set_index('customer_ID').isnull().reset_index(),
                                    group_col='customer_ID', agg_features=num_features, agg_funcs=na_agg_funcs)
    print(df_in_na_agg.shape, df_in_na_agg.columns)

    df_in_num_agg = get_agg_features(df_in=df_in, group_col='customer_ID', 
                                     agg_features=num_features, agg_funcs=num_agg_funcs)

    df_in_cat_agg = get_agg_features(df_in=df_in, group_col='customer_ID', 
                                     agg_features=CFG.cat_features, agg_funcs=cat_agg_funcs)
    
    df_in = pd.merge(df_in_num_agg, df_in_cat_agg, how = 'inner', on = 'customer_ID')
    df_in = pd.merge(df_in, df_in_na_agg, how = 'inner', on = 'customer_ID')

    del df_in_num_agg, df_in_cat_agg, df_in_na_agg
    gc.collect()

    for c in num_features:
        df_in[c + '_last_pct_mean'] = df_in[c + '_last'] / df_in[c + '_mean']
        df_in[c + '_diff_mean'] = df_in[c + '_last'] - df_in[c + '_mean']
        df_in[c + '_last_minus_first'] = df_in[c + '_last'] - df_in[c + '_first']

    return df_in

print('Processing train aggregates')
df_train = pd.read_parquet(CFG.train_feature_file)

df_train = get_null_handled_agg_features(df_train)
print(df_train.shape, df_train.columns)
df_train.to_parquet(CFG.output_dir + 'train_null_handled_agg_features.parquet')

del df_train
gc.collect()

print('Processing test aggregates')
df_test = pd.read_parquet(CFG.test_feature_file)

df_test = get_null_handled_agg_features(df_test)
print(df_test.shape)
df_test.to_parquet(CFG.output_dir + 'test_null_handled_agg_features.parquet')

del df_test
gc.collect()