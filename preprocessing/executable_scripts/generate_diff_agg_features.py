import gc
import numpy as np
import pandas as pd 

from preprocessing.helpers_preproc import get_agg_features 
from preprocessing.config_preproc import PreprocConfig as CFG    

def get_diff_transformed(df_in : pd.DataFrame) -> pd.DataFrame:
    """Maps customer df to diff-transformed customer df
       
    Args:
        df_in (pd.DataFrame): raw input dataframe

    Returns:
        pd.DataFrame: diff-transformed output
    """

    dtypes_dict = df_in[[c for c in df_in.columns if c != 'customer_ID']].dtypes.to_dict() 
    df_in = df_in.replace(-1, np.nan) # assumes all num features some with -1 nulls
    
    periods = range(CFG.max_n_statement)
    df_periods, df_diffs = [], [] 
    
    for p in periods:
        print(f'Processing period {p}')
        df_periods.append(df_in.groupby('customer_ID').nth([-CFG.max_n_statement+p]))

    for i in range(len(df_periods) - 1):
        df_diffs.append((df_periods[i+1] - df_periods[i]).fillna(-1).astype(dtypes_dict)) 
    
    del df_periods
    gc.collect()
    
    return pd.concat(df_diffs, axis=0).reset_index()

def get_diff_agg_features(df_in : pd.DataFrame) -> pd.DataFrame:
    """Maps customer df to aggregated diff features
       
    Args:
        df_in (pd.DataFrame): raw input dataframe

    Returns:
        pd.DataFrame: aggregated output features
    """

    num_features = [c for c in df_in.columns if c not in CFG.cat_features
                    and c not in CFG.non_features]
    agg_funcs = ['mean', 'std', 'min', 'max']
    
    df_in[num_features] = df_in[num_features].replace(-1, np.nan)
    df_in = get_diff_transformed(df_in[['customer_ID'] + num_features]) 
    print('diff done')
   
    new_cols = [f'{c}_diff1' for c in num_features] 
    df_in.columns = ['customer_ID'] + new_cols

    df_in = get_agg_features(df_in=df_in, group_col='customer_ID', 
                             agg_features=new_cols, agg_funcs=agg_funcs)

    return df_in

print('Processing train aggregates')

# df_train = pd.read_parquet(CFG.train_feature_file)
df_train = pd.read_parquet(CFG.output_dir + 'train_ex_last1.parquet')

df_train = get_diff_agg_features(df_train)
print(df_train.shape, df_train.columns)
print(df_train.info())

# df_train.to_parquet(CFG.output_dir + 'train_diff_agg_features.parquet')
df_train.to_parquet(CFG.output_dir + 'train_ex_last1_diff_agg_features.parquet')

del df_train
gc.collect()

# print('Processing test aggregates')

# df_test = pd.read_parquet(CFG.test_feature_file)

# df_test = get_diff_agg_features(df_test)
# print(df_test.shape)

# df_test.to_parquet(CFG.output_dir + 'test_diff_agg_features.parquet')

# del df_test
# gc.collect()