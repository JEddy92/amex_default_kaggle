import gc
import numpy as np
import pandas as pd 

from preprocessing.helpers_preproc import get_agg_features 
from preprocessing.config_preproc import PreprocConfig as CFG    

def get_date_features(df_in : pd.DataFrame) -> pd.DataFrame:
    """Maps customer df with S_2 column to aggregated date-based features
       
    Args:
        df_in (pd.DataFrame): raw input dataframe

    Returns:
        pd.DataFrame: aggregated output features aligned to last time step
    """

    df_in['month'] = pd.to_datetime(df_in['S_2']).dt.month
    df_in['month_gap'] = df_in.groupby('customer_ID')['month'].diff()
    df_in['month_gap'] = df_in['month_gap'] % 12
    df_in.loc[df_in['month_gap'] == 0, 'month_gap'] = 12 # handle case of full year gap
    df_in['month_gap'] = df_in['month_gap'] - 1 # set to 0 base (no gap)

    # count captures total # of statements
    # sum captures total gap in statements not just the mean (relative to # of statements) 
    agg_funcs = ['count', 'sum', 'max', 'std', 'mean', 'last'] 
    
    df_in_agg = get_agg_features(df_in=df_in, group_col='customer_ID', 
                                 agg_features=['month_gap'], agg_funcs=agg_funcs)

    df_in_agg['month_gap_minus_mean'] = df_in_agg['month_gap_last'] - df_in_agg['month_gap_mean']
    df_in_agg = df_in_agg.fillna(-1)

    print(df_in_agg.head(10))
    return df_in_agg

print('Processing train aggregates')
df_train = pd.read_parquet(CFG.train_feature_file, columns=['customer_ID','S_2'])
df_train = get_date_features(df_train)
df_train.to_parquet(CFG.output_dir + 'train_date_features.parquet')

pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', 13)
print(df_train[df_train['customer_ID'] == '026ef3a81feea5de51a09d5796b996a1e3ec306ccd7327dd96d55d8d440203a4'])

del df_train
gc.collect()

print('Processing test aggregates')
df_test = pd.read_parquet(CFG.test_feature_file, columns=['customer_ID','S_2'])
df_test = get_date_features(df_test)
df_test.to_parquet(CFG.output_dir + 'test_date_features.parquet')

del df_test
gc.collect()