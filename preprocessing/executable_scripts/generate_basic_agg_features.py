import gc
import pandas as pd 

from preprocessing.helpers_preproc import get_agg_features 
from preprocessing.config_preproc import PreprocConfig as CFG    

def get_basic_agg_features(df_in : pd.DataFrame) -> pd.DataFrame:
    """Maps customer df to basic aggregated features including last time step data
       
    Args:
        df_in (pd.DataFrame): raw input dataframe

    Returns:
        pd.DataFrame: aggregated output features aligned to last time step
    """

    num_features = [c for c in df_train.columns if c not in CFG.cat_features
                    and c not in CFG.non_features]

    num_agg_funcs = ['mean','std','min','max','last']
    cat_agg_funcs = ['count', 'last', 'nunique'] 
    
    df_in_num_agg = get_agg_features(df_in=df_in, group_col='customer_ID', 
                                     agg_features=num_features, agg_funcs=num_agg_funcs)\

    df_in_cat_agg = get_agg_features(df_in=df_in, group_col='customer_ID', 
                                     agg_features=cat_features, agg_funcs=cat_agg_funcs)
    
    df_in = pd.merge([df_in_num_agg, df_in_cat_agg], how = 'inner', on = 'customer_ID')

    del df_in_num_agg, df_in_cat_agg
    gc.collect()
    return df_in

print('processing train aggregates')
df_train = pd.read_parquet(CFG.train_feature_file)
df_train = get_basic_agg_features(df_train)
df_train.to_parquet(CFG.output_dir + 'train_basic_agg_features.parquet')

del df_train
gc.collect()

print('processing test aggregates')
df_test = pd.read_parquet(CFG.test_feature_file)
df_test = get_basic_agg_features(df_test)
df_test.to_parquet(CFG.output_dir + 'test_basic_agg_features.parquet')

del df_test
gc.collect()