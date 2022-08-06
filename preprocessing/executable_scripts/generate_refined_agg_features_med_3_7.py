import gc
import numpy as np
import pandas as pd 

from preprocessing.helpers_preproc import get_agg_features 
from preprocessing.config_preproc import PreprocConfig as CFG    

def get_refined_agg_features(df_in : pd.DataFrame, num_agg_features : list,
                             num_last_features : list) -> pd.DataFrame:
    """Maps customer df to aggregated features, with
    null handling on Raddar's dataset encoding, filtering high null count
    features out of numerical aggs, and additional aggs including more recent
    statement aggs
       
    Args:
        df_in (pd.DataFrame): raw input dataframe
        num_agg_features (list): features to include in aggregations
        num_last_features (list): features to take only last on

    Returns:
        pd.DataFrame: aggregated output features aligned to last time step
    """

    num_agg_funcs = ['median', 'std', 'min', 'max', 'last', 'first']
    num_agg_rec_funcs = ['median', 'std']
    cat_agg_funcs = ['last', 'first', 'nunique'] 

    df_in[num_agg_features] = df_in[num_agg_features].replace(-1, np.nan) 

    print('numeric aggs')
    df_in_num_agg = get_agg_features(df_in=df_in, group_col='customer_ID', 
                                     agg_features=num_agg_features, agg_funcs=num_agg_funcs)

    print('numeric last')
    df_in_num_last = get_agg_features(df_in=df_in, group_col='customer_ID', 
                                      agg_features=num_last_features, agg_funcs=['last'])

    print('numeric aggs - last 3')
    df_in_num_agg_rec3 = get_agg_features(df_in=df_in.groupby('customer_ID').tail(3)
                                                    .add_suffix('_3sts').rename(columns={'customer_ID_3sts' : 'customer_ID'}), 
                                         group_col='customer_ID', 
                                         agg_features=[f'{f}_3sts' for f in num_agg_features], 
                                         agg_funcs=num_agg_rec_funcs)

    print('numeric aggs - last 7')
    df_in_num_agg_rec7 = get_agg_features(df_in=df_in.groupby('customer_ID').tail(7)
                                                    .add_suffix('_7sts').rename(columns={'customer_ID_7sts' : 'customer_ID'}), 
                                          group_col='customer_ID', 
                                          agg_features=[f'{f}_7sts' for f in num_agg_features], 
                                          agg_funcs=num_agg_rec_funcs)

    print('cat aggs')
    df_in_cat_agg = get_agg_features(df_in=df_in, group_col='customer_ID', 
                                     agg_features=CFG.cat_features, agg_funcs=cat_agg_funcs)
    
    df_in = pd.merge(df_in_num_agg, df_in_num_last, how = 'inner', on = 'customer_ID')
    df_in = pd.merge(df_in, df_in_num_agg_rec3, how = 'inner', on = 'customer_ID')
    df_in = pd.merge(df_in, df_in_num_agg_rec7, how = 'inner', on = 'customer_ID')
    df_in = pd.merge(df_in, df_in_cat_agg, how = 'inner', on = 'customer_ID')

    del df_in_num_agg, df_in_num_agg_rec3, df_in_num_agg_rec7, df_in_cat_agg
    gc.collect()

    for c in num_agg_features:
        df_in[c + '_last_minus_median'] = df_in[c + '_last'] - df_in[c + '_median']
        df_in[c + '_last_minus_first'] = df_in[c + '_last'] - df_in[c + '_first']

        df_in[c + '_last_minus_3sts_median'] = df_in[c + '_last'] - df_in[c + '_3sts_median']
        df_in[c + '_last_minus_7sts_median'] = df_in[c + '_last'] - df_in[c + '_7sts_median']
        df_in[c + '_3sts_median_minus_median'] = df_in[c + '_3sts_median'] - df_in[c + '_median']
        df_in[c + '_3sts_median_minus_7sts_median'] = df_in[c + '_3sts_median'] - df_in[c + '_7sts_median']

    return df_in

print('Processing train aggregates')
df_train = pd.read_parquet(CFG.train_feature_file)

num_features = [c for c in df_train.columns if c not in CFG.cat_features
                and c not in CFG.non_features]

df_train[num_features] = df_train[num_features].replace(-1, np.nan)

na_col_counts = df_train[num_features].isnull().sum() / df_train.shape[0] 
high_na_col_mask = (na_col_counts >= .90)
print(f'There are {high_na_col_mask.sum()} >= 90 NA features')

num_agg_features = na_col_counts[~high_na_col_mask].index
num_last_features = na_col_counts[high_na_col_mask].index

df_train = get_refined_agg_features(df_train, num_agg_features, num_last_features)
print(df_train.shape, df_train.columns)
df_train.to_parquet(CFG.output_dir + 'train_refined_agg_features_med_3_7.parquet')

del df_train
gc.collect()

print('Processing test aggregates')
df_test = pd.read_parquet(CFG.test_feature_file)

df_test = get_refined_agg_features(df_test, num_agg_features, num_last_features)
print(df_test.shape)
df_test.to_parquet(CFG.output_dir + 'test_refined_agg_features_med_3_7.parquet')

del df_test
gc.collect()