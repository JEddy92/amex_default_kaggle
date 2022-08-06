import gc
import numpy as np
import pandas as pd 

from preprocessing.helpers_preproc import get_agg_features 
from preprocessing.config_preproc import PreprocConfig as CFG    

# credit to raddar for cluster-finding code
# https://www.kaggle.com/code/raddar/understanding-na-values-in-amex-competition/notebook
def get_na_feature_clusters(df_in : pd.DataFrame) -> np.ndarray:
    """Get array of clusters of sizable number of co-occuring NA columns from
       input dataframe

    Args:
        df_in (pd.DataFrame): dataframe to parse

    Returns:
        np.ndarray: array of cluster feature lists
    """

    df_in[df_in == -1] = np.nan # revert -1 to na encoding as in the original dataset

    cols = sorted(df_in.columns[2:].tolist())
    nas = df_in[cols].isna().sum(axis=0).reset_index(name='NA_count')
    nas['group_count'] = nas.loc[nas.NA_count > 0].groupby('NA_count').transform('count')
    na_clusters = nas.loc[nas.group_count > 10] \
                     .sort_values(['NA_count','index']) \
                     .groupby('NA_count')['index'] \
                     .apply(list).values
    print(na_clusters)
    return na_clusters

def get_na_agg_features(df_in : pd.DataFrame, na_clusters : np.ndarray) -> pd.DataFrame:
    """Maps customer df to aggregated features related to null values
       
    Args:
        df_in (pd.DataFrame): raw input dataframe

    Returns:
        pd.DataFrame: aggregated output features aligned to last time step
    """

    features = [f for f in df_in.columns if f not in CFG.non_features]
    df_in[features] = (df_in[features] == -1) | (df_in[features].isnull()) 
    df_in = df_in.rename(columns={f : f + '_is_null' for f in features}) 
    features = [f + '_is_null' for f in features]
    df_in['na_row_total'] = df_in[features].sum(axis=1) 

    # na_features = ['na_row_total']

    # np.argmin(np.append(a[~np.logical_and.accumulate(~a)], False))

    # for i, cluster in enumerate(na_clusters):
    #     df_in[f'na_clust{i}'] = \
    #         df_in[cluster].isnull().any(axis=1) | (df_in[cluster] == -1).any(axis=1)    
    #     na_features.append(f'na_clust{i}')
    #     print(df_in[f'na_clust{i}'].describe())

    # https://stackoverflow.com/questions/44611125/pandas-count-the-first-consecutive-true-values/44611917#44611917
    def first_True_island_len_IFELSE(a):
        a = a.values
        maxidx = a.argmax()
        pos = a[maxidx:].argmin()
        if a[maxidx]:
            if pos==0:
                return a.size - maxidx
            else:
                return pos
        else:
            return 0

    agg_funcs = ['mean','sum','last', 
                ('consecutive_start' , first_True_island_len_IFELSE)]  
    
    df_in_agg = get_agg_features(df_in=df_in, group_col='customer_ID', 
                                 agg_features=features, agg_funcs=agg_funcs)

    for f in features:
        df_in_agg[f'{f}_at_random_sum'] = df_in_agg[f'{f}_sum'] = df_in_agg[f'{f}_consecutive_start']

    return df_in_agg

print('Processing train aggregates')
df_train = pd.read_parquet(CFG.train_feature_file)

na_clusters = get_na_feature_clusters(df_train)

df_train = get_na_agg_features(df_train, na_clusters)
df_train.to_parquet(CFG.output_dir + 'train_NA_agg_features.parquet')

del df_train
gc.collect()

print('Processing test aggregates')
df_test = pd.read_parquet(CFG.test_feature_file)

df_test = get_na_agg_features(df_test, na_clusters)
df_test.to_parquet(CFG.output_dir + 'test_NA_agg_features.parquet')

del df_test
gc.collect()