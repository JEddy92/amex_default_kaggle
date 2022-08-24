import gc
import numpy as np
import pandas as pd 

from preprocessing.helpers_preproc import get_agg_features 
from preprocessing.config_preproc import PreprocConfig as CFG    

def get_group_rel_transformed(df_in : pd.DataFrame, group_f : str,
                              num_fs : list, time_aggs = ['mean','std','last']) -> pd.DataFrame:
    """Maps customer df to group_f relative-transformed customer for
    num_fs features, and aggregates across time according to time_aggs list
       
    Args:
        df_in (pd.DataFrame): raw input dataframe

    Returns:
        pd.DataFrame: group_f relative-transformed output (aggregated across time)
    """
    df_in = df_in[['customer_ID'] + [group_f] + num_fs]

    df_in = df_in.replace(-1, np.nan) # assumes all num features some with -1 nulls
    grouped = df_in.groupby(group_f) 
    group_means = grouped[num_fs].mean().astype('float32').reset_index()
    group_mean_cols = [f'{f}_{group_f}_mean' for f in num_fs] 
    group_means.columns = [group_f] + group_mean_cols
    print('computed grouped means')
    
    group_stds = grouped[num_fs].std().astype('float32').reset_index()
    group_std_cols = [f'{f}_{group_f}_std' for f in num_fs]
    group_stds.columns = [group_f] + group_std_cols
    print('computed grouped stds')

    df_in['orig_order'] = df_in.index
    df_in = pd.merge(df_in, group_means, on=group_f)
    del group_means
    gc.collect()

    df_in = pd.merge(df_in, group_stds, on=group_f)
    del group_stds
    gc.collect()

    # df_in = df_in.sort_values(by='orig_order').drop(columns=['orig_order'])

    print(f'computing relative features')
    df_in[num_fs] = (df_in[num_fs].values - df_in[group_mean_cols].values) / df_in[group_std_cols].values
    rel_fs = [f'{f}_rel_{group_f}_mean' for f in num_fs]
    df_in = df_in.rename(columns={f : rel_f for f, rel_f in zip(num_fs, rel_fs)})
    df_in = df_in.drop(columns=group_mean_cols + group_std_cols)

    # rel_fs = []
    # for f in num_fs:
    #     print(f'computing relative for {f}')
    #     df_in[f] = (df_in[f] - df_in[f'{f}_{group_f}_mean']) \
    #                / df_in[f'{f}_{group_f}_std'] 
    #     rel_f = f'{f}_rel_{group_f}_mean' 
    #     df_in = df_in.rename(columns={f : rel_f})
    #     rel_fs.append(rel_f)

    # rel_fs = []
    # for f in num_fs:
    #     df_in[f] = (df_in[f] - df_in[group_f].map(group_means[f])) \
    #                / df_in[group_f].map(group_stds[f]) 
    #     rel_f = f'{f}_rel_{group_f}_mean' 
    #     df_in = df_in.rename(columns={f : rel_f})
    #     rel_fs.append(rel_f)

    df_out = get_agg_features(df_in=df_in, group_col='customer_ID',
                              agg_features=rel_fs, agg_funcs=time_aggs)
    return df_out

print('Processing combined aggregates')

df_test = pd.read_parquet(CFG.test_feature_file)
test_cus = list(df_test['customer_ID'].unique())

num_features = [c for c in df_test.columns if c not in CFG.cat_features
                and c not in CFG.non_features]
important_features = ['P_2','B_9','B_1']
df_test['S_2'] = pd.to_datetime(df_test['S_2']).dt.to_period('M')

print(f'Processing month rel for test') 
df_month_rel_test = get_group_rel_transformed(df_in=df_test, group_f='S_2',
                                              num_fs=num_features, 
                                              time_aggs = ['mean','std','last','min','max'])
df_month_rel_test = df_month_rel_test.set_index('customer_ID')
# df_month_rels = get_group_rel_transformed(df_in=df_comb, group_f='S_2',
#                                           num_fs=num_features, 
#                                           time_aggs = ['mean','std','last','min','max'])
# df_months_rel = df_months_rel.set_index('customer_ID')

print(f'Processing month rel for train') 
df_train = pd.read_parquet(CFG.train_feature_file)
train_cus = list(df_train['customer_ID'].unique())
df_train['S_2'] = pd.to_datetime(df_train['S_2']).dt.to_period('M')

df_month_rel_train = get_group_rel_transformed(df_in=df_train, group_f='S_2',
                                               num_fs=num_features, 
                                               time_aggs = ['mean','std','last','min','max'])
df_month_rel_train = df_month_rel_train.set_index('customer_ID')

df_comb = pd.concat([df_train, df_test]).reset_index(drop=True)
del df_train, df_test
gc.collect()

df_cat_rels = []
for cat in CFG.cat_features:
    print(f'Processing cat {cat} rel for important features')
    df_cat_rel = get_group_rel_transformed(df_in=df_comb, group_f=cat,
                                           num_fs=important_features, 
                                           time_aggs = ['mean','std','last','min','max'])
    df_cat_rels.append(df_cat_rel.set_index('customer_ID')) 

df_comb = pd.concat(df_cat_rels, axis=1)
df_train = pd.concat([df_month_rel_train, df_comb[df_comb.index.isin(train_cus)]], axis=1).reset_index()
df_test = pd.concat([df_month_rel_test, df_comb[df_comb.index.isin(test_cus)]], axis=1).reset_index()

del df_comb
gc.collect()

print(df_train.shape, df_train.columns)
print(df_train.info())

print(df_test.shape, df_test.columns)
print(df_test.info())

df_train.to_parquet(CFG.output_dir + 'train_group_rel_agg_features.parquet')

del df_train
gc.collect()

df_test.to_parquet(CFG.output_dir + 'test_group_rel_agg_features.parquet')

del df_test
gc.collect()