import gc
from operator import attrgetter
import json
import numpy as np
import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder

from preprocessing.config_preproc import PreprocConfig as CFG    

# TO DO: PUT IN CFG
MISSING_NULL_VALUE = -3 #-3
MISSING_CAT_VALUE = 0

def gen_sequence_data(df_in : pd.DataFrame, df_cat_enc : pd.DataFrame, 
                      df_nulls : pd.DataFrame):
    """Maps customer df to sequential-tensor structured data
    (input to sequential neural net models). 
    
    Assumes fully na imputed input data.
       
    Args:
        df_in (pd.DataFrame): raw input dataframe
        df_cat_enc (pd.DataFrame): dataframe of all cat history mapped to encoding

    Returns:
        Tuple:
            np.ndarray: sequential output array (actual features)
            np.ndarray: sequential output array (isnull features)
            pd.DataFrame: single column of ordered customers
    """

    dtypes_dict = df_in.drop(columns=['S_2','customer_ID']).dtypes.to_dict()

    df_in = pd.concat([df_in, df_nulls], axis=1)
    null_cols = [c for c in df_in.columns if 'is_null' in c]
    null_dtypes_dict = df_in[null_cols].dtypes.to_dict()

    df_cus = df_in[['customer_ID']].drop_duplicates().sort_values(by='customer_ID', ascending=False)

    # Thanks Amex!
    # https://arxiv.org/pdf/2012.15330.pdf
    with open(CFG.output_dir + '/lgb_capping_feature_splits.json') as json_file:
        lgb_feature_splits = json.load(json_file)
    
    for f, splits in lgb_feature_splits.items():
        if f in df_in.columns:
            print(f'{f} min, max: {df_in[f].min()}, {df_in[f].max()}')
            
            min_clip, max_clip = 2 * splits[0] - splits[1], 2 * splits[-1] - splits[-2] 
            df_in[f] = df_in[f].clip(lower=min_clip, upper=max_clip)

            print(f'{f} post clip min, max: {df_in[f].min()}, {df_in[f].max()}')

    df_in['S_2'] = pd.to_datetime(df_in['S_2'])
    df_in['last_date'] = df_in.groupby('customer_ID')['S_2'].transform('last')
    df_in['months_ago'] = (df_in['last_date'].dt.to_period('M') - df_in['S_2'].dt.to_period('M')) \
                            .apply(attrgetter('n'))
    df_in['months_ago'] = df_in['months_ago'].astype('int')
    df_in['days_ago'] = (df_in['last_date'] - df_in['S_2']).dt.days
    df_in['days_ago'] = df_in['days_ago'].astype('int') 

    print(df_in[['S_2','months_ago','days_ago']].head())

    print(f'Pre complete months shape: {df_in.shape}')
    df_complete_months = pd.merge(pd.DataFrame({'customer_ID' : df_in['customer_ID'].unique()}), 
                                  pd.DataFrame({'months_ago' : list(range(CFG.max_n_statement))}), 
                                  how='cross')
    # df_complete_months['days_ago'] = 30.437 * df_complete_months['months_ago'] #think uneeded 
    df_in = pd.merge(df_in, df_complete_months, on=['customer_ID','months_ago'], 
                     how='outer', indicator=True)
    df_in = df_in.sort_values(by=['customer_ID','months_ago'], ascending=False)
    print(f'Post complete months shape: {df_in.shape}')

    del df_complete_months
    gc.collect()

    df_in['missing_statement'] = df_in['_merge'] == 'right_only'
    print(f"Total missing: {df_in['missing_statement'].sum()}")
    df_in = df_in.drop(columns=['_merge'])

    print(df_in.columns)

    # Fill missing null value into missing statements
    print('Filling missing statement null values')
    missing_mask = df_in['missing_statement'] == 1
    df_in.loc[missing_mask, 
              [c for c in df_in.columns if c not in (CFG.cat_features + ['missing_statement','months_ago'])]] = MISSING_NULL_VALUE
    df_in.loc[missing_mask, CFG.cat_features] = MISSING_CAT_VALUE 
    
    # split out null df
    df_nulls = df_in[null_cols].astype(null_dtypes_dict)
    
    # split out actual feature df, 
    # keep months ago and missing statement indicator
    df_in = df_in.drop(columns=null_cols + ['customer_ID','S_2','last_date']) 
    dtypes_dict['missing_statement'] = int
    df_in = df_in.astype(dtypes_dict)

    # make days ago, missing statement, and months ago (position embedding) last features
    # make categoricals first features
    df_in = df_in[[c for c in df_in.columns if c not in ['days_ago','missing_statement','months_ago']] + 
                  ['days_ago', 'missing_statement','months_ago']]
    df_in = df_in[CFG.cat_features + [f for f in df_in.columns if f not in CFG.cat_features]]

    np_return = np.split(df_in.to_numpy(), indices_or_sections=df_cus.shape[0])
    np_return = np.concatenate([np_cus.reshape(1, np_cus.shape[0], np_cus.shape[1]) 
                                for np_cus in np_return], axis=0)
    print(np_return.shape)

    np_nas_return = np.split(df_nulls.to_numpy(), indices_or_sections=df_cus.shape[0])
    np_nas_return = np.concatenate([np_cus.reshape(1, np_cus.shape[0], np_cus.shape[1]) 
                                    for np_cus in np_nas_return], axis=0)
    print(np_nas_return.shape)
    
    del df_in, df_nulls
    gc.collect()
    print('Np conversion complete')

    return np_return, np_nas_return, df_cus

print('Sequence Converting Train')
NULL_THRESH = .005 # cutoff of pct of train data null to include in null trackers

# credit to Elias for feature exclusion ideas
# https://www.kaggle.com/code/gehallak/amex-correlation/notebook
# column exclusions
# drop S_9 too as feature with big train/test difference? B_39?
# redundant_cols = ['B_1', 'B_11', 'B_23', 'B_37', 'D_103', 'D_118', 
#                   'D_137', 'D_139', 'D_141', 'D_74', 'D_77', 'S_24']
# low_cor_cols = ['B_15', 'B_27', 'D_106', 'D_109', 'D_144', 
#                  'D_69', 'D_73', 'R_18', 'R_23', 'R_28', 'S_12', 'S_18', 'S_19']
other_cols = ['D_103','D_139','B_29']
# keep low cor cols but remove redundant ones
excl_cols = other_cols

df_train_raw = pd.read_parquet(CFG.train_feature_file).drop(columns=excl_cols)
df_train_raw[df_train_raw == -1] = np.nan

df_train_nulls = df_train_raw.isnull()
null_cols = [c for c in df_train_nulls.columns if df_train_nulls[c].sum() > (df_train_nulls.shape[0] * NULL_THRESH)]
df_train_nulls = df_train_nulls[null_cols].add_suffix('_is_null')
df_train_nulls = df_train_nulls.astype(bool)

# df_train_nulls['customer_ID'] = df_train_raw['customer_ID']
print(df_train_nulls.columns)
print(df_train_nulls.shape)

del df_train_raw
gc.collect()

df_train = pd.read_parquet(CFG.output_dir + 'train_lgb_400_imputed.parquet')
df_train = df_train.drop(columns=excl_cols)
df_train_cat_enc = pd.read_parquet(CFG.output_dir + 'train_cat_all_p2_encoded_features.parquet')

print(df_train[CFG.cat_features].isnull().sum())
df_train[CFG.cat_features] = df_train[CFG.cat_features].fillna(999)

encoder = OrdinalEncoder()
df_train[CFG.cat_features] = encoder.fit_transform(df_train[CFG.cat_features]).astype(int)
df_train[CFG.cat_features] += 1 # offset by 1 to use 0 as missing cat value
print(df_train[CFG.cat_features])

# print out data on cat feature cardinalities
for c in CFG.cat_features:
    print(f'{df_train[c].nunique()} values for {c}')

seq_train, seq_nas_train, seq_train_cus = gen_sequence_data(df_train, df_train_cat_enc, df_train_nulls)
print(seq_train.shape)
print(seq_train_cus.shape)

with open(CFG.output_dir + 'seq_capped_train.npy', 'wb') as f:
    np.save(f, seq_train)
with open(CFG.output_dir + 'seq_capped_nas_train.npy', 'wb') as f:
    np.save(f, seq_nas_train)
seq_train_cus.to_parquet(CFG.output_dir + 'seq_capped_train_customers.parquet')   

del df_train, df_train_nulls, df_train_cat_enc, seq_train, seq_nas_train, seq_train_cus
gc.collect()

print('Sequence Converting Test 0')
df_test_raw = pd.read_parquet(CFG.test_feature_file)
df_test_raw[df_test_raw == -1] = np.nan

df_test_nulls = df_test_raw[null_cols].isnull().add_suffix('_is_null')
df_test_nulls = df_test_nulls.astype(bool)
# df_test_nulls['customer_ID'] = df_test_raw['customer_ID']
print(df_test_nulls.columns)
print(df_test_nulls.shape)

del df_test_raw
gc.collect()

df_test = pd.read_parquet(CFG.output_dir + 'test_lgb_400_imputed.parquet').reset_index(drop=True)
df_test = df_test.drop(columns=excl_cols)
df_test_cat_enc = pd.read_parquet(CFG.output_dir + 'test_cat_all_p2_encoded_features.parquet')
test_cus = df_test['customer_ID'].drop_duplicates()
n_cus = test_cus.shape[0]

cus_chunk_0, cus_chunk_1 = test_cus.iloc[:(n_cus//2)], test_cus.iloc[(n_cus//2):]
chunk_0_idx = df_test['customer_ID'].isin(cus_chunk_0)
chunk_1_idx = df_test['customer_ID'].isin(cus_chunk_1)

print(df_test.index, df_test_nulls.index)
df_test = df_test[chunk_0_idx]
df_test_nulls_chunk_0 = df_test_nulls[chunk_0_idx]
df_test_nulls_chunk_1 = df_test_nulls[chunk_1_idx]

del df_test_nulls
gc.collect()

df_test[CFG.cat_features] = df_test[CFG.cat_features].fillna(999)
df_test[CFG.cat_features] = encoder.transform(df_test[CFG.cat_features]).astype(int)
df_test[CFG.cat_features] += 1

seq_test, seq_nas_test, seq_test_cus = gen_sequence_data(df_test, df_test_cat_enc, df_test_nulls_chunk_0)

with open(CFG.output_dir + 'seq_capped_test_chunk0.npy', 'wb') as f:
    np.save(f, seq_test)
with open(CFG.output_dir + 'seq_capped_nas_test_chunk0.npy', 'wb') as f:
    np.save(f, seq_nas_test)
seq_test_cus.to_parquet(CFG.output_dir + 'seq_capped_test_customers_chunk0.parquet')   
print(seq_test.shape)
print(seq_test_cus.shape)

del df_test, df_test_nulls_chunk_0, seq_test, seq_nas_test, seq_test_cus
gc.collect()

print('Sequence Converting Test 1')
df_test = pd.read_parquet(CFG.output_dir + 'test_lgb_400_imputed.parquet').reset_index(drop=True)
df_test = df_test.drop(columns=excl_cols)
df_test = df_test[chunk_1_idx]
gc.collect()

df_test[CFG.cat_features] = df_test[CFG.cat_features].fillna(999)
df_test[CFG.cat_features] = encoder.transform(df_test[CFG.cat_features]).astype(int)
df_test[CFG.cat_features] += 1

seq_test, seq_nas_test, seq_test_cus = gen_sequence_data(df_test, df_test_cat_enc, df_test_nulls_chunk_1)

with open(CFG.output_dir + 'seq_capped_test_chunk1.npy', 'wb') as f:
    np.save(f, seq_test)
with open(CFG.output_dir + 'seq_capped_nas_test_chunk1.npy', 'wb') as f:
    np.save(f, seq_nas_test)
seq_test_cus.to_parquet(CFG.output_dir + 'seq_capped_test_customers_chunk1.parquet')   
print(seq_test.shape)
print(seq_test_cus.shape)

del df_test, df_test_nulls_chunk_1, seq_test, seq_nas_test, seq_test_cus
gc.collect()