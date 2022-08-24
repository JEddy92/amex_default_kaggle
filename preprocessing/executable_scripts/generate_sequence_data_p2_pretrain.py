import gc
from operator import attrgetter
import json
import numpy as np
import pandas as pd 

from preprocessing.config_preproc import PreprocConfig as CFG    

# TO DO: PUT IN CFG
MISSING_NULL_VALUE = -3 #-3

def gen_sequence_data(df_in : pd.DataFrame, df_cat_enc : pd.DataFrame):
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

    df_in['orig_order'] = df_in.index
    df_in = pd.merge(df_in, df_cat_enc, on=['customer_ID','S_2'])
    df_in = df_in.sort_values(by='orig_order').drop(columns=['orig_order'] + CFG.cat_features)
    dtypes_dict = df_in.drop(columns=['S_2','customer_ID']).dtypes.to_dict()

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

    print(f'Pre complete months shape: {df_in.shape}')
    df_complete_months = pd.merge(pd.DataFrame({'customer_ID' : df_in['customer_ID'].unique()}), 
                                  pd.DataFrame({'months_ago' : list(range(CFG.max_n_statement))}), 
                                  how='cross')
    print(df_complete_months.shape)

    df_in = pd.merge(df_in, df_complete_months, on=['customer_ID','months_ago'], 
                     how='outer', indicator=True).reset_index(drop=True)
    df_in = df_in.sort_values(by=['customer_ID','months_ago'], ascending=False)
    print(f'Post complete months shape: {df_in.shape}')

    del df_complete_months
    gc.collect()

    max_months_ago = df_in[(df_in['_merge'] != 'right_only')].groupby('customer_ID')['months_ago'].max()
    df_in['months_ago_valid_max'] = df_in['customer_ID'].map(max_months_ago).astype(int)

    # binary flags for both gap (missing) statements
    # and leading pre-history statements
    df_in['missing_statement'] = (df_in['_merge'] == 'right_only') & \
                                 (df_in['months_ago'] < df_in['months_ago_valid_max'])
    df_in['missing_statement'] = df_in['missing_statement'].astype(int)                            
    df_in['pre_history'] = (df_in['_merge'] == 'right_only') & \
                           (df_in['months_ago'] > df_in['months_ago_valid_max'])
    df_in['pre_history'] = df_in['pre_history'].astype(int) 
    print(f"Total missing: {df_in['missing_statement'].sum()}")
    print(f"Total pre-history: {df_in['pre_history'].sum()}")
    df_in = df_in.drop(columns=['_merge'])

    print(df_in.columns)

    # Setting customer-specific position feature
    df_in['customer_sequence_num'] = (df_in['months_ago_valid_max'] - df_in['months_ago']).astype(int)

    # Fill missing null value into missing statements
    print('Filling pre-history and missing statement null values')
    missing_mask = (df_in['missing_statement'] == 1) | (df_in['pre_history'] == 1) 
    indicator_cols = ['months_ago_valid_max', 'customer_sequence_num', 
                      'pre_history','missing_statement','months_ago']
    df_in.loc[missing_mask, [c for c in df_in.columns if c not in indicator_cols]] = MISSING_NULL_VALUE
    
    print(df_in[['S_2'] + indicator_cols].head(10))
    
    # split out actual feature df, 
    # keep months ago and missing statement indicator
    df_in = df_in.drop(columns=['customer_ID','S_2','last_date']) 
    for c in indicator_cols:
        dtypes_dict[c] = int
    print(df_in.info())
    df_in = df_in.astype(dtypes_dict)

    # make position/statement indicator cols last features for convenience
    # and place P_2 first for convenience
    df_in = df_in[[c for c in df_in.columns if c not in indicator_cols] + 
                  indicator_cols]
    df_in = df_in[['P_2'] + [c for c in df_in.columns if c != 'P_2']]
    print(df_in.columns[:5], df_in.columns[-5:])

    np_return = np.split(df_in.to_numpy(), indices_or_sections=df_cus.shape[0])
    np_return = np.concatenate([np_cus.reshape(1, np_cus.shape[0], np_cus.shape[1]) 
                               for np_cus in np_return], axis=0)
    print(np_return.shape)
    
    del df_in
    gc.collect()
    print('Np conversion complete')

    return np_return, df_cus

print('Sequence Converting Train')

# remove redundant and dif distribution cols
excl_cols = ['D_103','D_139','B_29']

df_train = pd.read_parquet(CFG.output_dir + 'train_lgb_400_imputed.parquet')
df_train = df_train.drop(columns=excl_cols)
df_train_cat_enc = pd.read_parquet(CFG.output_dir + 'train_cat_all_p2_encoded_features.parquet')

seq_train, seq_train_cus = gen_sequence_data(df_train, df_train_cat_enc)
print(seq_train.shape)
print(seq_train_cus.shape)

with open(CFG.output_dir + 'seq_capped_train.npy', 'wb') as f:
    np.save(f, seq_train)
seq_train_cus.to_parquet(CFG.output_dir + 'seq_capped_train_customers.parquet')   

del df_train, df_train_cat_enc, seq_train, seq_train_cus
gc.collect()

print('Sequence Converting Test 0')

df_test = pd.read_parquet(CFG.output_dir + 'test_lgb_400_imputed.parquet').reset_index(drop=True)
df_test = df_test.drop(columns=excl_cols)
df_test_cat_enc = pd.read_parquet(CFG.output_dir + 'test_cat_all_p2_encoded_features.parquet')
test_cus = df_test['customer_ID'].drop_duplicates()
n_cus = test_cus.shape[0]

cus_chunk_0, cus_chunk_1 = test_cus.iloc[:(n_cus//2)], test_cus.iloc[(n_cus//2):]
chunk_0_idx = df_test['customer_ID'].isin(cus_chunk_0)
chunk_1_idx = df_test['customer_ID'].isin(cus_chunk_1)

df_test = df_test[chunk_0_idx]
seq_test, seq_test_cus = gen_sequence_data(df_test, df_test_cat_enc)

with open(CFG.output_dir + 'seq_capped_test_chunk0.npy', 'wb') as f:
    np.save(f, seq_test)
seq_test_cus.to_parquet(CFG.output_dir + 'seq_capped_test_customers_chunk0.parquet')   
print(seq_test.shape)
print(seq_test_cus.shape)

del df_test, seq_test, seq_test_cus
gc.collect()

print('Sequence Converting Test 1')
df_test = pd.read_parquet(CFG.output_dir + 'test_lgb_400_imputed.parquet').reset_index(drop=True)
df_test = df_test.drop(columns=excl_cols)
df_test = df_test[chunk_1_idx]
gc.collect()

seq_test, seq_test_cus = gen_sequence_data(df_test, df_test_cat_enc)

with open(CFG.output_dir + 'seq_capped_test_chunk1.npy', 'wb') as f:
    np.save(f, seq_test)
seq_test_cus.to_parquet(CFG.output_dir + 'seq_capped_test_customers_chunk1.parquet')   
print(seq_test.shape)
print(seq_test_cus.shape)

del df_test, seq_test, seq_test_cus, df_test_cat_enc
gc.collect()