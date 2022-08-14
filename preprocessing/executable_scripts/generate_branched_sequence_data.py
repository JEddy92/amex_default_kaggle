import gc
from operator import attrgetter
import numpy as np
import pandas as pd 

from preprocessing.config_preproc import PreprocConfig as CFG    

# MISSING_NULL_OFFSET = .4
# LEADING_NULL_OFFSET = .2

MISSING_NULL_VALUE = -3
LEADING_NULL_VALUE = -2

# trial_cus = ['19bf7c6c1c1419d88486f4b4174fb11fb7fa76089be1bfcefab6cf750498827b',
#              'fffe2bc02423407e33a607660caeed076d713d8a5ad32321530e92704835da88'
#              ]

def gen_branched_sequence_data(df_in : pd.DataFrame, df_cat_enc : pd.DataFrame):
    """Maps customer df to sequential-tensor structured data with
    multiple branches for different feature categories
    (input to sequential neural net models)
       
    Args:
        df_in (pd.DataFrame): raw input dataframe
        df_cat_enc (pd.DataFrame): dataframe of all cat history mapped to encoding
        null_cols (list): list of columns to include null indicators for

    Returns:
        Tuple:
            list of np.ndarray: sequential output arrays
            pd.DataFrame: single column of ordered customers
    """

    df_in[df_in == -1] = np.nan
    df_meds = df_in.median() 

    print(df_in.shape)

    df_in['orig_order'] = df_in.index
    df_in = pd.merge(df_in, df_cat_enc, on=['customer_ID','S_2'])
    df_in = df_in.sort_values(by='orig_order')
    df_in = df_in.drop(columns=CFG.cat_features + ['orig_order'])

    dtypes_dict = df_in.drop(columns=['S_2','customer_ID']).dtypes.to_dict()

    df_cus = df_in[['customer_ID']].drop_duplicates().sort_values(by='customer_ID', ascending=False)

    # 0 is the most sensible NA fill for this specific feature(?) It represents something like
    # no risk or P_2 is not a valid calculation yet?
    print(f"Total P_2 null {df_in['P_2'].isnull().sum()}")
    df_in['P_2'] = df_in['P_2'].fillna(0)
    print(f"Total P_2 null {df_in['P_2'].isnull().sum()}")

    df_in['S_2'] = pd.to_datetime(df_in['S_2'])
    df_in['last_date'] = df_in.groupby('customer_ID')['S_2'].transform('last')
    df_in['months_ago'] = (df_in['last_date'].dt.to_period('M') - df_in['S_2'].dt.to_period('M')) \
                            .apply(attrgetter('n'))
    df_in['months_ago'] = df_in['months_ago'].astype('int')

    print(f'Pre complete months shape: {df_in.shape}')
    df_complete_months = pd.merge(pd.DataFrame({'customer_ID' : df_in['customer_ID'].unique()}), 
                                  pd.DataFrame({'months_ago' : list(range(CFG.max_n_statement))}), 
                                  how='cross')
    df_in = pd.merge(df_in, df_complete_months, on=['customer_ID','months_ago'], 
                     how='outer', indicator=True)
    df_in = df_in.sort_values(by=['customer_ID','months_ago'], ascending=False)
    print(f'Post complete months shape: {df_in.shape}')

    del df_complete_months
    gc.collect()

    df_in['missing_statement'] = df_in['_merge'] == 'right_only'
    print(f"Total missing: {df_in['missing_statement'].sum()}")
    df_in = df_in.drop(columns=['_merge'])

    # print(df_in[['customer_ID','months_ago']].head(10))
    # print(df_in.loc[df_in['customer_ID'].isin(trial_cus), ['customer_ID','months_ago']])

    # Forward fill statement missing values (at random)
    print('Forward filling null values')
    df_in = df_in.groupby('customer_ID') \
                 .ffill()

    print(df_in.columns)

    # Fill remaining null values (missing and leading null values)
    print('Filling remaining (leading) null values')
    missing_mask = df_in['missing_statement'] == 1
    df_in.loc[missing_mask] = MISSING_NULL_VALUE
    df_in.loc[~missing_mask] = df_in.loc[~missing_mask].fillna(df_meds) 
    
    dtypes_dict['missing_statement'] = int
    df_in = df_in.drop(columns=['S_2','last_date']) # keep months ago for position feature
    df_in = df_in.astype(dtypes_dict)

    # make months ago (position embedding) always last feature for convenience
    df_in = df_in[[c for c in df_in.columns if c != 'months_ago'] + ['months_ago']]

    branch_lists = []
    branch_patterns = ['D_','S_','P_','B_','R_']
    for pat in branch_patterns:
        branch_lists.append([c for c in df_in.columns if c[:2] == pat])
    branch_lists.append([c for c in df_in.columns if c[:2] not in branch_patterns])

    branch_names = branch_patterns + ['Other']

    np_return_dict = {}
    for name, cols in zip(branch_names, branch_lists):
        np_return = np.split(df_in[cols].to_numpy(), indices_or_sections=df_cus.shape[0])
        np_return = np.concatenate([np_cus.reshape(1, np_cus.shape[0], np_cus.shape[1]) 
                                    for np_cus in np_return], axis=0)
        np_return_dict[name] = np_return  
        print(np_return.shape)
    
    # print(np_return)
    
    del df_in
    gc.collect()
    print('Np conversion complete')

    return np_return_dict, df_cus

print('Sequence Converting Train')
df_train = pd.read_parquet(CFG.train_feature_file)

# column exclusions
df_train = df_train.drop(columns=['D_103','D_139','B_29'])

df_train_cat_enc = pd.read_parquet(CFG.output_dir + 'train_cat_all_p2_encoded_features.parquet')
# df_train = df_train.loc[df_train['customer_ID'].isin(trial_cus),
#                         ['customer_ID', 'S_2', 'P_2', 'D_39', 'D_115']]

seq_branch_train, seq_train_cus = gen_branched_sequence_data(df_train, df_train_cat_enc)

print([seq_train.shape for seq_train in seq_branch_train.values()])
print(seq_train_cus.shape)

with open(CFG.output_dir + 'seq_branched_train.npy', 'wb') as f:
    np.savez(f, **seq_branch_train)
seq_train_cus.to_parquet(CFG.output_dir + 'seq_branched_train_customers.parquet')   

del df_train, df_train_cat_enc, seq_branch_train, seq_train_cus
gc.collect()

print('Sequence Converting Test 0')
df_test = pd.read_parquet(CFG.test_feature_file)
df_test = df_test.drop(columns=['D_103','D_139','B_29'])
df_test_cat_enc = pd.read_parquet(CFG.output_dir + 'test_cat_all_p2_encoded_features.parquet')
test_cus = df_test['customer_ID'].drop_duplicates()
n_cus = test_cus.shape[0]

cus_chunk_0, cus_chunk_1 = test_cus.iloc[:(n_cus//2)], test_cus.iloc[(n_cus//2):]
df_test = df_test[df_test['customer_ID'].isin(cus_chunk_0)]
gc.collect()

seq_branch_test, seq_test_cus = gen_branched_sequence_data(df_test, df_test_cat_enc)

print([seq_test.shape for seq_test in seq_branch_test.values()])
print(seq_test_cus.shape)

with open(CFG.output_dir + 'seq_branched_test_chunk0.npy', 'wb') as f:
    np.savez(f, **seq_branch_test)

seq_test_cus.to_parquet(CFG.output_dir + 'seq_branched_test_customers_chunk0.parquet')   

del df_test, seq_branch_test, seq_test_cus
gc.collect()

print('Sequence Converting Test 1')
df_test = pd.read_parquet(CFG.test_feature_file)
df_test = df_test.drop(columns=['D_103','D_139','B_29'])
df_test = df_test[df_test['customer_ID'].isin(cus_chunk_1)]
gc.collect()

seq_branch_test, seq_test_cus = gen_branched_sequence_data(df_test, df_test_cat_enc)

print([seq_test.shape for seq_test in seq_branch_test.values()])
print(seq_test_cus.shape)

with open(CFG.output_dir + 'seq_branched_test_chunk1.npy', 'wb') as f:
    np.savez(f, **seq_branch_test)

seq_test_cus.to_parquet(CFG.output_dir + 'seq_branched_test_customers_chunk1.parquet')

del df_test, seq_branch_test, seq_test_cus
gc.collect()