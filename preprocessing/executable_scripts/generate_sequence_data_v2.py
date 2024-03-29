import gc
from operator import attrgetter
import numpy as np
import pandas as pd 

from preprocessing.config_preproc import PreprocConfig as CFG    

# MISSING_NULL_OFFSET = .4
# LEADING_NULL_OFFSET = .2

# TO DO: PUT IN CFG
MISSING_NULL_VALUE = -3
LEADING_NULL_VALUE = -2

# trial_cus = ['19bf7c6c1c1419d88486f4b4174fb11fb7fa76089be1bfcefab6cf750498827b',
#              'fffe2bc02423407e33a607660caeed076d713d8a5ad32321530e92704835da88'
#              ]

def gen_sequence_data(df_in : pd.DataFrame, df_cat_enc : pd.DataFrame):
    """Maps customer df to sequential-tensor structured data
    (input to sequential neural net models)
       
    Args:
        df_in (pd.DataFrame): raw input dataframe
        df_cat_enc (pd.DataFrame): dataframe of all cat history mapped to encoding
        null_cols (list): list of columns to include null indicators for

    Returns:
        Tuple:
            np.ndarray: sequential output array
            pd.DataFrame: single column of ordered customers
    """

    print(df_in.isnull().sum())
    print(df_in.isnull().sum().sum())
    # df_in[df_in == -1] = np.nan
    # df_meds = df_in.median() 
    
    # na_flags = [c + '_is_null' for c in null_colls] 
    # df_in[na_flags] = df_in[null_cols].isnull().astype('int')

    print(df_in.shape)

    df_in['orig_order'] = df_in.index
    df_in = pd.merge(df_in, df_cat_enc, on=['customer_ID','S_2'])
    df_in = df_in.sort_values(by='orig_order')
    df_in = df_in.drop(columns=CFG.cat_features + ['orig_order'])
    print(df_in.isnull().sum().sum())

    dtypes_dict = df_in.drop(columns=['S_2','customer_ID']).dtypes.to_dict()

    df_cus = df_in[['customer_ID']].drop_duplicates().sort_values(by='customer_ID', ascending=False)

    # 0 is the most sensible NA fill for this specific feature(?) It represents something like
    # no risk or P_2 is not a valid calculation yet?
    # print(f"Total P_2 null {df_in['P_2'].isnull().sum()}")
    # df_in['P_2'] = df_in['P_2'].fillna(0)
    # print(f"Total P_2 null {df_in['P_2'].isnull().sum()}")

    # Ryota's clipping code
    def clipping(df):
        df.loc[df['B_1'] < -0.5, 'B_1'] = -0.5
        df.loc[df['D_39'] > 200, 'D_39'] = 200
        df.loc[df['B_6'] > 200, 'B_6'] = 200
        df.loc[df['R_5'] > 50, 'R_5'] = 50
        df.loc[df['D_41'] > 8, 'D_41'] = 8
        df.loc[df['D_46'] > 6, 'D_46'] = 6
        df.loc[df['D_46'] < -5, 'D_46'] = -5
        df.loc[df['D_49'] > 500, 'D_49'] = 500
        df.loc[df['B_7'] < -1, 'B_7'] = -1
        df.loc[df['D_50'] < -5, 'D_50'] = -5
        df.loc[df['D_50'] > 150, 'D_50'] = 150
        df.loc[df['B_9'] > 20, 'B_9'] = 20
        df.loc[df['R_3'] > 70, 'R_3'] = 70
        df.loc[df['S_5'] > 100, 'S_5'] = 100
        df.loc[df['B_12'] > 100, 'B_12'] = 100
        df.loc[df['D_55'] > 2, 'D_55'] = 2
        df.loc[df['B_14'] > 30, 'B_14'] = 30
        df.loc[df['B_14'] < -2, 'B_14'] = -2
        df.loc[df['D_61'] > 50, 'D_61'] = 50
        df.loc[df['B_15'] < -1, 'B_15'] = -1
        df.loc[df['D_65'] > 2000, 'D_65'] = 2000
        df.loc[df['S_12'] > 50, 'S_12'] = 50
        df.loc[df['B_21'] > 100, 'B_21'] = 100
        df.loc[df['D_69'] > 1000, 'D_69'] = 1000
        df.loc[df['D_74'] > 50, 'D_74'] = 50
        df.loc[df['D_75'] > 50, 'D_75'] = 50
        df.loc[df['D_76'] > 15, 'D_75'] = 15
        df.loc[df['B_24'] > 30, 'B_24'] = 30
        df.loc[df['R_7'] > 300, 'R_7'] = 300
        df.loc[df['B_25'] < -1, 'B_25'] = -1
        df.loc[df['B_25'] > 10, 'B_25'] = 10
        df.loc[df['B_26'] > 500, 'B_26'] = 500
        df.loc[df['D_79'] > 20, 'D_79'] = 20
        df.loc[df['R_8'] > 25, 'R_8'] = 25
        df.loc[df['R_9']==0, 'R_9'] = 1
        df.loc[df['S_16'] > 150, 'S_16'] = 150
        df.loc[df['R_10'] > 10, 'R_10'] = 10
        df.loc[df['R_11'] > 10, 'R_11'] = 10
        df.loc[df['B_27'] > 0.4, 'B_27'] = 0.4
        df.loc[df['D_82'] > 6, 'D_82'] = 6
        df.loc[df['B_28'] > 10, 'B_28'] = 10
        df.loc[df['R_13'] > 20, 'R_13'] = 20
        df.loc[df['D_83'] > 20, 'D_83'] = 20
        df.loc[df['R_14'] > 2000, 'R_14'] = 2000
        df.loc[df['D_84'] > 35, 'D_84'] = 35
        df.loc[df['R_16'] > 25, 'R_16'] = 25
        df.loc[df['R_17'] > 20, 'R_17'] = 20
        df.loc[df['R_18'] > 15, 'R_18'] = 15
        df.loc[df['S_19'] > 0.4, 'S_19'] = 0.4
        df.loc[df['R_20'] > 40, 'R_20'] = 40
        df.loc[df['S_23'] < -50, 'S_23'] = -50
        df.loc[df['S_23'] > 50, 'S_23'] = 50
        df.loc[df['S_25'] > 4, 'S_25'] = 4
        df.loc[df['S_26'] > 100, 'S_26'] = 100
        df.loc[df['D_105'] > 4.5, 'D_105'] = 4.5
        df.loc[df['D_106'] > 3000, 'D_106'] = 3000
        df.loc[df['D_107'] > 15, 'D_107'] = 15
        df.loc[df['B_36'] > 0.8, 'B_36'] = 0.8
        df.loc[df['R_26'] > 250, 'R_26'] = 250
        df.loc[df['D_108'] > 8, 'D_108'] = 8
        df.loc[df['D_111'] > 2, 'D_111'] = 2
        df.loc[df['B_39'] < -0.1, 'B_39'] = -0.1
        df.loc[df['B_39'] > 2, 'B_39'] = 2
        df.loc[df['B_40'] > 1000, 'B_40'] = 1000
        df.loc[df['D_113'] > 20, 'D_113'] = 20
        df.loc[df['D_122'] > 13, 'D_122'] = 13
        df.loc[df['D_123'] > 50, 'D_123'] = 50
        df.loc[df['D_124'] > 60, 'D_124'] = 60
        df.loc[df['D_125'] > 10, 'D_125'] = 10
        df.loc[df['B_41'] > 20, 'B_41'] = 20
        df.loc[df['B_42'] > 7, 'B_42'] = 7
        df.loc[df['D_132'] > 5, 'D_132'] = 5
        df.loc[df['D_145'] > 45, 'D_145'] = 45
        df.loc[df['B_37'] < -2, 'B_37'] = -2
        df.loc[df['D_59'] > 100, 'D_59'] = 100
        df.loc[df['S_15'] > 60, 'S_15'] = 60

        return df

    # df_in = clipping(df_in)

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

    # df_in_mins, df_in_stds = df_in.min(), df_in.std()

    # Forward fill statement missing values (at random)
    # print('Forward filling null values')
    # df_in = df_in.groupby('customer_ID') \
    #              .ffill()
                #.apply(lambda grp: grp.fillna(method='ffill'))

    print(df_in.columns)

    # Fill missing null value into missing statements
    print('Filling missing statement null values')
    missing_mask = df_in['missing_statement'] == 1
    df_in.loc[missing_mask, 
              [c for c in df_in.columns if c not in ['missing_statement','months_ago']]] = MISSING_NULL_VALUE
    print(df_in[missing_mask].head(5))
    print(df_in[missing_mask][['missing_statement','months_ago']].head(5))
    # df_in.loc[~missing_mask] = df_in.loc[~missing_mask].fillna(df_meds) 
    
    # keep months ago and missing statement indicator
    df_in = df_in.drop(columns=['customer_ID','S_2','last_date']) 
    dtypes_dict['missing_statement'] = int
    df_in = df_in.astype(dtypes_dict)

    # make months ago (position embedding) always last feature for convenience
    df_in = df_in[[c for c in df_in.columns if c != 'months_ago'] + ['months_ago']]

    # # Rearrange column order to put categories first
    # cat_features = CFG.cat_features
    # num_features = [f for f in df_in.columns if f not in cat_features]
    # df_in = df_in[cat_features + num_features]

    np_return = np.split(df_in.to_numpy(), indices_or_sections=df_cus.shape[0])
    np_return = np.concatenate([np_cus.reshape(1, np_cus.shape[0], np_cus.shape[1]) 
                               for np_cus in np_return], axis=0)
    print(np_return.shape)
    # print(np_return)
    
    del df_in
    gc.collect()
    print('Np conversion complete')

    return np_return, df_cus

print('Sequence Converting Train')
df_train = pd.read_parquet(CFG.output_dir + 'train_lgb_no_lag_imputed.parquet')

# credit to Elias for feature exclusion ideas
# https://www.kaggle.com/code/gehallak/amex-correlation/notebook
# column exclusions
# redundant_cols = ['B_1', 'B_11', 'B_23', 'B_37', 'D_103', 'D_118', 
#                   'D_137', 'D_139', 'D_141', 'D_74', 'D_77', 'S_24']
# low_cor_cols = ['B_15', 'B_27', 'D_106', 'D_109', 'D_144', 
#                  'D_69', 'D_73', 'R_18', 'R_23', 'R_28', 'S_12', 'S_18', 'S_19']
other_cols = ['D_103','D_139','B_29']
excl_cols = other_cols  

df_train = df_train.drop(columns=excl_cols)

df_train_cat_enc = pd.read_parquet(CFG.output_dir + 'train_cat_all_p2_encoded_features.parquet')
# df_train = df_train.loc[df_train['customer_ID'].isin(trial_cus),
#                         ['customer_ID', 'S_2', 'P_2', 'D_39', 'D_115']]

# if you use this, this is incomplete because it's pre -1 -> np.nan conversion
# df_nulls = df_train.isnull() 
# null_colls = [c for c in df_nulls.columns if df_nulls[c].sum() > 0]

# del df_nulls
# gc.collect()

seq_train, seq_train_cus = gen_sequence_data(df_train, df_train_cat_enc)
print(seq_train.shape)
print(seq_train_cus.shape)

with open(CFG.output_dir + 'seq_v2_train.npy', 'wb') as f:
    np.save(f, seq_train)
seq_train_cus.to_parquet(CFG.output_dir + 'seq_v2_train_customers.parquet')   

del df_train, df_train_cat_enc, seq_train, seq_train_cus
gc.collect()

print('Sequence Converting Test 0')
df_test = pd.read_parquet(CFG.output_dir + 'test_lgb_no_lag_imputed.parquet')
df_test = df_test.drop(columns=excl_cols)
df_test_cat_enc = pd.read_parquet(CFG.output_dir + 'test_cat_all_p2_encoded_features.parquet')
test_cus = df_test['customer_ID'].drop_duplicates()
n_cus = test_cus.shape[0]

cus_chunk_0, cus_chunk_1 = test_cus.iloc[:(n_cus//2)], test_cus.iloc[(n_cus//2):]
df_test = df_test[df_test['customer_ID'].isin(cus_chunk_0)]
gc.collect()

seq_test, seq_test_cus = gen_sequence_data(df_test, df_test_cat_enc)

with open(CFG.output_dir + 'seq_v2_test_chunk0.npy', 'wb') as f:
    np.save(f, seq_test)
seq_test_cus.to_parquet(CFG.output_dir + 'seq_v2_test_customers_chunk0.parquet')   
print(seq_test.shape)
print(seq_test_cus.shape)

del df_test, seq_test, seq_test_cus
gc.collect()

print('Sequence Converting Test 1')
df_test = pd.read_parquet(CFG.output_dir + 'test_lgb_no_lag_imputed.parquet')
df_test = df_test.drop(columns=excl_cols)
df_test = df_test[df_test['customer_ID'].isin(cus_chunk_1)]
gc.collect()

seq_test, seq_test_cus = gen_sequence_data(df_test, df_test_cat_enc)

with open(CFG.output_dir + 'seq_v2_test_chunk1.npy', 'wb') as f:
    np.save(f, seq_test)
seq_test_cus.to_parquet(CFG.output_dir + 'seq_v2_test_customers_chunk1.parquet')   
print(seq_test.shape)
print(seq_test_cus.shape)

del df_test, seq_test, seq_test_cus
gc.collect()