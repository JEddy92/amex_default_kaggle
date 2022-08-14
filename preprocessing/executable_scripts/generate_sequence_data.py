import gc
from operator import attrgetter
import numpy as np
import pandas as pd 

from preprocessing.config_preproc import PreprocConfig as CFG    

MISSING_NULL_VALUE = -3
LEADING_NULL_VALUE = -2
RANDOM_NULL_VALUE = -1

# trial_cus = ['19bf7c6c1c1419d88486f4b4174fb11fb7fa76089be1bfcefab6cf750498827b',
#              'fffe2bc02423407e33a607660caeed076d713d8a5ad32321530e92704835da88'
#              ]

def gen_sequence_data(df_in : pd.DataFrame):
    """Maps customer df to sequential-tensor structured data
    (input to sequential neural net models)
       
    Args:
        df_in (pd.DataFrame): raw input dataframe

    Returns:
        Tuple:
            np.ndarray: sequential output array
            pd.DataFrame: single column of ordered customers
    """

    dtypes_dict = df_in.drop(columns=['S_2']).dtypes.to_dict()

    df_cus = df_in[['customer_ID']].drop_duplicates().sort_values(by='customer_ID')

    df_in['S_2'] = pd.to_datetime(df_in['S_2'])
    df_in['last_date'] = df_in.groupby('customer_ID')['S_2'].transform('last')
    df_in['months_ago'] = (df_in['last_date'].dt.to_period('M') - df_in['S_2'].dt.to_period('M')) \
                            .apply(attrgetter('n'))
    df_in['months_ago'] = df_in['months_ago'].astype('int')
    # print(df_in[['customer_ID','months_ago']].head(10))
    # print(df_in.loc[df_in['customer_ID'].isin(trial_cus), ['customer_ID','months_ago']])

    df_in[df_in == -1] = np.nan

    # Fill leading null values
    print('filling leading null values')
    df_in = df_in.groupby('customer_ID') \
                 .apply(lambda grp: grp.where(grp.ffill().notna(), LEADING_NULL_VALUE))

    # Fill random null values
    df_in = df_in.fillna(RANDOM_NULL_VALUE)
    print(df_in.info())
    
    df_outs = []
    
    for i in range(CFG.max_n_statement-1, -1, -1):
        print(f'Processing {i} months ago statement')

        df_out = df_in[df_in['months_ago'] == i]
        df_out = df_out.drop(columns=['S_2','last_date','months_ago'])

        # Fill missing statement null values
        df_out = pd.merge(df_out, df_cus, on='customer_ID', how='outer') \
                   .fillna(MISSING_NULL_VALUE)
        df_out = df_out.astype(dtypes_dict)
        df_out = df_out.sort_values(by='customer_ID')

        # Rearrange column order to put categories first
        cat_features = CFG.cat_features
        num_features = [f for f in df_out.columns if f not in cat_features]
        df_out = df_out[cat_features + num_features]

        df_outs.append(df_out.drop(columns=['customer_ID']))

    np_return = [np.split(df_out.to_numpy(), indices_or_sections=df_cus.shape[0], axis=0) 
                for df_out in df_outs]
    np_return = np.concatenate(np_return, axis=1)
    print(np_return.shape)
    
    del df_outs
    gc.collect()
    print('Np conversion complete')

    return np_return, df_cus

print('Sequence Converting Train')
df_train = pd.read_parquet(CFG.train_feature_file)
# df_train = df_train.loc[df_train['customer_ID'].isin(trial_cus),
#                         ['customer_ID', 'S_2', 'P_2', 'D_39', 'D_115']]

seq_train, seq_train_cus = gen_sequence_data(df_train)
print(seq_train.shape)
print(seq_train_cus.shape)

with open(CFG.output_dir + 'seq_train.npy', 'wb') as f:
    np.save(f, seq_train)
seq_train_cus.to_parquet(CFG.output_dir + 'seq_train_customers.parquet')   

del df_train, seq_train, seq_train_cus
gc.collect()

print('Sequence Converting Test 0')
df_test = pd.read_parquet(CFG.test_feature_file)
test_cus = df_test['customer_ID'].drop_duplicates()
n_cus = test_cus.shape[0]

cus_chunk_0, cus_chunk_1 = test_cus.iloc[:(n_cus//2)], test_cus.iloc[(n_cus//2):]
df_test = df_test[df_test['customer_ID'].isin(cus_chunk_0)]
gc.collect()

seq_test, seq_test_cus = gen_sequence_data(df_test)

with open(CFG.output_dir + 'seq_test_chunk0.npy', 'wb') as f:
    np.save(f, seq_test)
seq_test_cus.to_parquet(CFG.output_dir + 'seq_test_customers_chunk0.parquet')   
print(seq_test.shape)
print(seq_test_cus.shape)

del df_test, seq_test, seq_test_cus
gc.collect()

print('Sequence Converting Test 1')
df_test = pd.read_parquet(CFG.test_feature_file)
df_test = df_test[df_test['customer_ID'].isin(cus_chunk_1)]
gc.collect()

seq_test, seq_test_cus = gen_sequence_data(df_test)

with open(CFG.output_dir + 'seq_test_chunk1.npy', 'wb') as f:
    np.save(f, seq_test)
seq_test_cus.to_parquet(CFG.output_dir + 'seq_test_customers_chunk1.parquet')   
print(seq_test.shape)
print(seq_test_cus.shape)

del df_test, seq_test, seq_test_cus
gc.collect()