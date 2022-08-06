import gc
from operator import attrgetter
import numpy as np
import pandas as pd 

from preprocessing.config_preproc import PreprocConfig as CFG    

# credit to Carl McBride Ellis for nth approach. Use both this
# and based on month which is 2 different views (whether to keep holes
# in statement history as in the month-based version) 
# https://www.kaggle.com/competitions/amex-default-prediction/discussion/332880
def gen_entire_flat_features(df_in : pd.DataFrame) -> pd.DataFrame:
    """Maps customer df to flattened version of the entire raw data
       
    Args:
        df_in (pd.DataFrame): raw input dataframe

    Returns:
        pd.DataFrame: full flattened output data
    """

    dtypes_dict = df_in.drop(columns=['S_2']).dtypes.to_dict()
    dtypes_out_dict = {'customer_ID' :dtypes_dict['customer_ID']}

    df_in['S_2'] = pd.to_datetime(df_in['S_2'])
    df_in['last_date'] = df_in.groupby('customer_ID')['S_2'].transform('last')
    df_in['months_ago'] = (df_in['last_date'].dt.to_period('M') - df_in['S_2'].dt.to_period('M')) \
                            .apply(attrgetter('n'))
    df_in['months_ago'] = df_in['months_ago'].astype('int')
    print(df_in[['customer_ID','months_ago']].head(10))

    print(df_in.info())
    df_outs = []
    
    for i in range(0, CFG.max_n_statement):
        print(f'Processing {i}th history')
        for c, v in dtypes_dict.items():
            if c !=  'customer_ID':
                dtypes_out_dict[f'{c}_{i}'] = v

        df_out = df_in[df_in['months_ago'] == i]
        df_out = df_out.drop(columns=['S_2','last_date','months_ago'])
        df_out = df_out.add_suffix(f'_{i}')
        # df_out = df_in.groupby("customer_ID", as_index=False).nth([-i]).add_suffix(f'_{i}')
        df_out = df_out.rename(columns = {f'customer_ID_{i}':'customer_ID'})

        df_outs.append(df_out.set_index('customer_ID'))

    df_return = pd.concat(df_outs, axis=1).reset_index()
    del df_outs
    gc.collect()
    print('Concatenation complete')

    df_return = df_return.fillna(-1)
    return df_return.astype(dtypes_out_dict) 

print('Flattening Test')
df_test = pd.read_parquet(CFG.test_feature_file)
df_test = gen_entire_flat_features(df_test)
print(df_test.shape)
print(df_test.info())
df_test.to_parquet(CFG.output_dir + 'test_flattened_full.parquet')

del df_test
gc.collect()

print('Flattening Train')
df_train = pd.read_parquet(CFG.train_feature_file)
df_train = gen_entire_flat_features(df_train)
print(df_train.shape)
print(df_train.info())
df_train.to_parquet(CFG.output_dir + 'train_flattened_full.parquet')

del df_train
gc.collect()