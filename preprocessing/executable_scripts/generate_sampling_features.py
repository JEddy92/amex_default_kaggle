import gc
import pandas as pd 

from preprocessing.config_preproc import PreprocConfig as CFG 

def gen_sampling_features(df_in : pd.DataFrame) -> pd.DataFrame:
    """Maps customer df to date sampling features
       
    Args:
        df_in (pd.DataFrame): raw input dataframe

    Returns:
        pd.DataFrame: date sampling feaure outputs
    """

    df_in =  df_in.groupby('customer_ID').last().reset_index()
    df_in['last_month'] = pd.to_datetime(df_in['S_2']).dt.month
    df_in['S_2_count'] = df_in['S_2'].map(df_in['S_2'].value_counts())
    df_in['last_month_count'] = df_in['last_month'].map(df_in['last_month'].value_counts()) 
    df_in['S_2_month_freq'] = df_in['S_2_count'] / df_in['last_month_count']  

    return df_in.drop(columns=['S_2','last_month','S_2_count','last_month_count']) 

print('Processing Test')
df_test = pd.read_parquet(CFG.test_feature_file, columns=['customer_ID','S_2'])
df_test = gen_sampling_features(df_test)
print(df_test.shape)
print(df_test.info())
df_test.to_parquet(CFG.output_dir + 'test_sampling_features.parquet')

del df_test
gc.collect()

print('Processing Train')
df_train = pd.read_parquet(CFG.train_feature_file, columns=['customer_ID','S_2'])
df_train = gen_sampling_features(df_train)
print(df_train.shape)
print(df_train.info())
df_train.to_parquet(CFG.output_dir + 'train_sampling_features.parquet')

del df_train
gc.collect()