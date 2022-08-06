import gc
import pandas as pd 

from preprocessing.config_preproc import PreprocConfig as CFG 

def gen_ex_last_raw_data(df_in : pd.DataFrame, n_drops : int = 1) -> pd.DataFrame:
    """Maps customer df to version with last n_drops statements removed
       
    Args:
        df_in (pd.DataFrame): raw input dataframe

    Returns:
        pd.DataFrame: input with statements removed
    """

    for i in range(n_drops):
        df_in['last_S_2'] = df_in.groupby('customer_ID')['S_2'].transform('last')
        df_in = df_in[df_in['last_S_2'] != df_in['S_2']]
        df_in = df_in.drop(columns=['last_S_2']) 

    return df_in

print('Processing Test')
df_test = pd.read_parquet(CFG.test_feature_file)
df_test = gen_ex_last_raw_data(df_test)
print(df_test.shape)
print(df_test.info())
df_test.to_parquet(CFG.output_dir + 'test_ex_last1.parquet')

del df_test
gc.collect()

print('Processing Train')
df_train = pd.read_parquet(CFG.train_feature_file)
df_train = gen_ex_last_raw_data(df_train)
print(df_train.shape)
print(df_train.info())
df_train.to_parquet(CFG.output_dir + 'train_ex_last1.parquet')

del df_train
gc.collect()