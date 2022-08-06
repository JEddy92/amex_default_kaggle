import gc
import numpy as np
import pandas as pd 

from preprocessing.helpers_preproc import get_flat_history_features 
from preprocessing.config_preproc import PreprocConfig as CFG    

def gen_flat_history_features(df_in : pd.DataFrame, his_features : list) -> pd.DataFrame:
    """Maps customer df to entire flattened history features
       for select set of features
       
    Args:
        df_in (pd.DataFrame): raw input dataframe

    Returns:
        pd.DataFrame: full history flattened output features
    """

    df_outs = []
    for his_feature in his_features:
        df_outs.append(get_flat_history_features(df_in, his_feature).set_index('customer_ID'))
    
    return pd.concat(df_outs, axis=1).reset_index()

flatten_features = ['P_2', 'B_1'] #'B_2', 'D_48', 'R_1']

print('Processing train histories')
df_train = pd.read_parquet(CFG.train_feature_file)

df_train = gen_flat_history_features(df_train, flatten_features)
redundant_features = [f for f in df_train.columns if 'his_12' in f] # already getting last elsewhere
df_train = df_train.drop(columns=redundant_features)

df_train.to_parquet(CFG.output_dir + 'train_flattened_his_features.parquet')

print(df_train.head(10))

del df_train
gc.collect()

print('Processing test histories')
df_test = pd.read_parquet(CFG.test_feature_file)

df_test = gen_flat_history_features(df_test, flatten_features)
df_test = df_test.drop(columns=redundant_features)
df_test.to_parquet(CFG.output_dir + 'test_flattened_his_features.parquet')

del df_test
gc.collect()