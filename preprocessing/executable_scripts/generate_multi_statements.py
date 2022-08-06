import gc
import pandas as pd 

from preprocessing.config_preproc import PreprocConfig as CFG    

# credit to Carl McBride Ellis for use of nth 
# https://www.kaggle.com/competitions/amex-default-prediction/discussion/332880
def gen_multi_statement_features(df_in : pd.DataFrame, n_statements : int) -> pd.DataFrame:
    """Maps customer df to multiple training records for each customer
       
    Args:
        df_in (pd.DataFrame): raw input dataframe
        n_statements (int): number of statements to use

    Returns:
        pd.DataFrame: output data with multiple training records
    """

    df_in = df_in.drop(columns=['S_2']) # remove date col
    df_outs = []
    
    for i in range(1, n_statements + 1):
        print(f'Processing {i}th history')

        df_out = df_in.groupby("customer_ID", as_index=False).nth([-i])
        df_outs.append(df_out)

    df_return = pd.concat(df_outs).reset_index(drop=True)
    
    del df_outs
    gc.collect()

    return df_return 

# for now will augment train but not test
print('Processing Test')
df_test = pd.read_parquet(CFG.test_feature_file)
df_test = gen_multi_statement_features(df_test, 1)
df_test.to_parquet(CFG.output_dir + 'test_1_statements.parquet')

del df_test
gc.collect()

print('Processing Train')
df_train = pd.read_parquet(CFG.train_feature_file)
df_train = gen_multi_statement_features(df_train, 1)
df_train.to_parquet(CFG.output_dir + 'train_1_statements.parquet')

del df_train
gc.collect()