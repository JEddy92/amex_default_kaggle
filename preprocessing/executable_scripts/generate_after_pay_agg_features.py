import gc
import numpy as np
import pandas as pd 

from preprocessing.helpers_preproc import get_agg_features 
from preprocessing.config_preproc import PreprocConfig as CFG    

# credit to Jiwei Liu
# https://www.kaggle.com/code/jiweiliu/rapids-cudf-feature-engineering-xgb/notebook
def get_after_pay_agg_features(df_in : pd.DataFrame) -> pd.DataFrame:
    """Maps customer df to after pay aggregated features
       
    Args:
        df_in (pd.DataFrame): raw input dataframe

    Returns:
        pd.DataFrame: aggregated output features aligned to last time step
    """

    agg_funcs = ['mean', 'std', 'min', 'max', 'last', 'first'] 
    
    ap_features = []
    for bcol in [f'B_{i}' for i in [11,14,17]] +['D_39','D_131'] +[f'S_{i}' for i in [16,23]]:
        for pcol in ['P_2','P_3']:
            if bcol in df_in.columns:
                new_feature = f'{bcol}-{pcol}' 
                ap_features.append(new_feature)
                df_in[new_feature] = df_in[bcol] - df_in[pcol]

    return get_agg_features(df_in=df_in, group_col='customer_ID', 
                            agg_features=ap_features, agg_funcs=agg_funcs)

print('Processing train aggregates')
df_train = pd.read_parquet(CFG.train_feature_file)

df_train = get_after_pay_agg_features(df_train)
print(df_train.shape, df_train.info())
df_train.to_parquet(CFG.output_dir + 'train_after_pay_agg_features.parquet')

del df_train
gc.collect()

print('Processing test aggregates')
df_test = pd.read_parquet(CFG.test_feature_file)

df_test = get_after_pay_agg_features(df_test)
df_test.to_parquet(CFG.output_dir + 'test_after_pay_agg_features.parquet')

del df_test
gc.collect()