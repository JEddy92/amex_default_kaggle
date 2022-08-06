import pandas as pd
from sklearn.decomposition import TruncatedSVD

from preprocessing.config_preproc import PreprocConfig as CFG 

def get_agg_features(df_in : pd.DataFrame, group_col : str, 
                     agg_features : list, agg_funcs : list) -> pd.DataFrame:
    """Transform raw dataframe into grouped aggregations

    Args:
        df_in (pd.DataFrame): raw dataframe
        group_col (str): column to group by
        agg_features (list): features to aggregate
        agg_funcs (list): aggregator functions to apply

    Returns:
        pd.DataFrame: aggregated dataframe
    """

    df_agg = df_in.groupby(group_col)[agg_features].agg(agg_funcs)
    df_agg.columns = ['_'.join(x) for x in df_agg.columns]
    return df_agg.reset_index()

def get_flat_history_features(df_in : pd.DataFrame, his_feature : str) -> pd.DataFrame:
    """Transform raw dataframe into full flattened history for selected feature

    Args:
        df_in (pd.DataFrame): raw dataframe
        his_feature (str): feature to create flat history for

    Returns:
        pd.DataFrame: full history flattened output for feature 
    """

    df_his_features = df_in.groupby('customer_ID')[[his_feature]] \
                           .apply(lambda x: x.reset_index(drop=True).transpose()) \
                           .reset_index().drop(columns=['level_1'])
    df_his_features.columns = ['customer_ID'] + [f'{his_feature}_his_{i}' for i in range(CFG.max_n_statement)]

    return df_his_features

def get_svd_features(df_in : pd.DataFrame, n_components : int, 
                     feature_name : str) -> pd.DataFrame:
    """Map df feature subset to SVD factorized output

    Args:
        df_in (pd.DataFrame): df to factorize
        n_components (int): number of SVD components to return
        feature_name (str): name to use for output component features

    Returns:
        pd.DataFrame: factorized output
    """ 

    tSVD = TruncatedSVD(n_components = n_components)
    return pd.DataFrame(data=tSVD.fit_transform(df_in), 
                        columns=[f'{feature_name}_{i}' for i in range(n_components)])