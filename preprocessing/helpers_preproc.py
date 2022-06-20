import pandas as pd

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
    return df_agg.reset_index(drop=True)

