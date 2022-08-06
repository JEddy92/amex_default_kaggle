import numpy as np
import pandas as pd
import tensorflow as tf

# numpy metric created by https://www.kaggle.com/yunchonggan
# https://www.kaggle.com/competitions/amex-default-prediction/discussion/328020
def amex_metric_np(preds: np.ndarray, target: np.ndarray) -> float:
    """Fast approximation to the competition metric

    Args:
        preds (np.ndarray): predicted probabilities
        target (np.ndarray): binary ground truth

    Returns:
        float: ~amex metric
    """
    indices = np.argsort(preds)[::-1]
    preds, target = preds[indices], target[indices]

    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight / weight.sum()).cumsum()
    four_pct_mask = cum_norm_weight <= 0.04
    d = np.sum(target[four_pct_mask]) / np.sum(target)

    weighted_target = target * weight
    lorentz = (weighted_target / weighted_target.sum()).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    n_pos = np.sum(target)
    n_neg = target.shape[0] - n_pos
    gini_max = 10 * n_neg * (n_pos + 20 * n_neg - 19) / (n_pos + 20 * n_neg)

    g = gini / gini_max
    return 0.5 * (g + d)

# official metric
# https://www.kaggle.com/code/inversion/amex-competition-metric-python
def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    """Exact competition metric

    Args:
        y_true (pd.DataFrame): binary ground truth
        y_pred (pd.DataFrame): predicted probabilities

    Returns:
        float: exact amex metric
    """

    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
        
    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)

def amex_metric_np_lgb(y_pred, y_true):
    return 'amex_metric', amex_metric_np(y_pred, y_true.get_label()), True

def amex_metric_np_xgb(y_pred, y_true):
    return 'amex', amex_metric_np(y_pred, y_true.get_label())

# credit to Rohan Rao
# https://www.kaggle.com/code/rohanrao/amex-competition-metric-implementations
def amex_metric_tensorflow(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:

    # convert dtypes to float64
    y_true = tf.cast(y_true, dtype=tf.float64)
    y_pred = tf.cast(y_pred, dtype=tf.float64)

    # count of positives and negatives
    n_pos = tf.math.reduce_sum(y_true)
    n_neg = tf.cast(tf.shape(y_true)[0], dtype=tf.float64) - n_pos

    # sorting by descring prediction values
    indices = tf.argsort(y_pred, axis=0, direction='DESCENDING')
    preds, target = tf.gather(y_pred, indices), tf.gather(y_true, indices)

    # filter the top 4% by cumulative row weights
    weight = 20.0 - target * 19.0
    cum_norm_weight = tf.cumsum(weight / tf.reduce_sum(weight))
    four_pct_filter = cum_norm_weight <= 0.04

    # default rate captured at 4%
    d = tf.reduce_sum(target[four_pct_filter]) / n_pos

    # weighted gini coefficient
    lorentz = tf.cumsum(target / n_pos)
    gini = tf.reduce_sum((lorentz - cum_norm_weight) * weight)

    # max weighted gini coefficient
    gini_max = 10 * n_neg * (1 - 19 / (n_pos + 20 * n_neg))

    # normalized weighted gini coefficient
    g = gini / gini_max

    return 0.5 * (g + d)