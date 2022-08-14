import gc
from typing import Callable
import os
import joblib
import json
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers

from preprocessing.config_preproc import PreprocConfig as CFG_P
from utils.evaluation import amex_metric

# credit to Chris Deotte
# https://www.kaggle.com/code/cdeotte/tensorflow-transformer-0-790
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, feat_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="gelu"), layers.Dense(feat_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def train_save_seq_model(X_train_all : np.ndarray, train_cus_folds_y : pd.DataFrame,
                         X_test : np.ndarray, test_cus : pd.DataFrame,  
                         get_model : Callable, model_kwargs : dict, 
                         output_dir_name : str, runs_per_fold : int = 1,
                         list_input = False):

    model_log = {}
    model_log['model_kwargs'] = str(model_kwargs)
    model_log['runs_per_fold'] = runs_per_fold

    if not list_input:
        model_log['feature_dim'] = X_train_all.shape[-1]
    else:
        model_log['feature_dim'] = 'multiple inputs'

    oof_preds = train_cus_folds_y[['customer_ID']].drop_duplicates(subset=['customer_ID'], keep='first')
    oof_preds['oof_pred'] = 0

    test_preds = test_cus[['customer_ID']]
    test_preds['prediction'] = 0

    avg_score = 0

    n_folds = train_cus_folds_y['val_fold_n'].nunique() 
    for val_fold in range(n_folds):
        print(f'Training Fold {val_fold}: \n')
        val_mask = train_cus_folds_y['val_fold_n'] == val_fold
        dupe_mask = train_cus_folds_y.duplicated(subset=['customer_ID'], keep='first')

        if not list_input:
            X_train, X_val = X_train_all[~val_mask.values], X_train_all[val_mask.values & ~dupe_mask.values]
        else:
            X_train = [branch[~val_mask.values] for branch in X_train_all]
            X_val = [branch[val_mask.values & ~dupe_mask.values] for branch in X_train_all]
        
        y_train = train_cus_folds_y.loc[~val_mask, ['target']]
        
        oof_mask = oof_preds['customer_ID'].isin(train_cus_folds_y.loc[val_mask, 'customer_ID'].unique())
        y_val = train_cus_folds_y.loc[val_mask & ~dupe_mask, 'target'] 

        for _ in range(runs_per_fold):

            model, predict_func = get_model(X_train, y_train, X_val, y_val, 
                                            model_kwargs)

            oof_preds.loc[oof_mask,'oof_pred'] += predict_func(model, X_val) / runs_per_fold
            test_preds['prediction'] += predict_func(model, X_test) / (n_folds * runs_per_fold)
        
            tf.keras.backend.clear_session()

        score = amex_metric(pd.DataFrame({'target' : y_val.values}), 
                            pd.DataFrame({'prediction' : oof_preds.loc[oof_mask,'oof_pred'].values}))
        avg_score += score / n_folds

        print(f'Fold {val_fold} exact Amex val score: {score}')
        model_log[f'Fold {val_fold} OOF'] = score

        del X_train, X_val, y_train, y_val
        gc.collect()

        del model
        gc.collect()
    
    tot_score = amex_metric(pd.DataFrame({'target' : train_cus_folds_y.drop_duplicates(subset=['customer_ID'], keep='first')['target'].values}), 
                            pd.DataFrame({'prediction' : oof_preds['oof_pred'].values}))
    print(f'Total out of fold exact Amex val score is {tot_score}')
    print(f'Average out of fold exact Amex val score is {avg_score}')
    print('All fold scores:')
    print([model_log[f'Fold {val_fold} OOF'] for val_fold in range(n_folds)])
    model_log[f'Total OOF score'] = tot_score
    model_log[f'Average OOF score'] = avg_score
    
    out_path = CFG_P.model_output_dir + f'{output_dir_name}' 
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    oof_preds.to_parquet(out_path + f'/oof_preds.parquet')
    test_preds.to_csv(out_path + f'/test_preds.csv', index=False)

    with open(out_path + "/model_log.json", "w") as outfile:
        json.dump(model_log, outfile)

def get_seq_keras_model(X_train : np.ndarray, y_train : pd.Series, 
                        X_val : np.ndarray, y_val : pd.Series,
                        keras_kwargs : dict):
    
    model = keras_kwargs['architecture_constructor']()
    model.fit(X_train, y_train, 
              validation_data=(X_val, y_val), 
              **keras_kwargs['fit_kwargs'])

    def predict_func(model, X):
        return model.predict(X, verbose=0).flatten()

    return model, predict_func