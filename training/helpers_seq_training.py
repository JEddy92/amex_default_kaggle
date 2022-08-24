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

# credit to Chris Deotte for base version
# https://www.kaggle.com/code/cdeotte/tensorflow-transformer-0-790
# class TransformerBlock(layers.Layer):
#     def __init__(self, embed_dim, feat_dim, num_heads, ff_dim, rate=0.1):
#         super(TransformerBlock, self).__init__()
#         self.supports_masking = True
#         self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
#         self.ffn = keras.Sequential(
#             [layers.Dense(ff_dim, activation="gelu"), layers.Dense(feat_dim),]
#         )
#         self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
#         self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
#         self.dropout1 = layers.Dropout(rate)
#         self.dropout2 = layers.Dropout(rate)

#     # def compute_mask(self, inputs, mask=None):
#     #     return mask

#     def call(self, inputs, training, mask=None):
#         attention_mask = mask[:, tf.newaxis, tf.newaxis, :]
#         attn_output = self.att(inputs, inputs, attention_mask=attention_mask)
#         attn_output = self.dropout1(attn_output, training=training)
#         out1 = self.layernorm1(inputs + attn_output)
#         ffn_output = self.ffn(out1)
#         ffn_output = self.dropout2(ffn_output, training=training)
#         return self.layernorm2(out1 + ffn_output)

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

# https://towardsdatascience.com/the-time-series-transformer-2a521a0efad3
class Time2Vec(keras.layers.Layer):
    def __init__(self, kernel_size=1):
        super(Time2Vec, self).__init__(trainable=True, name='Time2VecLayer')
        self.k = kernel_size
    
    def build(self, input_shape):
        # trend
        self.wb = self.add_weight(name='wb',shape=(input_shape[1],),initializer='uniform',trainable=True)
        self.bb = self.add_weight(name='bb',shape=(input_shape[1],),initializer='uniform',trainable=True)
        # periodic
        self.wa = self.add_weight(name='wa',shape=(1, input_shape[1], self.k),initializer='uniform',trainable=True)
        self.ba = self.add_weight(name='ba',shape=(1, input_shape[1], self.k),initializer='uniform',trainable=True)
        super(Time2Vec, self).build(input_shape)
    
    def call(self, inputs, **kwargs):
        bias = self.wb * inputs + self.bb
        dp = K.dot(inputs, self.wa) + self.ba
        wgts = K.sin(dp) # or K.cos(.)

        ret = K.concatenate([K.expand_dims(bias, -1), wgts], -1)
        ret = K.reshape(ret, (-1, inputs.shape[1]*(self.k+1)))
        return ret
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]*(self.k + 1))

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
                                            X_test, 
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
                        X_test : np.ndarray, 
                        keras_kwargs : dict):
    
    model = keras_kwargs['architecture_constructor']()
    model.fit(X_train, y_train, 
              validation_data=(X_val, y_val), 
              **keras_kwargs['fit_kwargs'])

    def predict_func(model, X):
        return model.predict(X, verbose=0).flatten()

    return model, predict_func

def get_seq_keras_model_w_pretrain(X_train : np.ndarray, y_train : pd.Series, 
                                   X_val : np.ndarray, y_val : pd.Series,
                                   X_test : np.ndarray, 
                                   keras_kwargs : dict):
    
    # Pretrain Step
    # assumes P_2 is set as first column, pretrain target is last P_2 value
    X_comb = np.concatenate([X_train, X_test], axis=0)
    y_pretrain = X_comb[:,-1,0]  
    X_comb = X_comb[:,:,1:] # drop P_2

    pretrain_model = keras_kwargs['pretrain_architecture_constructor']()
    print(pretrain_model.summary())
    pretrain_model.fit(X_comb, y_pretrain, 
                       validation_split=.2, 
                       **keras_kwargs['pretrain_fit_kwargs'])

    del X_comb
    gc.collect()

    # drop P_2
    X_train = X_train[:,:,1:]
    X_val = X_val[:,:,1:]
    X_test = X_test[:,:,1:]

    # pretrain_model.save('model_temp')
    pretrained_base = keras.Model(pretrain_model.get_layer('input').input, 
                                  pretrain_model.get_layer('transformer_out').output)
    print(pretrained_base.summary())

    model = keras_kwargs['train_architecture_constructor'](pretrained_base)
    for l in model.layers[:2]:
        l.trainable = False
    print(model.summary())

    # warm up
    model.fit(X_train, y_train, 
              validation_data=(X_val, y_val), 
              **keras_kwargs['train_warmup_kwargs'])

    for l in model.layers[:2]:
        l.trainable = True        

    print(model.summary())
    # model.load_weights('model_temp', by_name=True)
    model.fit(X_train, y_train, 
              validation_data=(X_val, y_val), 
              **keras_kwargs['train_fit_kwargs'])

    def predict_func(model, X):
        return model.predict(X[:,:,1:], verbose=0).flatten() # drop P_2 in predict

    return model, predict_func