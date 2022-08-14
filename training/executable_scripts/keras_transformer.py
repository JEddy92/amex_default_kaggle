import gc
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from preprocessing.config_preproc import PreprocConfig as CFG_P
from training import helpers_flat_training as helpers_flat_tr
from training import helpers_seq_training as helpers_seq_tr
from utils.evaluation import amex_metric_tensorflow

# credit to Chris Deotte for base model architecture and params
# https://www.kaggle.com/code/cdeotte/tensorflow-transformer-0-790
feat_dim = 190
embed_dim = 64  # Embedding size for attention
num_heads = 4  #4  Number of attention heads
ff_dim = 128  # 128 Hidden layer size in feed forward network inside transformer
dropout_rate = 0.30 #.3
num_blocks = 3 #2

def architecture_constructor():
    
    # INPUT EMBEDDING LAYER
    inp = layers.Input(shape=(13,190))
    raw_num = inp[:,-1,11:]
    last_num = layers.Dense(64, activation="relu")(raw_num)
    #inp = layers.Masking(mask_value=-3)(inp) 
    embeddings = []
    for k in range(11):
        emb = layers.Embedding(15,4) 
        embeddings.append( emb(inp[:,:,k]) )
    x = layers.Concatenate()([inp[:,:,11:]]+embeddings)
    x = layers.Dense(feat_dim)(x)
    
    # TRANSFORMER BLOCKS
    for k in range(num_blocks):
        x_old = x
        transformer_block = helpers_seq_tr.TransformerBlock(embed_dim, feat_dim, num_heads, ff_dim, dropout_rate)
        x = transformer_block(x)
        #x = layers.Concatenate()([x_old, x])
        x = .9*x + .1*x_old # .9 .1 SKIP CONNECTION
    
    # CLASSIFICATION HEAD
    x = x[:,-1,:] 
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Concatenate()([x, last_num])
    #x = layers.Dropout(.20)(x)
    x = layers.Dense(64, activation="relu")(x)
    #x = layers.Dropout(.20)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = keras.Model(inputs=inp, outputs=outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=0.01) #.001
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(loss=loss, optimizer = opt, metrics=[amex_metric_tensorflow])
        
    return model

tf.random.set_seed(CFG_P.seed)
    
lr = ReduceLROnPlateau(monitor="val_loss", factor=0.7, 
                       patience=4, verbose=2)

es = EarlyStopping(monitor='val_amex_metric_tensorflow',
                   patience=10, #45 
                   verbose=1,
                   mode="max", 
                   restore_best_weights=True)

LR_START = 1e-6
LR_MAX = 1e-3
LR_MIN = 1e-6
LR_RAMPUP_EPOCHS = 0
LR_SUSTAIN_EPOCHS = 0
EPOCHS = 40

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        decay_total_epochs = EPOCHS - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS - 1
        decay_epoch_index = epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
        phase = math.pi * decay_epoch_index / decay_total_epochs
        cosine_decay = 0.5 * (1 + math.cos(phase))
        lr = (LR_MAX - LR_MIN) * cosine_decay + LR_MIN
    return lr

rng = [i for i in range(EPOCHS)]
lr_y = [lrfn(x) for x in rng]
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}". \
      format(lr_y[0], max(lr_y), lr_y[-1]))
LR = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)

# LR_START = 1e-6
# LR_MAX = 1e-3
# LR_MIN = 1e-6
# LR_RAMPUP_EPOCHS = 0
# LR_SUSTAIN_EPOCHS = 0
# EPOCHS = 42
# STEPS = [6,12,24] #

# def lrfn(epoch):
#     if epoch<STEPS[0]:
#         epoch2 = epoch
#         EPOCHS2 = STEPS[0]
#     elif epoch<STEPS[0]+STEPS[1]:
#         epoch2 = epoch-STEPS[0]
#         EPOCHS2 = STEPS[1]
#     elif epoch<STEPS[0]+STEPS[1]+STEPS[2]:
#         epoch2 = epoch-STEPS[0]-STEPS[1]
#         EPOCHS2 = STEPS[2]
    
#     if epoch2 < LR_RAMPUP_EPOCHS:
#         lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch2 + LR_START
#     elif epoch2 < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
#         lr = LR_MAX
#     else:
#         decay_total_epochs = EPOCHS2 - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS - 1
#         decay_epoch_index = epoch2 - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
#         phase = math.pi * decay_epoch_index / decay_total_epochs
#         cosine_decay = 0.5 * (1 + math.cos(phase))
#         lr = (LR_MAX - LR_MIN) * cosine_decay + LR_MIN
#     return lr

# rng = [i for i in range(EPOCHS)]
# lr_y = [lrfn(x) for x in rng]

# print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}". \
#           format(lr_y[0], max(lr_y), lr_y[-1]))
# lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)

callbacks = [LR, es, tf.keras.callbacks.TerminateOnNaN()]

fit_kwargs = {
    'epochs' : 100,
    'verbose' : 2,
    'batch_size' : 256,
    'shuffle' : True,
    'callbacks' : callbacks
}

keras_kwargs = {
    'architecture_constructor' : architecture_constructor,
    'fit_kwargs' : fit_kwargs
}

X_train_all = np.load(CFG_P.output_dir + 'seq_v2_train.npy')
X_train_all[:,:,:11] += 3 # fix for embedding integer scale
# X_train_all = np.ma.masked_equal(X_train_all, -3)
train_cus_folds_y = pd.read_parquet(CFG_P.output_dir + 'seq_v2_train_customers.parquet')
train_cus_folds_y = train_cus_folds_y.reset_index(drop=True)
print(train_cus_folds_y.index)
train_cus_folds_y['orig_order'] = train_cus_folds_y.index 

train_cus_folds_y = pd.merge(train_cus_folds_y, pd.read_parquet(CFG_P.output_dir + 'train_labels_w_10_p2_strat_folds.parquet'),
                             on='customer_ID')
train_cus_folds_y = train_cus_folds_y.sort_values(by='orig_order').drop(columns=['orig_order'])                             

X_test = np.concatenate([np.load(CFG_P.output_dir + 'seq_v2_test_chunk0.npy'),
                         np.load(CFG_P.output_dir + 'seq_v2_test_chunk1.npy')])
X_test[:,:,:11] += 3 # fix for embedding integer scale
#X_test = np.ma.masked_equal(X_test, -3)
test_cus = pd.concat([pd.read_parquet(CFG_P.output_dir + 'seq_v2_test_customers_chunk0.parquet'),
                      pd.read_parquet(CFG_P.output_dir + 'seq_v2_test_customers_chunk1.parquet')]).reset_index(drop=True)                         

print(X_train_all.shape, train_cus_folds_y.shape, X_test.shape)

def get_scale_reshape(X):

    return X[:,:,11:].reshape(-1, X[:,:,11:].shape[-1]) 

scaler = StandardScaler()
# scaler = RobustScaler()
scaler.fit(np.concatenate([get_scale_reshape(X_train_all),
                           get_scale_reshape(X_test)]))
X_train_all[:,:,11:] = scaler.transform(get_scale_reshape(X_train_all)).reshape(X_train_all[:,:,11:].shape)                          
X_test[:,:,11:] = scaler.transform(get_scale_reshape(X_test)).reshape(X_test[:,:,11:].shape) 

print(X_train_all.shape, X_test.shape)

helpers_seq_tr.train_save_seq_model(X_train_all, train_cus_folds_y, 
                                    X_test, test_cus,
                                    helpers_seq_tr.get_seq_keras_model, keras_kwargs,
                                    'keras_transformer_base_v3_new_data', runs_per_fold=3) 