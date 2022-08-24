import gc
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from preprocessing.config_preproc import PreprocConfig as CFG_P
from training import helpers_flat_training as helpers_flat_tr
from training import helpers_seq_training as helpers_seq_tr
from utils.evaluation import amex_metric_tensorflow

# TO DO: PUT IN CFG
MISSING_NULL_VALUE = -3

# credit to Chris Deotte for base model architecture and params
# https://www.kaggle.com/code/cdeotte/tensorflow-transformer-0-790
feat_dim = 128 #128
embed_dim = 32  # 32 Embedding size for attention
num_heads = 4  #4  Number of attention heads
ff_dim = 128  # 128 Hidden layer size in feed forward network inside transformer
dropout_rate = 0.3 #.3
num_blocks = 2 #2

def architecture_constructor():

    # INPUT EMBEDDING LAYER
    inp = layers.Input(shape=(13,187)) 
    num_embed = inp
    num_embed = layers.Dense(100)(inp)

    inp_na = layers.Input(shape=(13,83)) 
    na_embed = layers.Dense(28)(inp_na)
    
    x = layers.Concatenate()([num_embed, na_embed])
    x = layers.Dense(feat_dim, activation="relu")(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Dropout(.05)(x)
    
    # TRANSFORMER BLOCKS
    for k in range(num_blocks):
        x_old = x
        transformer_block = helpers_seq_tr.TransformerBlock(embed_dim, feat_dim, num_heads, ff_dim, rate=dropout_rate)
        x = transformer_block(x)
        x = .9*x + .1*x_old #.9 .1
    
    # CLASSIFICATION HEAD
    x = layers.Dense(64, activation="relu")(x[:,-1,:])
    x = layers.Dropout(.05)(x) # 64 / .05 best 
    x = layers.Dense(32, activation="relu")(x) #32
    x = layers.Dropout(.05)(x) # 32 / .05 best
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = keras.Model(inputs=[inp, inp_na], outputs=outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001) 
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(loss=loss, optimizer = opt, metrics=[amex_metric_tensorflow])
        
    return model

tf.random.set_seed(CFG_P.seed)

es = EarlyStopping(monitor='val_amex_metric_tensorflow', # val_loss
                   patience=4, 
                   verbose=1,
                   mode="max", #min
                   restore_best_weights=True)

LR_START = .5e-3 # # .5e-3 1e-3
LR_MAX = .5e-3 #1e-3
LR_MIN = .5e-6 # .5e-6 1e-6
LR_RAMPUP_EPOCHS = 0
LR_SUSTAIN_EPOCHS = 0
EPOCHS = 10 #10 

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

callbacks = [LR, es, tf.keras.callbacks.TerminateOnNaN()]

fit_kwargs = {
    'epochs' : 100,
    'verbose' : 2,
    'batch_size' : 1024, #1024 for 1 aug, 512
    'shuffle' : True,
    'callbacks' : callbacks
}

keras_kwargs = {
    'architecture_constructor' : architecture_constructor,
    'fit_kwargs' : fit_kwargs
}

X_train_all = [np.load(CFG_P.output_dir + 'seq_capped_train.npy'),
               np.load(CFG_P.output_dir + 'seq_capped_nas_train.npy')]
train_cus_folds_y = pd.read_parquet(CFG_P.output_dir + 'seq_capped_train_customers.parquet')
train_cus_folds_y = train_cus_folds_y.reset_index(drop=True)
print(train_cus_folds_y.index)
train_cus_folds_y['orig_order'] = train_cus_folds_y.index 

train_cus_folds_y = pd.merge(train_cus_folds_y, pd.read_parquet(CFG_P.output_dir + 'train_labels_w_10_p2_strat_folds.parquet'),
                             on='customer_ID')
train_cus_folds_y = train_cus_folds_y.sort_values(by='orig_order').drop(columns=['orig_order'])                             

X_test = [np.concatenate([np.load(CFG_P.output_dir + 'seq_capped_test_chunk0.npy'),
                          np.load(CFG_P.output_dir + 'seq_capped_test_chunk1.npy')]),
          np.concatenate([np.load(CFG_P.output_dir + 'seq_capped_nas_test_chunk0.npy'),
                          np.load(CFG_P.output_dir + 'seq_capped_nas_test_chunk1.npy')])]               
test_cus = pd.concat([pd.read_parquet(CFG_P.output_dir + 'seq_capped_test_customers_chunk0.parquet'),
                      pd.read_parquet(CFG_P.output_dir + 'seq_capped_test_customers_chunk1.parquet')]).reset_index(drop=True)                         

# Train data augmentation: series shift to exclude last statement
# Only perform for customers that meet threshold of # of statements history
AUG_THRESH = 2

non_aug_mask = X_train_all[0][:,-AUG_THRESH,-2] == 1 # use missing statement flag 
print(non_aug_mask.shape)

# Train data augmentation: series shift to exclude last statement
for i in range(len(X_train_all)):
    aug = X_train_all[i][~non_aug_mask][:,:-1,:]
    if i == 0:
        aug[:,:,-1] -= 1 # adjust months ago column for shifted data

    pad = np.full(X_train_all[i][~non_aug_mask][:,:1,:].shape, MISSING_NULL_VALUE, dtype=int) 
    aug = np.concatenate([pad, X_train_all[i][~non_aug_mask][:,:-1,:]], axis=1)
   
    if i == 0:
        aug[:,0,-1] = 12 # correct padded months ago column
        aug[:,0,-2] = 1 # correct padded is missing statement column 

    X_train_all[i] = np.concatenate([X_train_all[i], aug]) 

    del aug, pad
    gc.collect()

train_cus_folds_y = pd.concat([train_cus_folds_y, train_cus_folds_y[~non_aug_mask]]).reset_index(drop=True) # can add aug2
print(f'Augmented Shape: {train_cus_folds_y.shape}')

def get_scale_reshape(X):

    return X[:,:,:].reshape(-1, X[:,:,:].shape[-1]) 

for i in range(len(X_train_all)):
    scaler = StandardScaler()
    scaler.fit(np.concatenate([get_scale_reshape(X_train_all[i]),
                               get_scale_reshape(X_test[i])]))
    
    X_train_all[i] = scaler.transform(get_scale_reshape(X_train_all[i])).reshape(X_train_all[i].shape)                          
    X_test[i] = scaler.transform(get_scale_reshape(X_test[i])).reshape(X_test[i].shape)  

helpers_seq_tr.train_save_seq_model(X_train_all, train_cus_folds_y, 
                                    X_test, test_cus,
                                    helpers_seq_tr.get_seq_keras_model, keras_kwargs,
                                    'keras_transformer_na_branch', runs_per_fold=4,
                                    list_input=True) 