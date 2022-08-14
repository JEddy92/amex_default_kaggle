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
from adabelief_tf import AdaBeliefOptimizer

from preprocessing.config_preproc import PreprocConfig as CFG_P
from training import helpers_flat_training as helpers_flat_tr
from training import helpers_seq_training as helpers_seq_tr
from utils.evaluation import amex_metric_tensorflow

ALPHA= 5
GAMMA = 2
def FocalLoss(targets, inputs, alpha=ALPHA, gamma=GAMMA):  
    targets = tf.cast(targets, tf.float32)  
    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
    
    return focal_loss

# TO DO: PUT IN CFG
MISSING_NULL_VALUE = -3

# credit to Chris Deotte for base model architecture and params
# https://www.kaggle.com/code/cdeotte/tensorflow-transformer-0-790
feat_dim = 100 #128
embed_dim = 25  # 32 Embedding size for attention
num_heads = 6  #4  Number of attention heads
ff_dim = 100  # 128 Hidden layer size in feed forward network inside transformer
dropout_rate = 0.3 #.3
num_blocks = 2 #2

def architecture_constructor():

    # INPUT EMBEDDING LAYER
    inp = layers.Input(shape=(13,163))
    inp = layers.Masking(mask_value=MISSING_NULL_VALUE)(inp) 
    # inp = layers.Input(shape=(13,187))
    # x_last = inp[:,-1,:]
    # x_last = layers.Dense(64, activation='relu')(x_last)
    x = layers.Dense(feat_dim)(inp)
    # x = layers.Dropout(.10)(x)
    # x = layers.BatchNormalization()(x)
    
    # TRANSFORMER BLOCKS
    for k in range(num_blocks):
        x_old = x
        transformer_block = helpers_seq_tr.TransformerBlock(embed_dim, feat_dim, num_heads, ff_dim, rate=dropout_rate)
        x = transformer_block(x)
        x = .9*x + .1*x_old #.9 .1
    
    # CLASSIFICATION HEAD
    # 128 1 layer no dropout
    # x = layers.Dense(128, activation="relu")(x[:,-1,:])
    x = layers.Dense(64, activation="relu")(x[:,-1,:])
    x = layers.Dropout(.05)(x) 
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(.05)(x)
    # x = layers.BatchNormalization()(x)
    #x = layers.Dropout(.30)(x)
    # x = layers.concatenate([x, x_last])
    # x = layers.Dense(32, activation="relu")(x) #32
    #x = layers.Dropout(.30)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = keras.Model(inputs=inp, outputs=outputs)
    # opt = tf.keras.optimizers.Adam(learning_rate=0.001) 
    opt = AdaBeliefOptimizer(learning_rate=0.02, 
                             weight_decay = 1e-5,
                             epsilon = 1e-7,
                             print_change_log = False,
                            )
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(loss=FocalLoss, optimizer = opt, metrics=[amex_metric_tensorflow])
        
    return model

tf.random.set_seed(CFG_P.seed)
    
# lr = ReduceLROnPlateau(monitor="val_loss", factor=0.7, 
#                        patience=4, verbose=2)

lr = ReduceLROnPlateau(monitor="val_loss", 
                       factor=0.3, 
                       patience=5, 
                       mode = 'min', 
                       verbose=1)

es = EarlyStopping(monitor='val_amex_metric_tensorflow', # val_loss
                   patience=5, 
                   verbose=1,
                   mode="max", #min
                   restore_best_weights=True)

LR_START = 1e-3 # # .5e-3 1e-3
LR_MAX = 1e-3 #1e-3
LR_MIN = 1e-6 # .5e-6 1e-6
LR_RAMPUP_EPOCHS = 0
LR_SUSTAIN_EPOCHS = 0
EPOCHS = 8 #8 

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

callbacks = [lr, es, tf.keras.callbacks.TerminateOnNaN()]

fit_kwargs = {
    'epochs' : 100,
    'verbose' : 2,
    'batch_size' : 2048, #1024, 512
    'shuffle' : True,
    'callbacks' : callbacks
}

keras_kwargs = {
    'architecture_constructor' : architecture_constructor,
    'fit_kwargs' : fit_kwargs
}

X_train_all = np.load(CFG_P.output_dir + 'seq_v2_train.npy')
train_cus_folds_y = pd.read_parquet(CFG_P.output_dir + 'seq_v2_train_customers.parquet')
train_cus_folds_y = train_cus_folds_y.reset_index(drop=True)
print(train_cus_folds_y.index)
train_cus_folds_y['orig_order'] = train_cus_folds_y.index 

train_cus_folds_y = pd.merge(train_cus_folds_y, pd.read_parquet(CFG_P.output_dir + 'train_labels_w_10_p2_strat_folds.parquet'),
                             on='customer_ID')
train_cus_folds_y = train_cus_folds_y.sort_values(by='orig_order').drop(columns=['orig_order'])                             

X_test = np.concatenate([np.load(CFG_P.output_dir + 'seq_v2_test_chunk0.npy'),
                         np.load(CFG_P.output_dir + 'seq_v2_test_chunk1.npy')])
test_cus = pd.concat([pd.read_parquet(CFG_P.output_dir + 'seq_v2_test_customers_chunk0.parquet'),
                      pd.read_parquet(CFG_P.output_dir + 'seq_v2_test_customers_chunk1.parquet')]).reset_index(drop=True)                         

print(X_train_all.shape, train_cus_folds_y.shape, X_test.shape)

def get_scale_reshape(X):

    return X[:,:,:].reshape(-1, X[:,:,:].shape[-1]) 

# Scaling should be done with missing statement values masked
tr_mask = np.any(X_train_all == MISSING_NULL_VALUE, axis=-1)
te_mask = np.any(X_test == MISSING_NULL_VALUE, axis=-1)
print(tr_mask.shape)
print(tr_mask.sum())

scaler = StandardScaler()
scaler.fit(np.concatenate([X_train_all[~tr_mask],
                           X_test[~te_mask]]))
# scaler = RobustScaler()
# scaler.fit(np.concatenate([get_scale_reshape(X_train_all[~tr_mask]),
#                            get_scale_reshape(X_test[~te_mask])]))

# Train data augmentation: series shift to exclude last statement
# Only perform for customers that meet threshold of # of statements history
AUG_THRESH = 2 

non_aug_mask = np.any(X_train_all[:,-AUG_THRESH,:] == MISSING_NULL_VALUE, axis=-1) 
print(non_aug_mask.shape)

aug = X_train_all[~non_aug_mask][:,:-1,:]

pad = np.full(aug[:,:1,:].shape, MISSING_NULL_VALUE, dtype=int)
aug = np.concatenate([pad, aug], axis=1)
aug[:,:,-1] -= 1 # adjust months ago column for shifted data

X_train_all = np.concatenate([X_train_all, aug])
train_cus_folds_y = pd.concat([train_cus_folds_y, train_cus_folds_y[~non_aug_mask]]).reset_index(drop=True)
print(f'Augmented Shape: {X_train_all.shape}, {train_cus_folds_y.shape}')

del aug
gc.collect()

# recompute missing statement masks to handle augmentation
tr_mask = np.any(X_train_all == MISSING_NULL_VALUE, axis=-1)
te_mask = np.any(X_test == MISSING_NULL_VALUE, axis=-1)

# X_train_all[~tr_mask] = scaler.transform(get_scale_reshape(X_train_all[~tr_mask])).reshape(X_train_all[~tr_mask].shape)                          
# X_test[~te_mask] = scaler.transform(get_scale_reshape(X_test[~te_mask])).reshape(X_test.shape) 
X_train_all[~tr_mask] = scaler.transform(X_train_all[~tr_mask])                         
X_test[~te_mask] = scaler.transform(X_test[~te_mask])

print(X_train_all.shape, X_test.shape)

helpers_seq_tr.train_save_seq_model(X_train_all, train_cus_folds_y, 
                                    X_test, test_cus,
                                    helpers_seq_tr.get_seq_keras_model, keras_kwargs,
                                    'keras_transformer_full_na_imputes_mask', runs_per_fold=4) 