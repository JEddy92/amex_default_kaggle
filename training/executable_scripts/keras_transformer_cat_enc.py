import gc
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from scipy.stats import yeojohnson

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

# From Ryota
def auto_log_transform(df, th=5):
    """
    全部をチェックするのは難しい時
    歪度が一定以上であれば、log変換を行う
    """

    log_cols = []
    log_reverse_cols = []

    for col in tqdm(df.columns):
        try:
            skew_org = df[col].skew()
            if skew_org > 0:
                skew_log = np.log1p(df[col] - df[col].min()).skew()
                diff = np.abs(skew_org) - np.abs(skew_log)
                if diff > th:
                    log_cols.append(col)
            else:
                skew_log = np.log1p(-1 * df[col] + df[col].max()).skew()
                diff = np.abs(skew_org) - np.abs(skew_log)
                if diff > th:
                    log_reverse_cols.append(col) 
        except:
            pass

    for col in tqdm(log_cols):
        df[col] = np.log1p(df[col] - df[col].min())

    for col in tqdm(log_reverse_cols):
        df[col] = np.log1p(-1 * df[col] + df[col].max())

    return df

# From Ryota
# numpy version
def auto_log_transform_np(arr, th=5):
    """
    全部をチェックするのは難しい時
    歪度が一定以上であれば、log変換を行う
    """

    log_cols = []
    log_reverse_cols = []

    for col in range(arr.shape[1]):
        try:
            skew_org = np.skew(arr[:,col])
            if skew_org > 0:
                skew_log = np.log1p(arr[:,col] - np.min(arr[:,col])).skew()
                diff = np.abs(skew_org) - np.abs(skew_log)
                if diff > th:
                    log_cols.append(col)
            else:
                skew_log = np.log1p(-1 * arr[:,col] + np.max(arr[:,col])).skew()
                diff = np.abs(skew_org) - np.abs(skew_log)
                if diff > th:
                    log_reverse_cols.append(col) 
        except:
            pass

    for col in log_cols:
        arr[:,col] = np.log1p(arr[:,col] - np.min(arr[:,col]))

    for col in log_reverse_cols:
        arr[:,col] = np.log1p(-1 * arr[:,col] + np.max(arr[:,col]))

    return arr

# BEST: 400 impute version, standard setup with light dropout,
# 1024 batch size 10 epochs .5 learning rate. No masking
# Total out of fold exact Amex val score is 0.7928852093014731
# Average out of fold exact Amex val score is 0.7929292586403969
# All fold scores:
# [0.7902481994419164, 0.7956358518879043, 0.7902771516608931, 0.7960578031044196, 
# 0.787970054964523, 0.7903297313103965, 0.7932372974138766, 0.7996534665107811,
#  0.7896976959987885, 0.7961853341104688]

# indicator_cols = ['months_ago_valid_max', 'customer_sequence_num', 
#                   'pre_history','missing_statement','months_ago']

ALPHA = 5
GAMMA = 2
def FocalLoss(targets, inputs, alpha=ALPHA, gamma=GAMMA):  
    targets = tf.cast(targets, tf.float32)  
    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
    
    return focal_loss

# TO DO: PUT IN CFG
MISSING_NULL_VALUE = -3 #-3

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
    inp = layers.Input(shape=(13,190)) #adding 1 for days ago

    # TIME EMBEDDING
    # time2vec = helpers_seq_tr.Time2Vec(kernel_size=1)
    # time_embedding = keras.layers.TimeDistributed(time2vec)(inp[:,:,-3:-2])

    # inp = layers.Masking(mask_value=MISSING_NULL_VALUE)(inp) 
    # inp = layers.Input(shape=(13,187))
    # x_last = inp[:,-1,:]
    # x_last = layers.Dense(64, activation='relu')(x_last)
    x = layers.Dense(feat_dim)(inp)
    # x = layers.Concatenate()([x, time_embedding])
    # x = layers.SpatialDropout1D(.05)(x)
    x = layers.Dropout(.05)(x) # .05 best
    # x = layers.BatchNormalization()(x)
    
    # TRANSFORMER BLOCKS
    for k in range(num_blocks):
        x_old = x
        transformer_block = helpers_seq_tr.TransformerBlock(embed_dim, feat_dim, num_heads, ff_dim, rate=dropout_rate)
        x = transformer_block(x)
        x = .9*x + .1*x_old #.9 .1
    
    # CLASSIFICATION HEAD
    # x = layers.LSTM(32, recurrent_dropout=.30)(x)
    # x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="gelu")(x[:,-1,:])
    # x = layers.Dense(64, activation="gelu")(x)
    x = layers.Dropout(.05)(x) # 64 / .05 best 
    x = layers.Dense(32, activation="gelu")(x) #32
    x = layers.Dropout(.05)(x) # 32 / .05 best
    # x = layers.BatchNormalization()(x)
    #x = layers.Dropout(.30)(x)
    # x = layers.concatenate([x, x_last])
    # x = layers.Dense(32, activation="relu")(x) #32
    #x = layers.Dropout(.30)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = keras.Model(inputs=inp, outputs=outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001) 
    # opt = AdaBeliefOptimizer(learning_rate=0.02, #.02 
    #                          weight_decay = 1e-5,
    #                          epsilon = 1e-7,
    #                          print_change_log = False,
    #                         )
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(loss=loss, optimizer = opt, metrics=[amex_metric_tensorflow])
        
    return model

tf.random.set_seed(CFG_P.seed)
    
# lr = ReduceLROnPlateau(monitor="val_loss", factor=0.7, 
#                        patience=4, verbose=2)

lr = ReduceLROnPlateau(monitor="val_loss", 
                       factor=0.3, 
                       patience=4, #5 
                       mode = 'min', 
                       verbose=1)

es = EarlyStopping(monitor='val_loss', # val_amex_metric_tensorflow, val_loss
                   patience=3, 
                   verbose=1,
                   mode="min", #min
                   restore_best_weights=True)

# current best
# LR_START = .5e-3 # # .5e-3 1e-3
# LR_MAX = .5e-3 #1e-3
# LR_MIN = .5e-6 # .5e-6 1e-6
# LR_RAMPUP_EPOCHS = 0
# LR_SUSTAIN_EPOCHS = 0
# EPOCHS = 10 #10 

LR_START = .5e-3 # # .5e-3 1e-3
LR_MAX = .5e-3 #1e-3
LR_MIN = .5e-6 # .5e-6 1e-6
LR_RAMPUP_EPOCHS = 0
LR_SUSTAIN_EPOCHS = 0
EPOCHS = 12 #10

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

callbacks = [LR, tf.keras.callbacks.TerminateOnNaN()]

fit_kwargs = {
    'epochs' : 12,
    'verbose' : 2,
    'batch_size' : 1024, #1024 for 1 aug, 512
    'shuffle' : True,
    'callbacks' : callbacks
}

keras_kwargs = {
    'architecture_constructor' : architecture_constructor,
    'fit_kwargs' : fit_kwargs
}

X_train_all = np.load(CFG_P.output_dir + 'seq_capped_train.npy')
train_cus_folds_y = pd.read_parquet(CFG_P.output_dir + 'seq_capped_train_customers.parquet')
train_cus_folds_y = train_cus_folds_y.reset_index(drop=True)
print(train_cus_folds_y.index)
train_cus_folds_y['orig_order'] = train_cus_folds_y.index 

train_cus_folds_y = pd.merge(train_cus_folds_y, pd.read_parquet(CFG_P.output_dir + 'train_labels_w_10_p2_strat_folds.parquet'),
                             on='customer_ID')
train_cus_folds_y = train_cus_folds_y.sort_values(by='orig_order').drop(columns=['orig_order'])                             

X_test = np.concatenate([np.load(CFG_P.output_dir + 'seq_capped_test_chunk0.npy'),
                         np.load(CFG_P.output_dir + 'seq_capped_test_chunk1.npy')])
test_cus = pd.concat([pd.read_parquet(CFG_P.output_dir + 'seq_capped_test_customers_chunk0.parquet'),
                      pd.read_parquet(CFG_P.output_dir + 'seq_capped_test_customers_chunk1.parquet')]).reset_index(drop=True)                         

print(X_train_all.shape, train_cus_folds_y.shape, X_test.shape)

def get_scale_reshape(X):

    return X[:,:,:].reshape(-1, X[:,:,:].shape[-1]) 

# Scaling should be done with missing statement values masked
# tr_mask = np.any(X_train_all == MISSING_NULL_VALUE, axis=-1)
# te_mask = np.any(X_test == MISSING_NULL_VALUE, axis=-1)

# tr_mask = X_train_all[:,:,-2] == 1
# te_mask = X_test[:,:,-2] == 1

# print(X_train_all[~tr_mask].shape)

# tr_ind = X_train_all.shape[0]
# X_comb = np.concatenate([get_scale_reshape(X_train_all),
#                          get_scale_reshape(X_test)])

# for i in range(X_comb.shape[-1]):
#     print(i)
#     X_comb[:,i], _ = yeojohnson(X_comb[:,i])

# X_comb = X_comb.reshape((-1, X_train_all.shape[1], X_train_all.shape[-1]))  

# X_train_all = X_comb[:tr_ind,:,:]
# X_test = X_comb[tr_ind:,:,:]

# del X_comb
# gc.collect()

# scaler = PowerTransformer()
scaler = StandardScaler()
# scaler.fit(np.concatenate([X_train_all[~tr_mask],
#                            X_test[~te_mask]]))

# scaler.fit(np.concatenate([get_scale_reshape(X_train_all),
#                            get_scale_reshape(X_test)]))

# Power transformation to make data more gaussian
# This seems to be extremely slow
# print('Power transforming')
# tr_ind = X_train_all.shape[0]
# X_comb = np.concatenate([X_train_all, X_test])
# mask = np.any(X_comb == MISSING_NULL_VALUE, axis=-1)

# X_comb[~mask] = power_transform(X_comb[~mask])
# X_train_all = X_comb[:tr_ind,:,:]
# X_test = X_comb[tr_ind:,:,:]

# del X_comb, mask, tr_ind
# gc.collect()

# scaler = RobustScaler()
# scaler.fit(np.concatenate([get_scale_reshape(X_train_all[~tr_mask]),
#                            get_scale_reshape(X_test[~te_mask])]))

# Train data augmentation: series shift to exclude last statement
# Only perform for customers that meet threshold of # of statements history
AUG_THRESH = 2 
# AUG_THRESH_2 = 10 

# non_aug_mask = np.any(X_train_all[:,-AUG_THRESH,:] == MISSING_NULL_VALUE, axis=-1) 
non_aug_mask = ((X_train_all[:,:,-2].sum(axis=1) + X_train_all[:,:,-3].sum(axis=1)) \
                > (CFG_P.max_n_statement - AUG_THRESH))
# non_aug_mask = X_train_all[:,-AUG_THRESH,-2] == 1 # use missing statement flag 
# non_aug_mask2 = X_train_all[:,-AUG_THRESH_2,-2] == 1 
print(non_aug_mask.shape)

aug = X_train_all[~non_aug_mask][:,:-1,:]
aug[:,:,-1] -= 1 # adjust months ago column for shifted data
aug[:,:,-5] -= 1 # adjust months ago valid max column for shifted data

# aug2 = X_train_all[~non_aug_mask2][:,:-2,:] 

pad = np.full(aug[:,:1,:].shape, MISSING_NULL_VALUE, dtype=int)
aug = np.concatenate([pad, aug], axis=1)

# confirm that pad statements are fully nulled out for correct masking
# print((aug[:,0,:] != MISSING_NULL_VALUE).sum())

"""
# indicator_cols = ['months_ago_valid_max', customer_sequence_num', 
#                   'pre_history','missing_statement','months_ago']
"""

aug[:,0,-1] = 12 # correct padded months ago column
aug[:,0,-2] = 0 # correct padded is missing statement column 
aug[:,0,-3] = 1 # correct padded pre history column 
aug[:,0,-4] = aug[:,1,-4] - 1 # correct padded customer sequence number column 
aug[:,0,-5] = aug[:,1,-5] # correct padded months ago valid max column 

# pad2 = np.full(aug2[:,:2,:].shape, MISSING_NULL_VALUE, dtype=int)
# aug2 = np.concatenate([pad2, aug2], axis=1)
# aug2[:,:,-1] -= 2 # adjust months ago column for shifted data
# aug2[:,0,-1] = 12 # correct padded months ago columns
# aug2[:,1,-1] = 11
# aug2[:,0,-2] = 1 # correct padded is missing statement column 
# aug2[:,1,-2] = 1

X_train_all = np.concatenate([X_train_all, aug]) # can add aug2

# # mask out all missing statements
# X_train_all[X_train_all[:,:,-2] == 1] = MISSING_NULL_VALUE
# X_test[X_test[:,:,-2] == 1] = MISSING_NULL_VALUE

train_cus_folds_y = pd.concat([train_cus_folds_y, train_cus_folds_y[~non_aug_mask]]).reset_index(drop=True) # can add aug2
print(f'Augmented Shape: {X_train_all.shape}, {train_cus_folds_y.shape}')
print(f"Augmented Target 1s: {(train_cus_folds_y['target'] == 1).sum()}")

del aug, pad #aug/pad2
gc.collect()

# recompute missing statement masks to handle augmentation
# tr_mask = np.any(X_train_all == MISSING_NULL_VALUE, axis=-1)
# te_mask = np.any(X_test == MISSING_NULL_VALUE, axis=-1)
# tr_mask = X_train_all[:,:,-2] == 1
# te_mask = X_test[:,:,-2] == 1

tr_cut = X_train_all.shape[0] 
X_proc_comb = np.concatenate([X_train_all, X_test])
comb_shape = X_proc_comb.shape

X_proc_comb = auto_log_transform_np(get_scale_reshape(X_proc_comb))
X_proc_comb = scaler.fit_transform(X_proc_comb).reshape(comb_shape)

X_train_all = X_proc_comb[:tr_cut,:,:]
X_test = X_proc_comb[tr_cut:,:,:] 

del X_proc_comb
gc.collect()

# X_train_all = scaler.transform(get_scale_reshape(X_train_all)).reshape(X_train_all.shape)                          
# X_test = scaler.transform(get_scale_reshape(X_test)).reshape(X_test.shape) 
# X_train_all[~tr_mask] = scaler.transform(X_train_all[~tr_mask])                         
# X_test[~te_mask] = scaler.transform(X_test[~te_mask])

print(X_train_all.shape, X_test.shape)

helpers_seq_tr.train_save_seq_model(X_train_all, train_cus_folds_y, 
                                    X_test, test_cus,
                                    helpers_seq_tr.get_seq_keras_model, keras_kwargs,
                                    'keras_transformer_of_bag', runs_per_fold=4) 