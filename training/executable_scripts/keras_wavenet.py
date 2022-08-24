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

# TO DO: PUT IN CFG
MISSING_NULL_VALUE = -3 #-3
MISSING_CAT_VALUE = 0

n_filters = 100 # 32 
filter_width = 2
dilation_rates = [2**i for i in range(4)] * 2

embeds = [(4,1),
          (8,2),
          (3,1),
          (3,1),
          (8,2),
          (3,1),
          (4,1),
          (6,2),
          (5,2),
          (3,1),
          (8,2),
          ]

def architecture_constructor():

    # INPUT EMBEDDING LAYER
    inp = layers.Input(shape=(13,188))
 
    embeddings = []
    for k in range(11):
        emb = layers.Embedding(embeds[k][0] + 1, embeds[k][1]) # extra dim for missing cat value
        embeddings.append( emb(inp[:,:,k]) )
    
    x = layers.Concatenate()([inp[:,:,11:]] + embeddings)
    x = layers.Dropout(.05)(x)

    skips = []
    for dilation_rate in dilation_rates:
        
        # preprocessing - equivalent to time-distributed dense
        x = layers.Conv1D(n_filters, 1, padding='same', activation='gelu')(x) 
        x = layers.Dropout(.3)(x)
        
        # filter convolution
        x_f = layers.Conv1D(filters=n_filters,
                            kernel_size=filter_width, 
                            padding='causal',
                            dilation_rate=dilation_rate)(x)
        
        # gating convolution
        x_g = layers.Conv1D(filters=n_filters,
                            kernel_size=filter_width, 
                            padding='causal',
                            dilation_rate=dilation_rate)(x)
        
        # multiply filter and gating branches
        z = layers.Multiply()([layers.Activation('tanh')(x_f),
                               layers.Activation('sigmoid')(x_g)])
        
        # postprocessing - equivalent to time-distributed dense
        z = layers.Conv1D(n_filters, 1, padding='same', activation='gelu')(z)

        # residual connection
        x = layers.Add()([x, z])    
        
        # collect skip connections
        skips.append(z)

    # add all skip connection outputs 
    # out = layers.Activation('gelu')(layers.Add()(skips))
    out = x

    # final time-distributed dense layers 
    out = layers.Conv1D(n_filters, 1, padding='same')(out)
    out = layers.Activation('gelu')(out)
    out = layers.Dropout(.05)(out)
    out = layers.Flatten()(out)
    out = layers.Dense(64, activation='gelu')(out)
    out = layers.Dropout(.05)(out)
    outputs = layers.Dense(1, activation='sigmoid')(out)
    
    model = keras.Model(inputs=inp, outputs=outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=0.01) 
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(loss=loss, optimizer = opt, metrics=[amex_metric_tensorflow])
        
    return model

tf.random.set_seed(CFG_P.seed)

lr = ReduceLROnPlateau(monitor="val_loss", 
                       factor=0.7, 
                       patience=2, #5 
                       mode = 'min', 
                       verbose=1)

es = EarlyStopping(monitor='val_amex_metric_tensorflow', # val_loss
                   patience=6, 
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

callbacks = [lr, es, tf.keras.callbacks.TerminateOnNaN()]

fit_kwargs = {
    'epochs' : 100,
    'verbose' : 2,
    'batch_size' : 2048, #1024 for 1 aug, 512
    'shuffle' : True,
    'callbacks' : callbacks
}

keras_kwargs = {
    'architecture_constructor' : architecture_constructor,
    'fit_kwargs' : fit_kwargs
}

X_train_all = np.load(CFG_P.output_dir + 'seq_capped_train.npy')
print(X_train_all[:5,:,:11])
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

scaler.fit(np.concatenate([get_scale_reshape(X_train_all[:,:,11:]),
                           get_scale_reshape(X_test[:,:,11:])]))

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
# AUG_THRESH = 2 

# non_aug_mask = X_train_all[:,-AUG_THRESH,-2] == 1 # use missing statement flag 
# print(non_aug_mask.shape)

# aug = X_train_all[~non_aug_mask][:,:-1,:]
# aug[:,:,-1] -= 1 # adjust months ago column for shifted data
# aug[:,:,-3] -= np.repeat(aug[:,-1:,-3], repeats=12, axis=1) # adjust days ago column for shifted data

# pad = np.full(aug[:,:1,:].shape, MISSING_NULL_VALUE, dtype=int)
# pad[:,:,:11] = MISSING_CAT_VALUE
# aug = np.concatenate([pad, aug], axis=1)

# aug[:,0,-1] = 12 # correct padded months ago column (days ago consistently is nulled out)
# aug[:,0,-2] = 1 # correct padded is missing statement column 

# X_train_all = np.concatenate([X_train_all, aug]) # can add aug2

# train_cus_folds_y = pd.concat([train_cus_folds_y, train_cus_folds_y[~non_aug_mask]]).reset_index(drop=True) # can add aug2
print(f'Augmented Shape: {X_train_all.shape}, {train_cus_folds_y.shape}')

#del aug, pad #aug/pad2
#gc.collect()

X_train_all[:,:,11:] = scaler.transform(get_scale_reshape(X_train_all[:,:,11:])).reshape(X_train_all[:,:,11:].shape)                          
X_test[:,:,11:] = scaler.transform(get_scale_reshape(X_test[:,:,11:])).reshape(X_test[:,:,11:].shape) 

print(X_train_all.shape, X_test.shape)

helpers_seq_tr.train_save_seq_model(X_train_all, train_cus_folds_y, 
                                    X_test, test_cus,
                                    helpers_seq_tr.get_seq_keras_model, keras_kwargs,
                                    'keras_wavenet', runs_per_fold=4) 