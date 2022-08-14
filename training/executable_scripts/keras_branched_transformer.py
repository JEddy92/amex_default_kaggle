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

MISSING_NULL_VALUE = -3

# credit to Chris Deotte for base model architecture and params
# https://www.kaggle.com/code/cdeotte/tensorflow-transformer-0-790

branch_names = ['D_','S_','P_','B_','R_','Other']

inp_embs = [(94, 40), #30
            (21, 12), #8
            (3, 2), #1
            (39, 20), #14
            (28, 14), #10
            (2, 2)] #1

feat_dim = 90 #128
embed_dim = 32  # 32 Embedding size for attention
num_heads = 4  #4  Number of attention heads
ff_dim = 90  # 128 Hidden layer size in feed forward network inside transformer
dropout_rate = 0 #.3
num_blocks = 2 #2

def architecture_constructor():
    
    # BRANCHED INPUT EMBEDDINGS
    inps, embs = [], []
    for inp_emb in inp_embs:
        inp = layers.Input(shape=(13, inp_emb[0]))  
        emb = layers.Dense(inp_emb[1])(inp)

        inps.append(inp)
        embs.append(emb)
    
    x = layers.Concatenate()(embs)
    x = layers.Dense(feat_dim, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # TRANSFORMER BLOCKS
    for k in range(num_blocks):
        x_old = x
        transformer_block = helpers_seq_tr.TransformerBlock(embed_dim, feat_dim, num_heads, ff_dim, rate=dropout_rate)
        x = transformer_block(x)
        x = .75*x + .25*x_old #.9 .1
    
    # CLASSIFICATION HEAD
    x = layers.Dense(90, activation="relu")(x[:,-1,:]) #128
    x = layers.BatchNormalization()(x)
    x = layers.Dense(45, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = keras.Model(inputs=inps, outputs=outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001) 
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(loss=loss, optimizer = opt, metrics=[amex_metric_tensorflow])
        
    return model

tf.random.set_seed(CFG_P.seed)

es = EarlyStopping(monitor='val_amex_metric_tensorflow',
                   patience=4, 
                   verbose=1,
                   mode="max", 
                   restore_best_weights=True)

LR_START = 1e-3 #1e-3
LR_MAX = 1e-3 #1e-3
LR_MIN = 1e-6 #1e-6
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

callbacks = [LR, es, tf.keras.callbacks.TerminateOnNaN()]

fit_kwargs = {
    'epochs' : 100,
    'verbose' : 2,
    'batch_size' : 512, #512
    'shuffle' : True,
    'callbacks' : callbacks
}

keras_kwargs = {
    'architecture_constructor' : architecture_constructor,
    'fit_kwargs' : fit_kwargs
}

X_train_all = np.load(CFG_P.output_dir + 'seq_branched_train.npy')
X_train_all = [X_train_all[name] for name in branch_names]
train_cus_folds_y = pd.read_parquet(CFG_P.output_dir + 'seq_branched_train_customers.parquet')
train_cus_folds_y = train_cus_folds_y.reset_index(drop=True)
train_cus_folds_y['orig_order'] = train_cus_folds_y.index 

train_cus_folds_y = pd.merge(train_cus_folds_y, pd.read_parquet(CFG_P.output_dir + 'train_labels_w_10_p2_strat_folds.parquet'),
                             on='customer_ID')
train_cus_folds_y = train_cus_folds_y.sort_values(by='orig_order').drop(columns=['orig_order'])                             

X_test_0 = np.load(CFG_P.output_dir + 'seq_branched_train_chunk0.npy')
X_test_1 = np.load(CFG_P.output_dir + 'seq_branched_train_chunk1.npy')

X_test = []
for name in branch_names: 
    X_test.append(np.concatenate([X_test_0[name], X_test_1[name]]))

del X_test_0, X_test_1
gc.collect() 

test_cus = pd.concat([pd.read_parquet(CFG_P.output_dir + 'seq_branched_test_customers_chunk0.parquet'),
                      pd.read_parquet(CFG_P.output_dir + 'seq_branched_test_customers_chunk1.parquet')]).reset_index(drop=True)                         

# Train data augmentation: series shift to exclude last statement
for i in range(len(X_train_all)):
    pad = np.full(X_train_all[i][:,:1,:].shape, MISSING_NULL_VALUE, dtype=int) 
    aug = np.concatenate([pad, X_train_all[i][:,:-1,:]], axis=1)

    if i == len(X_train_all) - 1:
        aug[:,:,-1] -= 1 # adjust months ago column for shifted data     
    X_train_all[i] = np.concatenate([X_train_all[i], aug]) 

train_cus_folds_y = pd.concat([train_cus_folds_y, train_cus_folds_y]).reset_index(drop=True)

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
                                    'keras_transformer_branched', runs_per_fold=3,
                                    list_input=True) 