import gc
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
from tensorflow.keras.layers import Dense, Input, InputLayer, Add, Concatenate, Dropout, BatchNormalization

from preprocessing.config_preproc import PreprocConfig as CFG_P
from training import helpers_flat_training as helpers_tr
from utils.evaluation import amex_metric_tensorflow

# credit to Ambros M for base model architecture and params
# https://www.kaggle.com/code/ambrosm/amex-keras-quickstart-1-training
def architecture_constructor(n_inputs : int) -> tf.keras.models.Model:
    """Map feature dim to correct architecture keras model output

    Args:
        n_inputs (int): feature dim

    Returns:
        tf.keras.models.Model: compiled model with specified architecture
    """
    
    activation = 'swish' #'swish'
    reg = 4e-4 #4e-4
    
    inputs = Input(shape=(n_inputs, ))
    x0 = Dense(256, 
               activation=activation, kernel_regularizer=tf.keras.regularizers.l2(reg),
             )(inputs)
    x = Dense(64,
              activation=activation, kernel_regularizer=tf.keras.regularizers.l2(reg),
             )(x0)
    x = Dropout(0.35)(x)
    x = Dense(64,
              activation=activation, kernel_regularizer=tf.keras.regularizers.l2(reg),
             )(x)
    x = Concatenate()([x, x0])
    x = Dropout(0.35)(x)
    
    x = Dense(16, 
              activation=activation, kernel_regularizer=tf.keras.regularizers.l2(reg),
             )(x)
    x = Dropout(0.10)(x)
    x = Dense(1, 
              activation='sigmoid',
             )(x)
    model = Model(inputs, x)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[amex_metric_tensorflow])
    return model

tf.random.set_seed(CFG_P.seed)
    
lr = ReduceLROnPlateau(monitor="val_loss", factor=0.7, 
                       patience=4, verbose=2)

es = EarlyStopping(monitor='val_amex_metric_tensorflow',
                   patience=12, 
                   verbose=1,
                   mode="max", 
                   restore_best_weights=True)

callbacks = [lr, es, tf.keras.callbacks.TerminateOnNaN()]

fit_kwargs = {
    'epochs' : 200,
    'verbose' : 2,
    'batch_size' : 2048,
    'shuffle' : True,
    'callbacks' : callbacks
}

keras_kwargs = {
    'architecture_constructor' : architecture_constructor,
    'fit_kwargs' : fit_kwargs
}

feature_fnames = ['{}_NA_agg_features.parquet',
                  '{}_refined_agg_features.parquet',
                  '{}_cat_p2_encoded_features.parquet',
                  #'{}_refined_agg_features_med_3_7.parquet',
                  #'{}_cat_word_count_features.parquet',
                  #'{}_date_features.parquet',
                  #'{}_diff_agg_features.parquet',
                  '{}_diff2_features.parquet']
                  #'{}_cat_time_svd_features.parquet']           
tr_feature_files = [CFG_P.output_dir + f.format('train') for f in feature_fnames]
te_feature_files = [CFG_P.output_dir + f.format('test') for f in feature_fnames]

# df_train = helpers_tr.load_flat_features(tr_feature_files)
# df_train = pd.merge(df_train, pd.read_parquet(CFG_P.output_dir + 'train_labels_w_p2_strat_folds.parquet'),
#                     on='customer_ID')

df_train = helpers_tr.load_flat_features(tr_feature_files)
df_train = pd.merge(df_train, pd.read_parquet(CFG_P.output_dir + 'train_labels_w_10_p2_strat_folds.parquet'),
                    on='customer_ID')
df_train = helpers_tr.filter_out_features(df_train, 
                                          CFG_P.model_output_dir + 'lgbm_dart_big_cat_p2_na_streak/model_log.json',
                                          imp_threshold = 4000) # lgbm_dart_big_cat_word_p2_strat for v4
df_train = df_train.fillna(-1) # 0 for v4

df_test = helpers_tr.load_flat_features(te_feature_files)
df_test = helpers_tr.filter_out_features(df_test, 
                                         CFG_P.model_output_dir + 'lgbm_dart_big_cat_p2_na_streak/model_log.json',
                                         imp_threshold = 4000)
df_test = df_test.fillna(-1) # 0 for v4

print(df_train.shape, df_test.shape)

features = [f for f in df_train.columns if f not in CFG_P.non_features]

scaler = StandardScaler()
print('partial scaler fitting train')
scaler.partial_fit(df_train[features].iloc[:(df_train.shape[0] // 2)])
scaler.partial_fit(df_train[features].iloc[(df_train.shape[0] // 2):])

print('partial scaler fitting test')
chunk_size = (df_test.shape[0] // 4) + 1
for i in range(4): 
    print(f' test chunk {i}')
    low, high = i * chunk_size, (i + 1) * chunk_size
    scaler.partial_fit(df_test[features].iloc[low:high])

print('Transforming data')
df_test[features] = scaler.transform(df_test[features])
gc.collect()
df_train[features] = scaler.transform(df_train[features])
gc.collect()

helpers_tr.train_save_flat_model(df_train, df_test,
                                 helpers_tr.get_flat_keras_model, keras_kwargs,
                                 helpers_tr.get_keras_imp, 'keras_flat_base_v5_10F_bags',
                                 cat_mode='ohe', runs_per_fold=3) # 5 for v4