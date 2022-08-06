import gc
from typing import Callable
import os
import joblib
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import tensorflow as tf
from tensorflow.keras.models import Model

from preprocessing.config_preproc import PreprocConfig as CFG_P
from utils.evaluation import amex_metric, amex_metric_np_lgb, amex_metric_np_xgb

def load_flat_features(fnames : list, parser : Callable = pd.read_parquet, 
                       is_meta : bool = False) -> pd.DataFrame:
    """Load flat features from parquet files into dataframe

    Args:
        fnames (list): feature files in parquet format: 1 row <-> 1 customer_ID
        is_meta (bool, optional): flag for naming oof preds for meta learning. Defaults to False.

    Returns:
        pd.DataFrame: complete feature dataframe, including labels and val folds if train
    """

    feature_dfs = [parser(f).set_index('customer_ID') for f in fnames]
    feature_df = pd.concat(feature_dfs, axis=1).reset_index()

    if is_meta:
        feature_df.columns = ['customer_ID'] + [f"oof_pred_{fn.split('/')[-2]}" for fn in fnames]

    return feature_df

def filter_out_features(df_in : pd.DataFrame, model_log_file : str,
                        imp_threshold : float = 0) -> pd.DataFrame:

    with open(model_log_file, "r") as read_file:
        feature_imps = json.load(read_file)['Aggregated_FI']
    exclude_features = [f for f, i in feature_imps if i <= imp_threshold]
    exclude_features = [f for f in exclude_features if f in df_in.columns]
    
    return df_in.drop(columns=exclude_features)

def add_round_features_in_place(df_in : pd.DataFrame, str_pattern : str = None, 
                                replace_orig : bool = False):

    dtypes = df_in.dtypes
    num_cols = list(dtypes[(dtypes == 'float32') | (dtypes == 'float64')].index)
    
    if str_pattern:
        num_cols = [col for col in num_cols if str_pattern in col]
    
    for col in num_cols:
        df_in[col + '_round2'] = df_in[col].round(2)

    if replace_orig:
        df_in.drop(columns=num_cols, inplace=True)

def train_save_flat_model(df_train : pd.DataFrame, df_test : pd.DataFrame, 
                          get_model : Callable, model_kwargs : dict, 
                          get_imp : Callable, output_dir_name : str,
                          cat_mode : str = 'label_mode', runs_per_fold : int = 1):

    model_log = {}
    model_log['model_kwargs'] = str(model_kwargs)

    # TO DO: probably a safer/smarter way to handle the category part
    features = [f for f in df_train.columns if f not in CFG_P.non_features]
    cat_features = [c for c in CFG_P.cat_features if c in features]
    cat_features += [f'{c}_last' for c in CFG_P.cat_features if f'{c}_last' in features]
    cat_features += [f'{c}_first' for c in CFG_P.cat_features if f'{c}_first' in features]
    cat_features += [f'{c}_mode' for c in CFG_P.cat_features if f'{c}_mode' in features]
    for i in range(1, CFG_P.max_n_statement+1):
        cat_features += [f'{c}_{i}' for c in CFG_P.cat_features if f'{c}_{i}' in features]

    if cat_mode == 'drop':
        df_train = df_train.drop(columns=cat_features)
        df_test = df_test.drop(columns=cat_features)
        features = [f for f in features if f not in cat_features]
    
    elif cat_mode == 'label_mode': 
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df_train[cat_features] = encoder.fit_transform(df_train[cat_features]).astype(int)
        df_test[cat_features] = encoder.transform(df_test[cat_features]).astype(int)

    elif cat_mode == 'xgb_cat_mode':
        df_train[cat_features] = df_train[cat_features].astype('category')
        df_test[cat_features] = df_test[cat_features].astype('category')

    elif cat_mode == 'ohe':
        ohe = OneHotEncoder(drop='first', sparse=False, dtype=np.float32, handle_unknown='ignore')
        ohe.fit(df_train[cat_features])
        
        df_train_cat = pd.DataFrame(ohe.transform(df_train[cat_features]).astype(np.float16)).rename(columns=str)
        df_train = pd.concat([df_train.drop(columns=cat_features), df_train_cat], axis=1)

        df_test_cat = pd.DataFrame(ohe.transform(df_test[cat_features]).astype(np.float16)).rename(columns=str)
        df_test = pd.concat([df_test.drop(columns=cat_features), df_test_cat], axis=1)

        features = [f for f in df_train.columns if f not in CFG_P.non_features]

        del df_train_cat, df_test_cat
        gc.collect()

    model_log['features'] = features

    oof_preds = df_train[['customer_ID']].drop_duplicates(subset=['customer_ID'], keep='first')
    oof_preds['oof_pred'] = 0

    test_preds = df_test[['customer_ID']]
    test_preds['prediction'] = 0

    FI_series = pd.Series(index = features, data = 0)
    avg_score = 0

    n_folds = df_train['val_fold_n'].nunique() 
    for val_fold in range(n_folds):
        print(f'Training Fold {val_fold}: \n')
        val_mask = df_train['val_fold_n'] == val_fold 

        X_train, X_val = df_train.loc[~val_mask, features], df_train.loc[val_mask]
        y_train = df_train.loc[~val_mask, 'target']
        
        # handle duplicates for multi-statement train aug
        oof_mask = oof_preds['customer_ID'].isin(X_val['customer_ID'].unique())
        X_val = X_val.drop_duplicates(subset=['customer_ID'], keep='first')
        X_val, y_val = X_val[features], X_val['target'] 
        
        for _ in range(runs_per_fold):

            model, predict_func = get_model(X_train, y_train, X_val, y_val, 
                                            cat_features, model_kwargs)

            oof_preds.loc[oof_mask,'oof_pred'] += predict_func(model, X_val) / runs_per_fold

            # TO DO - PORT TO CONFIG
            chunk_size = (df_test.shape[0] // 5) + 1
            for i in range(5): 
                print(f'Predicting test chunk {i}')
                low, high = i * chunk_size, (i + 1) * chunk_size
                test_preds['prediction'].iloc[low:high] \
                    += predict_func(model, df_test[features].iloc[low:high]) / (n_folds * runs_per_fold)
        
            tf.keras.backend.clear_session()

        score = amex_metric(pd.DataFrame({'target' : y_val.values}), 
                            pd.DataFrame({'prediction' : oof_preds.loc[oof_mask,'oof_pred'].values}))
        avg_score += score / n_folds

        print(f'Fold {val_fold} exact Amex val score: {score}')
        model_log[f'Fold {val_fold} OOF'] = score

        del X_train, X_val, y_train, y_val
        gc.collect()
        
        try:
            FI_series += get_imp(model) / CFG_P.n_folds
            # FI_series += model.feature_importance(importance_type='gain') / CFG_P.n_folds
            print(FI_series)
        except:
            pass

        del model
        gc.collect()
    
    tot_score = amex_metric(pd.DataFrame({'target' : df_train.drop_duplicates(subset=['customer_ID'], keep='first')['target'].values}), 
                            pd.DataFrame({'prediction' : oof_preds['oof_pred'].values}))
    print(f'Total out of fold exact Amex val score is {tot_score}')
    print(f'Average out of fold exact Amex val score is {avg_score}')
    print('All fold scores:')
    print([model_log[f'Fold {val_fold} OOF'] for val_fold in range(n_folds)])
    model_log[f'Total OOF score'] = tot_score
    model_log[f'Average OOF score'] = avg_score
    model_log['Aggregated_FI'] = list(FI_series.sort_values(ascending=False).items())
    
    out_path = CFG_P.model_output_dir + f'{output_dir_name}' 
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    oof_preds.to_parquet(out_path + f'/oof_preds.parquet')
    test_preds.to_csv(out_path + f'/test_preds.csv', index=False)

    with open(out_path + "/model_log.json", "w") as outfile:
        json.dump(model_log, outfile)

def get_lgb_model(X_train : pd.DataFrame, y_train : pd.Series, 
                  X_val : pd.DataFrame, y_val : pd.Series,
                  cat_features : list, lgb_kwargs : dict) -> lgb.Booster:

    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature = cat_features)
    lgb_val = lgb.Dataset(X_val, y_val, categorical_feature = cat_features)
    
    model = lgb.train(train_set = lgb_train, valid_sets = [lgb_train, lgb_val],
                      feval = amex_metric_np_lgb, **lgb_kwargs)

    del lgb_train, lgb_val
    gc.collect()

    def predict_func(model, X):
        return model.predict(X)

    return model, predict_func

def get_lgb_imp(model : lgb.Booster) -> np.ndarray:
    return model.feature_importance(importance_type='gain')

# credit to 1110Ra 
# https://www.kaggle.com/competitions/amex-default-prediction/discussion/332575
global max_score
max_score = .5

def get_lgb_dart_callback() -> Callable:

    def callback(env):
        
        global max_score
        iteration = env.iteration
        score = env.evaluation_result_list[3][2]
        
        if iteration % 100 == 0:
                print('iteration {}, score= {:.05f}'.format(iteration,score))
        
        if score > max_score:
            max_score = score
            path = 'temp_dart_model/'
            for fname in os.listdir(path):
                if fname.startswith("temp_model_"):
                    os.remove(os.path.join(path, fname))
            print('High Score: iteration {}, score={:.05f}'.format(iteration, score))
            joblib.dump(env.model, 'temp_dart_model/temp_model_{:.05f}.pkl'.format(score))
    
    callback.order = 0
    return callback

# TO DO: this temp path shouldn't really be a variable the model gets
def get_lgb_dart_model(X_train : pd.DataFrame, y_train : pd.Series, 
                       X_val : pd.DataFrame, y_val : pd.Series,
                       cat_features : list, lgb_kwargs : dict, 
                       temp_model_path : str = 'temp_dart_model'):

    global max_score

    _, predict_func = get_lgb_model(X_train, y_train, X_val, y_val,
                                    cat_features, lgb_kwargs)

    dart_model_file = temp_model_path + '/' + os.listdir(temp_model_path)[0]
    model = joblib.load(dart_model_file)
    os.remove(dart_model_file)
    max_score = .5

    return model, predict_func

def get_xgb_model(X_train : pd.DataFrame, y_train : pd.Series, 
                  X_val : pd.DataFrame, y_val : pd.Series,
                  cat_features : list, xgb_kwargs : dict):

    dtrain = xgb.DMatrix(data = X_train, label = y_train, enable_categorical = True)
    dvalid = xgb.DMatrix(data= X_val, label = y_val, enable_categorical = True)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    model = xgb.train(dtrain = dtrain, evals = watchlist, 
                      feval = amex_metric_np_xgb, maximize = True,
                      **xgb_kwargs)

    del dtrain, dvalid
    gc.collect()

    def predict_func(model, X):
        X = xgb.DMatrix(data = X, enable_categorical = True)
        return model.predict(X, iteration_range=(0, model.best_ntree_limit))

    return model, predict_func

def get_xgb_imp(model : xgb.Booster) -> np.ndarray:
    return np.array(model.get_score(importance_type='weight').values())

def get_catboost_model(X_train : pd.DataFrame, y_train : pd.Series, 
                       X_val : pd.DataFrame, y_val : pd.Series,
                       cat_features : list, catboost_kwargs : dict):
    
    model = CatBoostClassifier(**catboost_kwargs)
    model.fit(X_train, y_train, eval_set = [(X_val, y_val)], 
              cat_features = cat_features, verbose = 100, use_best_model = True)

    def predict_func(model, X):
        return model.predict_proba(X)[:, 1]

    return model, predict_func

def get_catboost_imp(model : lgb.Booster) -> np.ndarray:
    return model.get_feature_importance()

def get_flat_keras_model(X_train : pd.DataFrame, y_train : pd.Series, 
                         X_val : pd.DataFrame, y_val : pd.Series,
                         cat_features : list, keras_kwargs : dict):
    
    model = keras_kwargs['architecture_constructor'](X_train.shape[1])

    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_val = scaler.transform(X_val)

    model.fit(X_train, y_train, 
              validation_data=(X_val, y_val), 
              **keras_kwargs['fit_kwargs'])

    def predict_func(model, X):
        return model.predict(X, verbose=0).flatten()

    return model, predict_func

    # def predict_func(scaler_model, X):
    #     return scaler_model[1].predict(scaler_model[0].transform(X), 
    #                                    verbose=0).flatten()

    # return (scaler, model), predict_func

def get_keras_imp(model : Model) -> int:
    return 0 # not defined

def get_logreg_model(X_train : pd.DataFrame, y_train : pd.Series,
                     X_val : pd.DataFrame, y_val : pd.Series,
                     cat_features : list, logreg_kwargs : dict) -> LogisticRegression:
    
    model = LogisticRegression(**logreg_kwargs)
    model.fit(X_train, y_train)

    def predict_func(model, X):
        return model.predict_proba(X)[:,1]

    return model, predict_func

def get_logreg_imp(model : LogisticRegression) -> np.ndarray:
    return model.coef_[0]

def get_avgrank_model(X_train : pd.DataFrame, y_train : pd.Series,
                      X_val : pd.DataFrame, y_val : pd.Series,
                      cat_features : list, avgrank_kwargs : dict) -> LogisticRegression:
    
    model = 'placeholder'

    def predict_func(model, X):
        return np.mean(rankdata(X, axis=0), axis=1)

    return model, predict_func

def get_avgrank_imp(model : str) -> np.ndarray:
    return 0