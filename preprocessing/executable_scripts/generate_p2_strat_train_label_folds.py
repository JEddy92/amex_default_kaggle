import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from preprocessing.config_preproc import PreprocConfig as CFG

# Credit to Vladislav Bogorod
# https://www.kaggle.com/code/bogorodvo/cv-bayesianoptimization-p-2-lst-stratification
def get_P_2_buckets(df, nfolds=5):
    """
    CV stabilization trick 2.
    Create buckets to stratify train set by P_2 | TARGET values.
    Help to reduce noise on hold-out CV.
    """
    
    df = df.sort_values(by='P_2', ascending=False) \
           .reset_index().rename({'index':'row_id'}, axis=1)

    buckets = np.zeros(df.shape[0])

    p0, p1, ind = 0, 0, 0
    for i in range(df.shape[0]):
        buckets[i] = ind
        p0 += np.int8(df.loc[i, 'target'] == 0)
        p1 += np.int8(df.loc[i, 'target'] == 1)
        if p0 >= nfolds and p1 >= nfolds:
            ind += 1
            p0, p1 = 0, 0

    df.loc[:, 'bucket_id'] = buckets

    df.loc[df.loc[:, 'P_2'].isnull(), 'bucket_id'] = -1
    df.loc[df.loc[:, 'bucket_id'] == 0, 'bucket_id'] = 1
    df.loc[df.loc[:, 'bucket_id'] == np.max(df.loc[:, 'bucket_id']), 'bucket_id'] = np.max(df.loc[:, 'bucket_id']) - 1

    df = df.sort_values(by='row_id', ascending=True).reset_index(drop=True)
    
    return np.int64(df.loc[:, 'bucket_id'] + df.loc[:, 'target']*10**6) 

df_train_labels = pd.read_csv(CFG.train_label_file)
df_train_p2 = pd.read_parquet(CFG.train_feature_file, columns=['customer_ID','P_2']) 
df_train_p2 = df_train_p2.groupby('customer_ID').last()

df_train_labels = pd.merge(df_train_labels, df_train_p2, on='customer_ID')

df_train_labels['val_fold_n'] = 0

bucket_id = get_P_2_buckets(df_train_labels, nfolds = 10) 
kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = CFG.seed)

# bucket_id = get_P_2_buckets(df_train_labels) 
# kfold = StratifiedKFold(n_splits = CFG.n_folds, shuffle = True, random_state = CFG.seed)

for fold, (_, val_ind) in enumerate(kfold.split(bucket_id, bucket_id)):
    df_train_labels['val_fold_n'].iloc[val_ind] = fold

# df_train_labels.drop(columns=['P_2']) \
#                .to_parquet(CFG.output_dir + 'train_labels_w_p2_strat_folds.parquet')

df_train_labels.drop(columns=['P_2']) \
               .to_parquet(CFG.output_dir + 'train_labels_w_10_p2_strat_folds.parquet')