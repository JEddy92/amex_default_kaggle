import pandas as pd
from sklearn.model_selection import StratifiedKFold

from preprocessing.config_preproc import PreprocConfig as CFG

df_train_labels = pd.read_csv(CFG.train_label_file)
df_train_labels['val_fold_n'] = 0
y = df_train_labels['target'] 
kfold = StratifiedKFold(n_splits = CFG.n_folds, shuffle = True, random_state = CFG.seed)

for fold, (_, val_ind) in enumerate(kfold.split(y, y)):
    df_train_labels['val_fold_n'].iloc[val_ind] = fold

df_train_labels.to_parquet(CFG.output_dir + 'train_labels_w_folds.parquet')