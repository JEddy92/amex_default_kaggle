import gc
import numpy as np
import pandas as pd 

from preprocessing.helpers_preproc import get_agg_features
from preprocessing.config_preproc import PreprocConfig as CFG  
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  

cat_features = CFG.cat_features 
use_cols = ['customer_ID'] + cat_features 
print('Reading and concatenating data')

df_train = pd.read_parquet(CFG.train_feature_file, columns=use_cols)
df_train['dset'] = 'train'

df_test = pd.read_parquet(CFG.test_feature_file, columns=use_cols)
df_test['dset'] = 'test'

# concatenation to extract additional count information from test,
# may be relevant to tf-idf calculations
df_comb = pd.concat([df_train, df_test]).reset_index(drop=True)
print(f'Combined shape: {df_comb.shape}')
cus_dset = df_comb[['customer_ID','dset']].drop_duplicates(subset=['customer_ID'])

del df_train, df_test
gc.collect()

print("Aggregating category histories into 'documents'")

df_comb = df_comb.astype('str')
cat_doc_func = [('doc', lambda x: ' '.join(x))]
df_comb = get_agg_features(df_in=df_comb, group_col='customer_ID', 
                           agg_features=cat_features, agg_funcs=cat_doc_func)

print(df_comb.head(10))

print("Count and tf-idf vectorizing category 'documents'")
df_outs = []

for c in [c + '_doc' for c in cat_features]:
    print(f"Running count extractors on {c} 'documents'")
    cv, tfidf = CountVectorizer(analyzer='char'), TfidfVectorizer(analyzer='char')
    cv_out = cv.fit_transform(df_comb[c])
    tfidf_out = tfidf.fit_transform(df_comb[c]) 

    cv_out = pd.DataFrame(cv_out.todense(), columns=[f'{c}_cv_{i}' for i in range(cv_out.shape[1])])
    tfidf_out = pd.DataFrame(tfidf_out.todense(), columns=[f'{c}_tfidf_{i}' for i in range(tfidf_out.shape[1])])
    
    df_out = pd.concat([cv_out, tfidf_out], axis=1)
    print(df_out.head(10), df_out.shape)
    df_outs.append(df_out)

df_out = pd.concat(df_outs, axis=1)
del df_outs
gc.collect()

df_out['customer_ID'] = df_comb['customer_ID']
df_out = pd.merge(df_out, cus_dset, on='customer_ID')

print(df_out.shape)
print(df_out.head(10))

print(df_out[df_out['dset'] == 'train'].drop(columns='dset').shape)

df_out[df_out['dset'] == 'train'].drop(columns='dset') \
    .to_parquet(CFG.output_dir + 'train_cat_word_count_features.parquet')

df_out[df_out['dset'] == 'test'].drop(columns='dset') \
    .to_parquet(CFG.output_dir + 'test_cat_word_count_features.parquet')