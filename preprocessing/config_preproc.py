class PreprocConfig:

    train_feature_file = '/home/ubuntu/amex_default_kaggle/data/raw/train.parquet'  
    test_feature_file = '/home/ubuntu/amex_default_kaggle/data/raw/test.parquet' 
    train_label_file = '/home/ubuntu/amex_default_kaggle/data/raw/train_labels.csv' 

    output_dir = '/home/ubuntu/amex_default_kaggle/data/extracted_features/' 
    model_output_dir = '/home/ubuntu/amex_default_kaggle/data/model_outputs/'  
    
    max_n_statement = 13
    cat_features = ['B_30', 'B_38', 'D_114', 'D_116', 
                    'D_117', 'D_120', 'D_126', 'D_63', 
                    'D_64', 'D_66', 'D_68']
    non_features = ['customer_ID', 'S_2', 'val_fold_n', 'target']

    seed = 28
    n_folds = 5