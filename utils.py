import random
import numpy as np
import os
import pandas as pd
import torch
import openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import make_column_transformer
import xgboost as xgb
import time
from torch.optim import Adam

# Global parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!!!")
RESULT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'experiments')
os.makedirs(RESULT_DIR, exist_ok=True)
CONTRASTIVE_LEARNING_MAX_EPOCHS = 500
SUPERVISED_LEARNING_MAX_EPOCHS = 100
CLS_CORR_REFRESH_SAMPLER_PERIOD = 10
FRACTION_LABELED = 0.3
CORRUPTION_RATE = 0.4
BATCH_SIZE = 256
SEEDS = [614579, 336466, 974761, 450967, 743562, 843198, 502837, 328984]
assert len(SEEDS) == len(set(SEEDS))
# All the methods to experiment
ALL_METHODS = ['no_pretrain', 'rand_corr-rand_feats', 'cls_corr-rand_feats', 'orc_corr-rand_feats', 'cls_corr-leastRela_feats', 'cls_corr-mostRela_feats']  
P_VAL_SIGNIFICANCE = 0.05
CORRELATED_FEATURES_RANDOMIZE_SAMPLING = True
CORRELATED_FEATURES_RANDOMIZE_SAMPLING_TEMPERATURE = 0.25

# Result processing metric
METRIC = "accuracy"
# METRIC = "auroc"

XGB_FEATURECORR_CONFIG = {
    "n_estimators": 100, 
    "max_depth": 10, 
    "eta": 0.1, 
    "subsample": 0.7, 
    "colsample_bytree": 0.8,
    "enable_categorical": True,
    "tree_method": "hist"
}

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_openml_list(DIDS):
    datasets = []
    datasets_list = openml.datasets.list_datasets(DIDS, output_format='dataframe')

    for ds in datasets_list.index:
        entry = datasets_list.loc[ds]

        print('Loading', entry['name'], entry.did, '..')

        if entry['NumberOfClasses'] == 0.0:
            raise Exception("Regression not supported for now")
            exit(1)    
        else:
            dataset = openml.datasets.get_dataset(int(entry.did))
            # since under SCARF corruption, the replacement by sampling happens before one-hot encoding, load the 
            # data in its original form
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                dataset_format="dataframe", target=dataset.default_target_attribute
            )
            
            assert isinstance(X, pd.DataFrame) and isinstance(y, pd.Series)        

            order = np.arange(y.shape[0])
            # Don't think need to re-seed here
            np.random.shuffle(order)
            X, y = X.iloc[order], y.iloc[order]

            assert X is not None

        datasets += [[entry['name'], 
                      entry.did, 
                      int(entry['NumberOfClasses']), 
                      np.sum(categorical_indicator), 
                      len(X.columns), 
                      X, 
                      y]]

    return datasets

def preprocess_datasets(train_data, test_data, normalize_numerical_features):
    assert isinstance(train_data, pd.DataFrame) and \
             isinstance(test_data, pd.DataFrame)
    assert np.all(train_data.columns == test_data.columns)
    features_dropped = []
    for col in train_data.columns:
        # drop columns with all null values or with a constant value on training data
        if train_data[col].isnull().all() or train_data[col].nunique() == 1:
            train_data.drop(columns=col, inplace=True)
            test_data.drop(columns=col, inplace=True)
            features_dropped.append(col)
            continue
        # fill the missing values
        if train_data[col].isnull().any() or test_data[col].isnull().any():
            # for categorical features, fill with the mode in the training data
            if train_data[col].dtype.name == "category":
                val_fill = train_data[col].mode(dropna=True)[0]
            # for numerical features, fill with the mean of the training data
            else:
                val_fill = train_data[col].mean(skipna=True)
            train_data[col].fillna(val_fill, inplace=True)
            test_data[col].fillna(val_fill, inplace=True)
   
    if normalize_numerical_features:
        # z-score transform numerical values
        scaler = StandardScaler()
        non_categorical_cols = train_data.select_dtypes(exclude='category').columns
        if len(non_categorical_cols) == 0:
            print("No numerical features presen! Skip numerical z-score normalization.")
        else:
            train_data[non_categorical_cols] = scaler.fit_transform(train_data[non_categorical_cols])
            test_data[non_categorical_cols] = scaler.transform(test_data[non_categorical_cols])   

    print(f"Data preprocess finished! Dropped {len(features_dropped)} features: {features_dropped}. {'Normalized numerical features.' if normalize_numerical_features else ''}")
    
    # retain the pandas dataframe format for later one-hot encoder
    return train_data, test_data

def fit_one_hot_encoder(one_hot_encoder_raw, train_data):
    categorical_cols = train_data.select_dtypes(include='category').columns
    one_hot_encoder = make_column_transformer((one_hot_encoder_raw, categorical_cols), remainder='passthrough')
    one_hot_encoder.fit(train_data)
    return one_hot_encoder

def get_bootstrapped_targets(data, targets, classifier_model, mask_labeled, one_hot_encoder):
    # use the classifier to predict for all data first
    classifier_model.module.eval()
    with torch.no_grad():
        pred_logits = classifier_model.module.get_classification_prediction_logits(
                            torch.tensor(one_hot_encoder.transform(data).astype(float), dtype=torch.float32).to(DEVICE)).cpu().numpy()
    preds = np.argmax(pred_logits, axis=1)
    return np.where(mask_labeled, targets, preds)

# expect a pandas dataframe
# fit xgboost models on pandas dataframe and series
def compute_feature_mutual_influences(data):
    assert isinstance(data, pd.DataFrame)
    label_encoder_tmp = LabelEncoder()
    feat_impt = []
    start_time = time.time()
    feat_impt_range_avg = 0
    for i, col in enumerate(data.columns):
        if data[col].dtype == "category":
            xgb_model = xgb.XGBClassifier(**XGB_FEATURECORR_CONFIG)
            target = label_encoder_tmp.fit_transform(data[col])
        else:
            xgb_model = xgb.XGBRegressor(**XGB_FEATURECORR_CONFIG)
            target = data[col]
        xgb_model.fit(data.drop(col, axis=1), target)
        # the xgb_obj.feature_importances_ is the normalized score for gain
        feat_impt_range_avg += np.ptp(xgb_model.feature_importances_)
        feat_impt.append(np.insert(xgb_model.feature_importances_, obj=i, values=0))
    feat_impt = np.array(feat_impt)
    feat_impt_range_avg = feat_impt_range_avg/len(data.columns)
    print(f"Feature importances computated for {len(data)} samples each with {np.shape(data)[1]} features! Took {time.time()-start_time:.2f} seconds. The average range is {feat_impt_range_avg}")
    return feat_impt, feat_impt_range_avg

def initialize_adam_optimizer(model):
    return Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=0.001)
