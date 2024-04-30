import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import torch
from sklearn import datasets, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
#from sklearn.manifold import TSNE
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report, confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split

import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm.autonotebook import tqdm

from model import Neural_Net
from dataset_samplers import RandomCorruptSampler, ClassCorruptSampler, SupervisedSampler 
from corruption_mask_generators import RandomMaskGenerator, CorrelationMaskGenerator
from training import train_contrastive_loss, train_classification
from utils import *

import warnings
warnings.filterwarnings('ignore')
print("Disabled warnings!")

print(f"Using DEVICE: {DEVICE}")

ALL_DIDS = [11, 14, 15, 16, 18, 22, 
            23, 29, 31, 37, 50, 54, 
            188, 458, 469, 1049, 1050, 1063, 
            1068, 1462, 1464, 1480, 1494, 1510,    
            6332, 23381, 40966, 40975, 40982, 40994]


if __name__ == "__main__":   
    # OpenML dataset
    all_datasets = load_openml_list(ALL_DIDS)

    for ds in all_datasets:
        dataset_name, dataset_did, n_classes, n_cat_feats_before_processing, n_feats_before_processing, data, target = ds
        print(f"Loaded dataset: {dataset_name} ({dataset_did}), with data shape: {data.shape} (including {n_cat_feats_before_processing} categorical features), and target shape: {target.shape}")
        assert len(data.select_dtypes(include='category').columns)==n_cat_feats_before_processing 
        accuracies, aurocs = {}, {}
        for key in ALL_METHODS:
            accuracies[key], aurocs[key] = [], []       

        # run each experiment multiple times with varying seeds
        for seed in SEEDS:
            fix_seed(seed)
            # train, test splits
            train_data, test_data, train_targets, test_targets = train_test_split(
                data, target, test_size=0.2, stratify=target, random_state=seed)
            assert np.all(np.unique(train_targets).sort() == np.unique(test_targets).sort())
            assert n_classes == len(np.unique(train_targets))

            # preprocess datasets
            train_data, test_data = preprocess_datasets(train_data, test_data, normalize_numerical_features=True)
            one_hot_encoder = fit_one_hot_encoder(preprocessing.OneHotEncoder(handle_unknown='ignore', \
                                                                              drop='if_binary',   \
                                                                              sparse_output=False), \
                                                  train_data)
            label_encoder_target = preprocessing.LabelEncoder()
            train_targets = label_encoder_target.fit_transform(train_targets)
            test_targets = label_encoder_target.transform(test_targets)
            print(f"Class distributions:")
            unique, counts = np.unique(train_targets, return_counts=True)
            print(f"Training: cls: {unique}; counts: {counts}")
            unique, counts = np.unique(test_targets, return_counts=True)
            print(f"Testing: cls: {unique}; counts: {counts}")
            
            # separate out labeled subset
            n_train_samples_labeled = int(len(train_data)*FRACTION_LABELED)
            idxes_tmp = np.random.permutation(len(train_data))[:n_train_samples_labeled]
            mask_train_labeled = np.zeros(len(train_data), dtype=bool)
            mask_train_labeled[idxes_tmp] = True
            supervised_sampler = SupervisedSampler(data=train_data[mask_train_labeled], target=train_targets[mask_train_labeled])

            # train xgboost to learn predicting one feature based on the rest
            # use learned feature importance to identify feature correlations
            feat_impt, feat_impt_range = compute_feature_mutual_influences(train_data)

            # prepare models
            models, contrastive_loss_histories, supervised_loss_histories = {}, {}, {}
            for method in ALL_METHODS:
                models[method] = nn.DataParallel(Neural_Net(
                    input_dim=one_hot_encoder.transform(train_data).shape[1],  # model expect one-hot encoded input
                    emb_dim=256,
                    output_dim=n_classes   
                )).to(DEVICE)

            # Firstly, train the supervised learning model on the labeled subset
            # freeze the supervised learning encoder as initialized
            models['no_pretrain'].module.freeze_encoder()  
            print("Supervised training for no_pretrain...")
            train_losses = train_classification(models['no_pretrain'], supervised_sampler, one_hot_encoder)
            supervised_loss_histories['no_pretrain'] = train_losses

            ############# Prepare data samplers for corruption ############
            contrastive_samplers = {}
            # Random Sampling: Ignore class information in original corruption
            contrastive_samplers['rand_corr'] = RandomCorruptSampler(train_data) 
            # Oracle Class Sampling: Use oracle info on training labels
            contrastive_samplers['orc_corr'] = ClassCorruptSampler(train_data, train_targets) 
            # Predicted Class Sampling: Use supervised model to obtain pseudo labels at the beginning
            bootstrapped_train_targets = get_bootstrapped_targets( \
                train_data, train_targets, models['no_pretrain'], mask_train_labeled, one_hot_encoder)
            contrastive_samplers['cls_corr'] = ClassCorruptSampler(train_data, bootstrapped_train_targets) 

            ################ Prepare feature selections for masking #############
            # prepare mask generator
            mask_generators = {}
            mask_generators['rand_feats'] = RandomMaskGenerator(train_data.shape[1])
            mask_generators['leastRela_feats'] = CorrelationMaskGenerator(train_data.shape[1], high_correlation=False)
            mask_generators['mostRela_feats'] = CorrelationMaskGenerator(train_data.shape[1], high_correlation=True)
            mask_generators['leastRela_feats'].initialize_feature_importances(feat_impt)
            mask_generators['mostRela_feats'].initialize_feature_importances(feat_impt)


            for method in ALL_METHODS:
                if method == "no_pretrain":
                    continue
                assert '-' in method
                corrupt_method, corrupt_loc = method.split('-')
                # Contrastive training
                train_losses = train_contrastive_loss(models[method], 
                                                      method,
                                                      contrastive_samplers[corrupt_method],
                                                      supervised_sampler,
                                                      mask_generators[corrupt_loc],
                                                      mask_train_labeled,
                                                      one_hot_encoder)
                contrastive_loss_histories[method] = train_losses

                # fine tune the pre-trained models on the down-stream supervised learning task
                models[method].module.freeze_encoder()
                print(f"Supervised fine-tuning for {method}...")
                train_losses = train_classification(models[method], supervised_sampler, one_hot_encoder)
                supervised_loss_histories[method] = train_losses

            # Evaluation on prediction accuracies and aucs
            for method in ALL_METHODS:
                models[method].module.eval()
                with torch.no_grad():
                    test_prediction_logits_normalized = models[method].module.get_classification_prediction_logits( \
                                        torch.tensor(one_hot_encoder.transform(test_data), dtype=torch.float32).to(DEVICE)).softmax(dim=1).cpu().numpy()
                    if n_classes == 2:
                        auroc = roc_auc_score(y_true=test_targets, y_score=test_prediction_logits_normalized[:,1])
                    else:
                        auroc = roc_auc_score(y_true=test_targets, y_score=test_prediction_logits_normalized, multi_class='ovr')
                    test_predictions = np.argmax(test_prediction_logits_normalized,axis=1)
                    accuracy = np.mean(test_predictions==test_targets)*100
                    accuracies[method].append(accuracy)
                    aurocs[method].append(auroc)
                    print(f"{method} accuracy: {accuracy:.2f}%; auroc: {auroc:.2f}")

        # same all the trial accuracy results to numpy file  
        os.makedirs(os.path.join(RESULT_DIR, f"DID_{dataset_did}"), exist_ok=True) 
        for method in ALL_METHODS:
            np.save(os.path.join(RESULT_DIR, f"DID_{dataset_did}", f"{method}_accuracies.npy"), accuracies[method])  
            np.save(os.path.join(RESULT_DIR, f"DID_{dataset_did}", f"{method}_aurocs.npy"), aurocs[method])  

        # write the dataset specifications and experiment hyperparameters into a file
        spec_file = os.path.join(RESULT_DIR, f"DID_{dataset_did}", "experimentSpecs.txt")
        with open(spec_file, "w") as res_f:
            print(f"Writing experimental specs into the file {spec_file}...")
            res_f.write(f"Experiment specs: Corruption rate: {CORRUPTION_RATE}; " +
                        f"Fraction of data labeled: {FRACTION_LABELED}; " +  
                        f"Number of seeds: {len(SEEDS)}; " + 
                        f"Contrastive learning epochs: {CONTRASTIVE_LEARNING_MAX_EPOCHS}; " + 
                        f"Supervised learning epochs: {SUPERVISED_LEARNING_MAX_EPOCHS}.\n" +
                        f"Whether randomized sampling for correlated features sampling: {CORRELATED_FEATURES_RANDOMIZE_SAMPLING}. " + 
                        f"Correlated features sampling softmax temperature: {CORRELATED_FEATURES_RANDOMIZE_SAMPLING_TEMPERATURE}.\n" + 
                        f"{dataset_name} ({dataset_did}) with {n_classes} cls, {n_feats_before_processing} feats ({n_cat_feats_before_processing} categorical), " +
                        f"feature importance range {feat_impt_range:.2f}\n") 
        
        print(f"<<<<<<<<<<<<<<<{dataset_name} finished!>>>>>>>>>>>>>>")

    print("Main function finished!") 
