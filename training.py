import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from dataset_samplers import ClassCorruptSampler
from utils import *

def _train_contrastive_loss_oneEpoch(model, 
                                     data_sampler, 
                                     mask_generator, 
                                     optimizer,
                                     one_hot_encoder):
    model.train()
    epoch_loss = 0
    for _ in range(data_sampler.n_batches):
        anchors, random_samples = data_sampler.sample_batch()
        # firstly, corrupt on the original pandas dataframe
        corruption_masks = mask_generator.get_masks(np.shape(anchors)[0])
        assert np.shape(anchors) == np.shape(corruption_masks)
        anchors_corrupted = np.where(corruption_masks, random_samples, anchors)
        # after corruption, do one-hot encoding
        anchors, anchors_corrupted = one_hot_encoder.transform(pd.DataFrame(data=anchors,columns=data_sampler.columns)), \
                                        one_hot_encoder.transform(pd.DataFrame(data=anchors_corrupted,columns=data_sampler.columns))

        anchors, anchors_corrupted = torch.tensor(anchors.astype(float), dtype=torch.float32).to(DEVICE), \
                                        torch.tensor(anchors_corrupted.astype(float), dtype=torch.float32).to(DEVICE)

        # reset gradients
        optimizer.zero_grad()

        # get embeddings
        emb_final_anchors = model.module.get_final_embedding(anchors)
        emb_final_corrupted = model.module.get_final_embedding(anchors_corrupted)

        # compute loss
        loss = model.module.contrastive_loss(emb_final_anchors, emb_final_corrupted)
        loss.backward()

        # update model weights
        optimizer.step()

        # log progress
        epoch_loss += loss.item()

    return epoch_loss / data_sampler.n_batches


def train_contrastive_loss(model, 
                           method_key, 
                           contrastive_sampler, 
                           supervised_sampler, 
                           mask_generator, 
                           mask_train_labeled, 
                           one_hot_encoder):
    print(f"Contrastive learning for {method_key}....")
    train_losses = []
    optimizer = initialize_adam_optimizer(model)
    
    for i in tqdm(range(1, CONTRASTIVE_LEARNING_MAX_EPOCHS+1)):
        if i%CLS_CORR_REFRESH_SAMPLER_PERIOD == 0 and 'cls_corr' in method_key:
            model.module.freeze_encoder()
            # train the current model on down-stream supervised task
            _ = train_classification(model, supervised_sampler, one_hot_encoder)
            # use the current model to do pseudo labeling
            bootstrapped_train_targets = get_bootstrapped_targets( 
                                            pd.DataFrame(data=contrastive_sampler.data, columns=contrastive_sampler.columns), 
                                            contrastive_sampler.target, 
                                            model, 
                                            mask_train_labeled, 
                                            one_hot_encoder)
            # get the class based sampler based on more reliable model predictions
            contrastive_sampler = ClassCorruptSampler(pd.DataFrame(data=contrastive_sampler.data, columns=contrastive_sampler.columns), 
                                                      bootstrapped_train_targets) 
            model.module.unfreeze_encoder()
        
        epoch_loss = _train_contrastive_loss_oneEpoch(model, 
                                                      contrastive_sampler, 
                                                      mask_generator, 
                                                      optimizer, 
                                                      one_hot_encoder)
        train_losses.append(epoch_loss)

    return train_losses

def train_classification(model, supervised_sampler, one_hot_encoder):
    train_losses = []
    optimizer = initialize_adam_optimizer(model)
    model.module.initialize_classification_head()

    for _ in range(SUPERVISED_LEARNING_MAX_EPOCHS):
        model.module.train()
        epoch_loss = 0.0
        for _ in range(supervised_sampler.n_batches):
            inputs, targets = supervised_sampler.sample_batch()

            inputs = one_hot_encoder.transform(pd.DataFrame(data=inputs, columns=supervised_sampler.columns))
            inputs = torch.tensor(inputs.astype(float), dtype=torch.float32).to(DEVICE)
            # seemingly int64 is often used as the type for indices
            targets = torch.tensor(targets.astype(int), dtype=torch.int64).to(DEVICE)

            # reset gradients
            optimizer.zero_grad()

            # get classification predictions
            pred_logits = model.module.get_classification_prediction_logits(inputs)

            # compute loss
            loss = model.module.classification_loss(pred_logits, targets)
            loss.backward()

            # update model weights
            optimizer.step()

            # log progress
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / supervised_sampler.n_batches)

    return train_losses
 