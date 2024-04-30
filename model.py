import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class MLP(torch.nn.Sequential):
    """Simple multi-layer perceptron with ReLu activation and optional dropout layer"""

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout_prob):
        layers = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(torch.nn.Dropout(dropout_prob))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, output_dim))

        super().__init__(*layers)


class Neural_Net(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim,
        output_dim
    ):
        """Implementation of SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption.
        It consists in an encoder that learns the embeddings.
        It is done by minimizing the contrastive loss of a sample and a corrupted view of it.
        The corrupted view is built by remplacing a random set of features by another sample randomly drawn independently.

            Args:
                input_dim (int): size of the inputs
                emb_dim (int): dimension of the embedding space
                encoder_depth (int, optional): number of layers of the encoder MLP. Defaults to 4.
                head_depth (int, optional): number of layers of the pretraining head. Defaults to 2.
                corruption_rate (float, optional): fraction of features to corrupt. Defaults to 0.6.
                encoder (nn.Module, optional): encoder network to build the embeddings. Defaults to None.
                pretraining_head (nn.Module, optional): pretraining head for the training procedure. Defaults to None.
        """
        super().__init__()

        self.encoder_depth = 4
        self.pretrain_head_depth = 2
        self.classification_head_depth = 2
        self.contrastive_loss_temperature = 1.0
        self.dropout_prob = 0.0

        self.encoder = MLP(input_dim, emb_dim, emb_dim, self.encoder_depth, self.dropout_prob)
        self.pretraining_head = MLP(emb_dim, emb_dim, emb_dim, self.pretrain_head_depth, self.dropout_prob)
        self.classification_head = MLP(emb_dim, emb_dim, output_dim, self.classification_head_depth, self.dropout_prob)

        # initialize weights
        self.encoder.apply(self._init_weights)
        self.pretraining_head.apply(self._init_weights)
        # also need classification head re-initialization from outside
        self.initialize_classification_head() 

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    
    def get_middle_embedding(self, input_batch):
        emb_batch = self.encoder(input_batch)

        return emb_batch
    
    def get_final_embedding(self, input_batch):
        # compute middle embeddings first
        emb_batch = self.get_middle_embedding(input_batch)
        emb_batch = self.pretraining_head(emb_batch)

        return emb_batch
    
    def get_classification_prediction_logits(self, input_batch):
        # compute middle embeddings first
        emb_batch = self.get_middle_embedding(input_batch)
        # With pytorch's cross-entropy loss, only logits are required from the neural net
        predictions_batch = self.classification_head(emb_batch)

        return predictions_batch

    def freeze_encoder(self):
        self.encoder.requires_grad_(False)
        return

    def unfreeze_encoder(self):
        self.encoder.requires_grad_(True)
        return
    
    def initialize_classification_head(self):
        self.classification_head.apply(self._init_weights)
        return

    def contrastive_loss(self, z_i, z_j):
        """
        NT-Xent loss for contrastive learning using cosine distance as similarity metric as used in [SimCLR](https://arxiv.org/abs/2002.05709).
        Implementation adapted from https://theaisummer.com/simclr/#simclr-loss-implementation

        hyper-parameter: temperature (float, optional): scaling factor of the similarity metric. Defaults to 1.0.
        
        Compute NT-Xent loss using only anchor and positive batches of samples. Negative samples are the 2*(N-1) samples in the batch

        Args:
            z_i (torch.tensor): anchor batch of samples
            z_j (torch.tensor): positive batch of samples

        Returns:
            float: loss
        """
        batch_size_here = z_i.size(0) # account for last incomplete batch sampled

        # compute similarity between the sample's embedding and its corrupted view
        z = torch.cat([z_i, z_j], dim=0)
        similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity, batch_size_here)
        sim_ji = torch.diag(similarity, -batch_size_here)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        mask = (~torch.eye(batch_size_here * 2, batch_size_here * 2, dtype=torch.bool)).float().to(DEVICE)
        numerator = torch.exp(positives / self.contrastive_loss_temperature)
        denominator = mask * torch.exp(similarity / self.contrastive_loss_temperature)

        all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size_here)

        return loss

    def classification_loss(self, pred_logits, targets):
        return F.cross_entropy(input=pred_logits, target=targets, reduction='mean')
    

    
