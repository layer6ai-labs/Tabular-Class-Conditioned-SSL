import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from utils import *

class BasicSampler():
    def __init__(self, data, target):
        assert isinstance(data, pd.DataFrame)
        self.data = np.array(data, dtype='object')
        self.columns = data.columns
        self.target = target
        self.n_samples = np.shape(self.data)[0]
        self.n_features = np.shape(self.data)[1]
        self.n_batches = int(np.ceil(self.n_samples/BATCH_SIZE))
        self.batch_end_pointer = 0
        assert len(self.columns) == self.n_features
        if self.target is not None:
            assert self.target.ndim == 1, "expecting targets to be in a row vector."

    def __len__(self):
        return np.shape(self.data)[0]

    def __str__(self):
        return f"A BasicSampler object, with datasize {len(self)}."

    def _initialize_epoch(self):
        perm = np.random.permutation(len(self))
        self.data = self.data[perm]
        # shuffle the target to align with data for class-based sampler
        if isinstance(self.target, np.ndarray):
            self.target = self.target[perm]
        self.batch_end_pointer = 0
        return

    # To be overwritten by subclasses for its own needs
    def _get_one_sample_pair(self):
        raise NotImplementedError
    
    def sample_batch(self):
        data_batch_1, data_batch_2 = [], []
        new_batch_end_pointer = min(self.batch_end_pointer + BATCH_SIZE, self.n_samples)
        for i in range(self.batch_end_pointer, new_batch_end_pointer):
            data_1, data_2 = self._get_one_sample_pair(i)
            data_batch_1.append(data_1)
            data_batch_2.append(data_2)
        if new_batch_end_pointer == self.n_samples:
            self._initialize_epoch()
        else:
            self.batch_end_pointer = new_batch_end_pointer
    
        return np.array(data_batch_1, dtype='object'), \
                np.array(data_batch_2, dtype='object')
    
    # sample from each column of the data independently
    # achieve uniform value sampling from each feature
    def _sample_columns_iid(self, data):
        assert data.shape[1] == self.n_features
        res = []
        for i in range(self.n_features):
            res.append(np.random.choice(data[:, i]))        
        return np.array(res)
    
    def get_data(self):
        return self.data
    
    @property
    def shape(self):
        return self.data.shape


# As a base class with random corruption, does not need targets
class RandomCorruptSampler(BasicSampler):
    def __init__(self, data, target=None):
        super().__init__(data, target)        

    def __str__(self):
        return f"A RandomCorruptSampler object, with datasize {len(self)}."
                             
    def _get_one_sample_pair(self, index):
        # the dataset must return a pair of samples: the anchor and a randomly composed one (feature-wise independently)
        # from the dataset that will be used to corrupt the anchor
        anchor = self.data[index]

        # randomly sample from each column independently to compose a row for corruption
        corrupt_src = self._sample_columns_iid(self.data)

        return anchor, corrupt_src
    


# Can be used with both predicted classes: bootstrapping from semi-supervised learning;
# or with oracle class labels
class ClassCorruptSampler(RandomCorruptSampler):
    def __init__(self, data, target):
        super().__init__(data, target)

    def __str__(self):
        return f"A ClassCorruptSampler object, with datasize {len(self)}."

    # Modification: the sample used to corrupt the anchor has to be from the same class
    def _get_one_sample_pair(self, index):
        anchor = self.data[index]

        # prune the table to keep only rows with the same class id as the anchor
        candidate_idxes = np.where(self.target == self.target[index])[0]
        
        corrupt_src = self._sample_columns_iid(self.data[candidate_idxes])
        
        return anchor, corrupt_src

    
# Used for supervised learning
class SupervisedSampler(BasicSampler):
    def __init__(self, data, target):
        super().__init__(data, target)

    def __str__(self):
        return f"A SupervisedSampler object, with datasize {len(self)}."
    
    # Supervised learning setting: sample a (data, target) pair
    def _get_one_sample_pair(self, index):
        data_single = self.data[index]
        target_single = self.target[index]
        return data_single, target_single
    