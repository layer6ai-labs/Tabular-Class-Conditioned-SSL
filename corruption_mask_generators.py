import numpy as np
from sklearn.cluster import SpectralClustering
from utils import *

class RandomMaskGenerator():
    def __init__(self, n_features):
        self.n_features = n_features
        self.corruption_len = int(np.ceil(CORRUPTION_RATE * self.n_features))
        assert self.corruption_len < self.n_features
    
    def get_masks(self, n_samples):
        corruption_masks = np.zeros((n_samples, self.n_features), dtype=bool)
        for i in range(n_samples):
            corruption_idxes = np.random.permutation(self.n_features)[:self.corruption_len]
            corruption_masks[i, corruption_idxes] = True
        return corruption_masks
    

class CorrelationMaskGenerator(RandomMaskGenerator):
    def __init__(self, n_features, high_correlation):
        super().__init__(n_features)
        self.high_correlation = high_correlation
        self.softmax_temporature = 0.3

    def initialize_feature_importances(self, feat_impt):
        assert np.shape(feat_impt) == (self.n_features, self.n_features)
        if not self.high_correlation:
            for i in range(self.n_features):
                # simply flip the indices and reassign the probability values
                # first remove diagonal entries and then add them back in (since they shouldn't be in the order)
                feat_impt_row_tmp = np.delete(feat_impt[i], obj=i)
                feat_impt_row_tmp = np.sort(feat_impt_row_tmp)[::-1][np.argsort(np.argsort(feat_impt_row_tmp))]
                feat_impt[i] = np.insert(feat_impt_row_tmp, obj=i, values=0)
        self.feat_impt = feat_impt
        return 
    
    def get_masks(self, n_samples):
        corruption_masks = np.zeros((n_samples, self.n_features), dtype=bool)
        for i in range(n_samples):
            selected_idxes = []
            remaining_idxes = np.arange(self.n_features)
            selected_id = np.random.choice(self.n_features)
            selected_idxes.append(selected_id)
            remaining_idxes = np.delete(remaining_idxes, obj=selected_id)
            for _ in range(1, self.corruption_len):
                sampling_prob_onerow_tmp = self.feat_impt[selected_idxes][:,remaining_idxes]
                # consider the weakest link from features already selected
                sampling_prob_onerow = np.min(sampling_prob_onerow_tmp, axis=0)
                if np.sum(sampling_prob_onerow) == 0:
                    # every feature remaining has one zero-connection to at least one feature in the selected set
                    selected_id  = np.random.choice(remaining_idxes)
                else:
                    if CORRELATED_FEATURES_RANDOMIZE_SAMPLING:
                        sampling_p = sampling_prob_onerow/np.sum(sampling_prob_onerow)
                        # use softmax with very low temperature to make the distribution more peaky
                        sampling_p = \
                            np.exp(sampling_p/CORRELATED_FEATURES_RANDOMIZE_SAMPLING_TEMPERATURE)/np.sum(np.exp(sampling_p/CORRELATED_FEATURES_RANDOMIZE_SAMPLING_TEMPERATURE))
                        selected_id = np.random.choice(remaining_idxes, p=sampling_p)
                    else:
                        # deterministically select the largest one
                        selected_id = remaining_idxes[np.argmax(sampling_prob_onerow)]
                selected_idxes.append(selected_id)
                remaining_idxes = np.delete(remaining_idxes, np.where(remaining_idxes==selected_id))
            assert len(selected_idxes) == self.corruption_len
            corruption_masks[i, selected_idxes] = True
        return corruption_masks    
