#Author : Youness Landa
import os, pickle

import torch
import torch.utils.data as data
import torch.nn.functional as F

from config import Config as cfg

class Dataset(data.Dataset):
    def __init__(self, pckl_dir):
        self.pckl_dir = pckl_dir
        self.pckl_files = os.listdir(self.pckl_dir)

    def __len__(self):
        return len(self.pckl_files)
    
    def __getitem__(self, index):
        filename = self.pckl_files[index]
        with open(os.path.join(self.pckl_dir, filename), 'rb') as handle:
            data = pickle.load(handle)

        feature = data['feature']
        emotion_idx = cfg.EMOTION_LABEL_2_IXD[data['emotion']]
        gender_idx = data['gender']
        length = feature.shape[-1]
        
        # TODO : change !!
        if feature.shape[0] > 1:
            feature = feature[0].unsqueeze(0)
        
        return feature, length, emotion_idx, gender_idx
    
def collate_fn(batch):
    """
      Creates mini-batch tensors from the list 
      of tuples (feature, length, emotion_idx, gender_idx)
    """
    #sort in descending order according to length
    batch.sort(key=lambda x: x[1], reverse=True)
    
    #from list of tuples to tuple of lists
    features, lengths, emotion_idxs, gender_idxs = zip(*batch)

    batch_size, num_channels, spec_dimension, temporal_length = len(features), 1, cfg.INPUT_SPEC_SIZE, max(lengths)
    padded_features = torch.zeros(batch_size, num_channels, spec_dimension, temporal_length)

    for i, feature in enumerate(features):
        padded_features[i] = F.pad(feature, (0, max(lengths) - lengths[i]))

    return (padded_features.to(cfg.DEVICE), torch.tensor(lengths).to(cfg.DEVICE),
           torch.tensor(emotion_idxs).to(cfg.DEVICE), torch.tensor(gender_idxs).to(cfg.DEVICE))