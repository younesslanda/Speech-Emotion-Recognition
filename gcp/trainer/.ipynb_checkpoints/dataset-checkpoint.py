#Author : Youness Landa
import torch
import torch.utils.data as data
import torch.nn.functional as

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
        emotion_idx = data['emotion']
        gender_idx = data['gender']
        length = feature.shape[-1]

        return feature, length, emotion_idx, gender_idx
    
def collate_fn(batch):
    """
      Creates mini-batch tensors from the list 
      of tuples (feature, length, emotion_idx, gender_idx)
    """

    #sort in descending order
    batch.sort(key=lambda x: x[1], reverse=True)
    
    #from list of tuples to tuple of lists
    features, lengths, emotion_idxs, gender_idxs = zip(*batch)

    batch_size, num_channels, spec_dimension, temporal_length = len(features), 1, cfg.INPUT_SPEC_SIZE, max(lengths)
    padded_features = torch.zeros(batch_size, num_channels, spec_dimension, temporal_length)

    for i, feature in enumerate(features):
        padded_features[i] = F.pad(feature, (0, max(lengths) - lengths[i]))

    return padded_features, torch.tensor(lengths), torch.tensor(emotion_idxs), torch.tensor(gender_idxs)
