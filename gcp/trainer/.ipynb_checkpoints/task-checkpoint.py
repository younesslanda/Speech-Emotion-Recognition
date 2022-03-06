#Author : Youness Landa
import os

import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from utils import download_data_from_gcs
from model import Speech2Emotion
from dataset import Dataset, collate_fn

from config import Config as cfg

def main():
    #Downloading data from GCS
    download_data_from_gcs()
    
    #DataLoaders
    train_pck_dir = os.path.join(cfg.LOCAL_PATH, cfg.TRAIN_DATA_DIR) 
    dataset_train = Dataset(train_pck_dir)
    
    valid_pck_dir = os.path.join(cfg.LOCAL_PATH, cfg.VALID_DATA_DIR) 
    dataset_valid = Dataset(valid_pck_dir)
    
    test_pck_dir  = os.path.join(cfg.LOCAL_PATH,  cfg.TEST_DATA_DIR) 
    dataset_test  = Dataset(test_pck_dir)
    
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=cfg.BATHC_SIZE, shuffle=True, collate_fn=collate_fn)
    dataloader_valid = DataLoader(dataset=dataset_valid, batch_size=cfg.BATHC_SIZE, shuffle=True, collate_fn=collate_fn)
    dataloader_test  = DataLoader(dataset=dataset_test,  batch_size=cfg.BATHC_SIZE, shuffle=True, collate_fn=collate_fn)
    
    #Defining the model
    model = Speech2Emotion()
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    loss = nn.CrossEntropyLoss()
    
    
if __name__ == '__main__':
    main()