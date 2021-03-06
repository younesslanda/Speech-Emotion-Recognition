#Author : Youness Landa
import os, logging

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from distutils.version import LooseVersion
from torch.utils.tensorboard import SummaryWriter

from model import Speech2Emotion
from experiment import Experiment
from dataset import Dataset, collate_fn
from utils import download_data_from_gcs, make_directories, export_to_gcs

from config import Config as cfg

def main():
    log_file_name = os.path.join(cfg.OUTPUT_PATH, cfg.LOG_DIR, cfg.LOG_FILE)
    logging.basicConfig(
        format   = '%(asctime)s : %(message)s',
        filename = log_file_name,
        level    = logging.INFO,
    )
    
    logging.info('Training job starting ...\n')
    
    # Making the necessary directories
    make_directories()
    
    # Downloading data from GCS Bucket
    #download_data_from_gcs()
    
    # DataLoaders
    train_pck_dir = os.path.join(cfg.TRAIN_DATA_DIR) 
    #train_pck_dir = os.path.join(cfg.LOCAL_PATH, cfg.TRAIN_DATA_DIR) 
    dataset_train = Dataset(train_pck_dir)
    
    valid_pck_dir = os.path.join(cfg.VALID_DATA_DIR) 
    #valid_pck_dir = os.path.join(cfg.LOCAL_PATH, cfg.VALID_DATA_DIR) 
    dataset_valid = Dataset(valid_pck_dir)
    
    test_pck_dir  = os.path.join(cfg.TEST_DATA_DIR) 
    #test_pck_dir  = os.path.join(cfg.LOCAL_PATH,  cfg.TEST_DATA_DIR) 
    dataset_test  = Dataset(test_pck_dir)
    
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=cfg.BATHC_SIZE, shuffle=True, collate_fn=collate_fn)
    dataloader_valid = DataLoader(dataset=dataset_valid, batch_size=cfg.BATHC_SIZE, shuffle=True, collate_fn=collate_fn)
    dataloader_test  = DataLoader(dataset=dataset_test,  batch_size=1, collate_fn=collate_fn)
    
    # Defining the model
    logging.info('DEVICE used : {}'.format(cfg.DEVICE))
    model = Speech2Emotion().to(cfg.DEVICE)
    
    optimizer = Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    # Defining the tensorboard writer
    tensorboard_log_dir = os.path.join(cfg.OUTPUT_PATH, cfg.LOG_DIR, cfg.TENSORBOARD_LOG_DIR)
    writer = SummaryWriter(tensorboard_log_dir)
    
    # Running the experiment
    exp = Experiment(dataloader_train, dataloader_valid, dataloader_test, optimizer, criterion, writer)
    model = exp.run(model)
    exp.test(model)
    
    # Saving the model
    model_path = os.path.join(cfg.OUTPUT_PATH, cfg.MODEL_DIR, cfg.MODEL_NAME)
    torch.save(model.state_dict(), model_path)
    print('Model is saved to : {}'.format(model_path))
    logging.info('Model is saved to : {}'.format(model_path))
    
    # Closing the tensorboard writer
    print('Tensorboard logs are saved to : {}'.format(tensorboard_log_dir))
    logging.info('Tensorboard logs are saved to : {}, closing the tensorboard writer.'.format(tensorboard_log_dir))
    writer.close()
    
    # Exporting training files to GCS
    export_to_gcs()
    
    logging.info('Training job completed. Exiting...')
    
if __name__ == '__main__':
    main()