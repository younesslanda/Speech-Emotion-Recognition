#Author : Youness Landa
import os
import subprocess

from config import Config as cfg

def download_data_from_gcs():
    '''
        Downloads data from Google Cloud Storage (GCS) bucket.
    '''
    if not os.path.exists(cfg.LOCAL_PATH):
        os.mkdir(cfg.LOCAL_PATH)
        
    if not os.path.exists(cfg.TRAIN_DATA_DIR):
        os.mkdir(os.path.join(cfg.LOCAL_PATH, cfg.TRAIN_DATA_DIR))
        
    if not os.path.exists(cfg.VALID_DATA_DIR):
        os.mkdir(cfg.LOCAL_PATH, cfg.VALID_DATA_DIR)
        
    if not os.path.exists(cfg.TEST_DATA_DIR):
        os.mkdir(cfg.LOCAL_PATH, cfg.TEST_DATA_DIR)
        
    subprocess.call([
        'gsutil', '-q', '-m', 'cp',
        # Storage path
        os.path.join(cfg.STORAGE_BUCKET, cfg.TRAIN_DATA_DIR, '*.pickle'),
        # Local path
        os.path.join(cfg.LOCAL_PATH, cfg.TRAIN_DATA_DIR)
    ])
    
    subprocess.call([
        'gsutil', '-q', '-m', 'cp',
        # Storage path
        os.path.join(cfg.STORAGE_BUCKET, cfg.VALID_DATA_DIR, '*.pickle'),
        # Local path
        os.path.join(cfg.LOCAL_PATH, cfg.VALID_DATA_DIR)
    ])
    
    subprocess.call([
        'gsutil', '-q', '-m', 'cp',
        # Storage path
        os.path.join(cfg.STORAGE_BUCKET, cfg.TEST_DATA_DIR, '*.pickle'),
        # Local path
        os.path.join(cfg.LOCAL_PATH, cfg.TEST_DATA_DIR)
    ])
    
    print('- Data has been successfully downloaded from GCS -\n')