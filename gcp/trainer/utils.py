#Author : Youness Landa
import os, subprocess, logging

from config import Config as cfg

def make_directories():
    logging.info('Making necessary directories')
    if not os.path.exists(cfg.LOCAL_PATH):
        os.mkdir(cfg.LOCAL_PATH)
        
    if not os.path.exists(cfg.OUTPUT_PATH):
        os.mkdir(cfg.OUTPUT_PATH)
        
        
    if not os.path.exists(os.path.join(cfg.LOCAL_PATH, cfg.TRAIN_DATA_DIR)):
        os.mkdir(os.path.join(cfg.LOCAL_PATH, cfg.TRAIN_DATA_DIR))
        
    if not os.path.exists(os.path.join(cfg.LOCAL_PATH, cfg.VALID_DATA_DIR)):
        os.mkdir(os.path.join(cfg.LOCAL_PATH, cfg.VALID_DATA_DIR))
        
    if not os.path.exists(os.path.join(cfg.LOCAL_PATH, cfg.TEST_DATA_DIR)):
        os.mkdir(os.path.join(cfg.LOCAL_PATH, cfg.TEST_DATA_DIR))


    if not os.path.exists(os.path.join(cfg.OUTPUT_PATH, cfg.MODEL_DIR)):
        os.mkdir(os.path.join(cfg.OUTPUT_PATH, cfg.MODEL_DIR))
        
    if not os.path.exists(os.path.join(cfg.OUTPUT_PATH, cfg.LOG_DIR)):
        os.mkdir(os.path.join(cfg.OUTPUT_PATH, cfg.LOG_DIR))
        
    if not os.path.exists(os.path.join(cfg.OUTPUT_PATH, cfg.LOG_DIR, cfg.TENSORBOARD_LOG_DIR)):
        os.mkdir(os.path.join(cfg.OUTPUT_PATH, cfg.LOG_DIR, cfg.TENSORBOARD_LOG_DIR))
    logging.info('End of making necessary directories \n')
    
def download_data_from_gcs():
    '''
        Downloads data from Google Cloud Storage (GCS) bucket.
    '''
    logging.info('- Downloading data from Google Cloud Storage -')
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
    logging.info('- Data has been successfully downloaded from GCS -\n')
    
def export_to_gcs():
    '''
        Exports all training files to GCS Bucket
    '''
    logging.info(' - Exporting all training files to GCS Bucket -\n')
    subprocess.call([
        'gsutil', '-q', '-m', 'cp', '-r',
        # Output path
        os.path.join(cfg.OUTPUT_PATH),
        # GCS path
        os.path.join(cfg.STORAGE_BUCKET)
    ])