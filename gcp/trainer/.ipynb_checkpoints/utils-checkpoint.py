#Author : Youness Landa
import os, subprocess, logging
import subprocess

from config import Config as cfg

def make_directories():
    logger.info('Making necessary directories')
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
    

def download_data_from_gcs():
    '''
        Downloads data from Google Cloud Storage (GCS) bucket.
    '''
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
    
def export_to_gcs(fitted_pipeline: Pipeline, model_dir: str):
    '''
        Exports all training files to GCS Bucket
    '''
    scheme, bucket, path, file = process_gcs_uri(model_dir)
    if scheme != "gs:":
        raise ValueError("URI scheme must be gs")
    
    # Upload the model to GCS
    b = storage.Client().bucket(bucket)
    export_path = os.path.join(path, 'model.pkl')
    blob = b.blob(export_path)
    
    blob.upload_from_string(pickle.dumps(fitted_pipeline))
    return scheme + "//" + os.path.join(bucket, export_path)