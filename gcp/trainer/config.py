import torch

class Config:
    DEVICE  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    STORAGE_BUCKET  = 'gs://storage_bucket_speech'
    TRAIN_DATA_DIR  = 'train'
    VALID_DATA_DIR  = 'valid'
    TEST_DATA_DIR   = 'test'
    
    LOCAL_PATH = '/tmp/'
    
    #Model parameters :
    N_MELS              = 128
    INPUT_SPEC_SIZE     = 3 * N_MELS
    RNN_CELL            = 'lstm' # 'lstm' | 'gru'
    CNN_FILTER_SIZE     = 64
    NUM_GENDER_CLASSES  = 2
    NUM_EMOTION_CLASSES = 8
    
    BATHC_SIZE   = 64
    LR           = 1e-3
    WEIGHT_DECAY = 1e-05
    NUM_EPOCHS   = 100
    
    #For loss function
    ALPHA = 1
    BETA  = 1