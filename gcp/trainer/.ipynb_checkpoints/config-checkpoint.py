#Author : Youness Landa
import torch

class Config:
    DEVICE  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    STORAGE_BUCKET = 'gs://storage_bucket_speech'
    LOCAL_PATH     = 'tmp'
    OUTPUT_PATH    = 'output'
    
    TRAIN_DATA_DIR = '../../feature-extraction/train'
    VALID_DATA_DIR = '../../feature-extraction/valid'
    TEST_DATA_DIR  = '../../feature-extraction/test'
    
    MODEL_DIR           = 'model'
    MODEL_NAME          = 'speech2emotion.pt'
    
    LOG_DIR             = 'logs'
    LOG_FILE            = "trainer.log"
    TENSORBOARD_LOG_DIR = 'tensorboardlogs'
    
    # Model parameters :
    N_MELS              = 128
    INPUT_SPEC_SIZE     = 3 * N_MELS
    RNN_CELL            = 'lstm' # 'lstm' | 'gru'
    CNN_FILTER_SIZE     = 64
    NUM_GENDER_CLASSES  = 2
    NUM_EMOTION_CLASSES = 8
    
    BATHC_SIZE   = 128
    LR           = 1e-4
    WEIGHT_DECAY = 1e-06
    NUM_EPOCHS   = 1
    
    # For loss function
    ALPHA = 1
    BETA  = 1
    
    # Classes
    EMOTION_NAMES       = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    EMOTION_LABELS      = [1, 2, 3, 4, 5, 6, 7, 8]
    EMOTION_LABEL_2_IXD = {1 : 0, 2 : 1, 3 : 2, 4 : 3, 5 : 4, 6 : 5, 7 : 6, 8 : 7}
    
    GENDER_NAMES   = ['female', 'male']
    GENDER_LABELS  = [0, 1]