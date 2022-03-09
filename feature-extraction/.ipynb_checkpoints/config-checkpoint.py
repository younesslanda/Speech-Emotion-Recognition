import os

class Config:
    _RIR = "_rir"
    _RIR_PATH = os.path.join(_RIR, "rir.wav")
    _NOISE = "_noise"

    SAMPLE_RATE = 22050
    N_FFT = 1024
    WIN_LENGTH = 800
    HOP_LENGTH = 400
    N_MELS = 128
    
    MIN_SNR_DB = 7
    MAX_SNR_DB = 25
    
    TIME_MASK_PARAM = 60
    FREQ_MASK_PARAM = 60
    
    TRAIN_DIR = 'train'
    TEST_DIR  = 'test'
    VALID_DIR = 'valid'
    