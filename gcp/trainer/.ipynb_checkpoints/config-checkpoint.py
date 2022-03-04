class Config:
    #Model parameters :
    N_MELS = 128
    INPUT_SPEC_SIZE = 3 * N_MELS
    RNN_CELL = 'lstm' # 'lstm' | 'gru'
    CNN_FILTER_SIZE = 64
    
    NUM_GENDER_CLASSES = 2
    NUM_EMOTION_CLASSES = 8
    