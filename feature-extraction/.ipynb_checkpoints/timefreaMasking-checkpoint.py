import random
import torchaudio.transforms as aT

from config import Config as cfg

class TimefreaMasking:
    def __init__(self, time_mask_param=cfg.TIME_MASK_PARAM, freq_mask_param=cfg.FREQ_MASK_PARAM):
        self.timeMask = aT.TimeMasking(time_mask_param)
        self.freqMask = aT.FrequencyMasking(freq_mask_param)

        self.mask = [self.timeMask, self.freqMask]
        
    def __call__(self, spec):
        mask = random.choice(self.mask)
        spec = mask(spec)
        return spec