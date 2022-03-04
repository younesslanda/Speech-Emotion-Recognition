import random
import torchaudio

from config import Config as cfg

class RandomSpeedChange:
    def __init__(self):
        self.speed_factors = [0.9, 1.0, 1.1]

    def __call__(self, waveform):
        speed_factor = random.choice(self.speed_factors)
        if speed_factor == 1.0: # no change
            return waveform

        # change speed and resample to original rate:
        sox_effects = [
            ["speed", str(speed_factor)],
            ["rate", str(cfg.SAMPLE_RATE)],
        ]
        transformed_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, cfg.SAMPLE_RATE, sox_effects)
        return transformed_audio