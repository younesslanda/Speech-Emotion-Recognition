import math
import os
import pathlib
import random

import torch
import torchaudio
import torchaudio.functional as aF

from config import Config as cfg

def _get_sample(path, resample=None):
    effects = [
    ["remix", "1"]
    ]
    if resample:
        effects.extend([
          ["lowpass", f"{resample // 2}"],
          ["rate", f'{resample}'],
        ])
    return torchaudio.sox_effects.apply_effects_file(path, effects=effects)

def get_rir_sample(*, resample=None, processed=False):
    rir_raw, sample_rate = _get_sample(cfg._RIR_PATH, resample=resample)
    if not processed:
        return rir_raw, sample_rate
    rir = rir_raw[:, int(sample_rate*1.01):int(sample_rate*1.3)]
    rir = rir / torch.norm(rir, p=2)
    rir = torch.flip(rir, [1])
    return rir, sample_rate

class RandomBackgroundNoise:
    def __init__(self, noise_dir, min_snr_db=cfg.MIN_SNR_DB, max_snr_db=cfg.MAX_SNR_DB):

        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

        if not os.path.exists(noise_dir):
            raise IOError(f'Noise directory `{noise_dir}` does not exist')
        # find all WAV files including in sub-folders:
        self.noise_files_list = list(pathlib.Path(noise_dir).glob('**/*.wav'))
        if len(self.noise_files_list) == 0:
            raise IOError(f'No .wav file found in the noise directory `{noise_dir}`')

    def __call__(self, waveform_):
        '''
            waveform_ : tensor of the waveform
        '''
        random_noise_file = random.choice(self.noise_files_list)

        # Apply RIR
        rir, _ = get_rir_sample(resample=None, processed=True)
        effects = [
            ['remix', '1'], # convert to mono
            ['rate', str(cfg.SAMPLE_RATE)], # resample
        ]
        rir, _ = torchaudio.sox_effects.apply_effects_tensor(rir, cfg.SAMPLE_RATE, effects)
        waveform_, _ = torchaudio.sox_effects.apply_effects_tensor(waveform_, cfg.SAMPLE_RATE, effects)

        waveform_ = torch.nn.functional.pad(waveform_, (rir.shape[1]-1, 0))
        waveform_ = torch.nn.functional.conv1d(waveform_[None, ...], rir[None, ...])[0]
        
        # Add background noise
        # Because the noise is recorded in the actual environment, we consider that
        # the noise contains the acoustic feature of the environment. Therefore, we add
        # the noise after RIR application.
        effects = [
            ['remix', '1'], # convert to mono
            ['rate', str(cfg.SAMPLE_RATE)], # resample
            ['vol', '0.5'] # change the volume

        ]
        noise, _ = torchaudio.sox_effects.apply_effects_file(random_noise_file, effects, normalize=True)

        audio_length = waveform_.shape[-1]
        noise_length = noise.shape[-1]
        if noise_length > audio_length:
            offset = random.randint(0, noise_length-audio_length)
            noise = noise[..., offset:offset+audio_length]
        elif noise_length < audio_length:
            noise = torch.cat([noise, torch.zeros((noise.shape[0], audio_length-noise_length))], dim=-1)

        snr_db = random.randint(self.min_snr_db, self.max_snr_db)
        snr = math.exp(snr_db / 10)
        audio_power = waveform_.norm(p=2)
        noise_power = noise.norm(p=2)
        scale = snr * noise_power / audio_power
        waveform_ = (scale * waveform_ + noise ) / 2
        
        # Apply filtering 
        waveform_, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
                          waveform_,
                          cfg.SAMPLE_RATE,
                          effects=[
                              ["lowpass", "4000"],
                              ["compand", "0.02,0.05", "-60,-60,-30,-10,-20,-8,-5,-8,-2,-8", "-8", "-7", "0.05"],
                              ["rate", str(cfg.SAMPLE_RATE)],
                          ],
                        )
        return waveform_