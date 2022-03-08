import torch
import torchaudio
import torchaudio.transforms as aT

from randomBackgroundNoise import RandomBackgroundNoise
from randomSpeedChange import RandomSpeedChange
from timefreaMasking import TimefreaMasking

from config import Config as cfg

class ExtractFeaturesTrain:
    '''
        returns a feature tensor of size (1 x 3 * N_MELS x Temporal_lenght)
    '''
    def __init__(self):
        self.mel_spectrogram = aT.MelSpectrogram(
                                    sample_rate=cfg.SAMPLE_RATE,
                                    n_fft=cfg.N_FFT,
                                    win_length=cfg.WIN_LENGTH,
                                    hop_length=cfg.HOP_LENGTH,
                                    center=True,
                                    pad_mode="reflect",
                                    power=2.0,
                                    norm='slaney',
                                    onesided=True,
                                    n_mels=cfg.N_MELS,
                                    mel_scale="htk",
                                  )
        self.delta = aT.ComputeDeltas(win_length=5, mode='replicate')

        self.noise_transform = RandomBackgroundNoise(cfg._NOISE)
        self.speed_transform = RandomSpeedChange()
        self.timefreq_transform = TimefreaMasking()

    def __call__(self, wave_path, addBackgroundNoise=False, changeSpeed=False, timefreqMask=False):
        waveform, sample_rate = torchaudio.load(wave_path)

        resampler = aT.Resample(sample_rate, cfg.SAMPLE_RATE, dtype=waveform.dtype)
        waveform_ = resampler(waveform)
        
        effects = [
            ['channels', '1'], # convert to 1 channel
        ]
        waveform_, _ = torchaudio.sox_effects.apply_effects_tensor(waveform_, cfg.SAMPLE_RATE, effects)

        if changeSpeed:
            waveform_ = self.speed_transform(waveform_)

        if addBackgroundNoise:
            waveform_ = self.noise_transform(waveform_)

        
        melspec = self.mel_spectrogram(waveform_)

        if timefreqMask:
            melspec = self.timefreq_transform(melspec)

        deltaspec = self.delta(melspec)

        delta2spec = self.delta(deltaspec)

        feature = torch.cat((melspec, deltaspec, delta2spec), dim=1)

        return feature
    
if __name__ == '__main__':
    extractFeaturesTrain = ExtractFeaturesTrain()
    print(extractFeaturesTrain(cfg._NOISE + '/' + 'noise1.wav', addBackgroundNoise=True, timefreqMask=True).shape)
    