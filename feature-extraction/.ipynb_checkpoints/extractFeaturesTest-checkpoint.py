import torch
import torchaudio
import torchaudio.functional as aF
import torchaudio.transforms as aT

from config import Config as cfg

class ExtractFeaturesTest:
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

    def __call__(self, wave_path):
        waveform, sample_rate = torchaudio.load(wave_path)
        
        resampler = aT.Resample(sample_rate, cfg.SAMPLE_RATE, dtype=waveform.dtype)
        waveform_ = resampler(waveform)

        melspec = self.mel_spectrogram(waveform_)

        deltaspec = self.delta(melspec)

        delta2spec = self.delta(deltaspec)

        feature = torch.cat((melspec, deltaspec, delta2spec), dim=1)

        return feature