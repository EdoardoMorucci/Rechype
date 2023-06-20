import torch
import torchaudio.transforms as T


class Resample(object):

    """Resample a waveform to a new sample rate"""

    def __init__(self, resample_rate: int):
        # Check the resample rate is compliant
        assert isinstance(resample_rate, int)

        # Store the resample rate
        self.resample_rate = resample_rate

    def __call__(self, waveform: torch.tensor, sample_rate: int):
        """Apply the resampling of a waveform from a sample_rate to a resample_rate"""

        # Resample the waveform
        resampled_wavform = T.Resample(sample_rate, self.resample_rate)(waveform)

        # Return the resampled waveform and new sample rate
        return resampled_wavform, self.resample_rate
