import torch
from torchaudio.transforms import MelSpectrogram


class Spectrogram(object):

    """Creates the Mel Spectrogram of a waveform"""

    def __init__(
        self,
        n_fft: int = 400,
        win_length: int = 400,
        hop_length: int = 200,
        f_min: float = 0,
        f_max: float = None,
    ):
        """Initialise Mel Spectrogram

        sample_rate: sample rate of waveform
        n_fft: size of FFT, creates n_fft //2 +1 bins
        win_length: window size
        hop_length: length of hop between stft windows
        f_min:minimum frequency
        f_max: maximum frequency

        """

        self.spectrogram = MelSpectrogram(
            n_fft=n_fft,
            win_length=win_length,
            f_min=f_min,
            f_max=f_max,
            hop_length=hop_length,
            n_mels=32,
        )

    def __call__(self, waveform: torch.tensor, sample_rate: int):
        # Fix the Mel Spectrogram sample rate
        self.spectrogram.sample_rate = sample_rate

        # Create spectrogram
        spectrum = self.spectrogram(waveform)[0, ...]

        return spectrum, sample_rate
