import librosa
import torch

from typing import Tuple
from starlette.datastructures import UploadFile


def load_audio(audio_file: UploadFile) -> Tuple[torch.tensor, float, str]:
    """Load actual audio waveform from file uploaded to the API. Returns the waveform, sample rate
    and the file name. The waveform is returned as a PyTorch tensor."""

    # Load waveform and sample rate (discard the latter)
    waveform, sample_rate = librosa.load(audio_file.file, mono=True, sr=None)

    # Reshape the waveform to introduce the batch dimension
    waveform = waveform.reshape((1, -1))

    return torch.tensor(waveform), sample_rate, audio_file.filename
