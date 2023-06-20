import librosa
import torch

from torch.nn.utils.rnn import pad_sequence
from typing import List
from starlette.datastructures import UploadFile
from fastapi import HTTPException


def load_batch(
    audio_files: List[UploadFile], max_length: float = 5.0, max_files: int = 16
):
    """Load audio from file being uploaded to the API

    audio_files: list of audio files already 'opened'. These are starlette datastructures

    max_length: float number with maximum duration (in seconds) for audio files. If the number is exceeed, only the first
    max_length portion of the audio file will be loaded, and the rest discarded.

    max_files: int number with maximum number of audio files that can be proceessed at once. Raises a 413 error if
    max_files threshold is exceeded.

    """

    # Exceed number of admissible audio files to be processed
    # in batch
    if len(audio_files) > max_files:
        # Create error message
        ERROR = {
            "code": "413",
            "type": "payload too large",
            "error": "too many files uploaded at once",
            "message": f"resubmit request with no more than {max_files} files",
        }

        raise HTTPException(status_code=413, detail=ERROR)

    # Load audio files, together with their sample rate and filename up to a maximum duration
    batch_audio, batch_names = zip(
        *[load_audio(audio_file, max_length) for audio_file in audio_files]
    )

    # Pad the shortest waveform with zeros (on the right)
    batch_audio = pad_sequence(batch_audio, batch_first=True)

    return batch_audio, batch_names


def load_audio(audio_file: UploadFile, max_length: float = 5.0):
    """Load actual audio waveform from file uploaded to the API. Returns the waveform and
    the file name. The waveform is returned as a PyTorch tensor."""

    # Load waveform and sample rate (discard the latter)
    waveform, _sample_rate = librosa.load(
        audio_file.file, mono=True, sr=None, duration=max_length
    )

    return torch.tensor(waveform), audio_file.filename
