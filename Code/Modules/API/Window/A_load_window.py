import librosa
import torch
import numpy as np

from torch.nn.utils.rnn import pad_sequence
from typing import List
from starlette.datastructures import UploadFile
from fastapi import HTTPException


def load_window(audio_file: UploadFile, window_length: float = 3.0):
    """Load actual audio waveform from file uploaded to the API. The waveform is subdivided into
    samples of length window_length and then stacked to create a batch.
    The batch is returned in the form of a Torch tensor"""

    # Load waveform and sample rate
    waveform, sample_rate = librosa.load(audio_file.file, mono=True, sr=None)

    # Convert the window length from seconds to
    # total number of samples, given the sample rate of
    # the audio clip
    window_length_sampled = round(window_length * sample_rate)

    # If the length of the window is in fact shorter that the waveform,
    # We split the waveform in windows, that we then stack to create a batch
    if window_length_sampled <= len(waveform):
        # Compute the number of splits we shall divide the waveform into
        n_splits = len(waveform) // window_length_sampled

        if len(waveform) % window_length_sampled == 0:
            n_splits -= 1

        if len(waveform) % window_length_sampled == 0:
            n_splits -= 1

        # Create vector of waveforms, to be stacked later on
        stacked_waveforms = [
            waveform[i * window_length_sampled : (i + 1) * window_length_sampled]
            for i in range(n_splits + 1)
        ]

        # The last entry is most likely of the wrong size
        # Hence, we pad it with zeros on the right
        stacked_waveforms[-1] = np.concatenate(
            (
                stacked_waveforms[-1],
                np.zeros(
                    window_length_sampled - len(stacked_waveforms[-1]), dtype="float32"
                ),
            )
        )

        # Actuall stack the waveforms
        batch = torch.tensor(np.vstack(stacked_waveforms))

        # Generate the splits in seconds
        splits = np.append(
            np.arange(0, len(waveform) / sample_rate, window_length),
            len(waveform) / sample_rate,
        )

    # If the window is larger, there is no need to do all of that, we just create
    # a dummy batch
    else:
        # Create dummy batch
        batch = torch.unsqueeze(torch.tensor(waveform), dim=0)

        # Create dummy split
        splits = np.array([0.0, len(waveform) / sample_rate])

    return batch, audio_file.filename, splits
