import os
import librosa
import torch

from pathlib import Path
from torch.utils.data import Dataset
from ..SoundProcessing.find_events import find_events


class AudioDataset(Dataset):
    def __init__(
        self,
        dataset_directory: str,
        **kwargs,
    ):
        """
        dataset_directory: Path to directory containing audio (and labels)
        """

        # Store directory containing the audio files
        self.audio_directory = os.path.join(dataset_directory, "Audio")

        # Get the audio files
        self.audio_files = librosa.util.find_files(self.audio_directory)

        # Store keyword arguments
        self.kwargs = kwargs

    def __len__(self):
        """Return the total number of samples present in the dataset"""

        return len(self.audio_files)

    def __getitem__(self, index):
        """Return item corresponding to a certain index"""

        # In inference mode, only return the waveform without class
        audio_path = self.audio_files[index]

        # From audio path, get events, sample rate and peak times
        events, sr, peak_times = find_events(audio_path, **self.kwargs)

        return torch.tensor(events), sr, peak_times, audio_path
