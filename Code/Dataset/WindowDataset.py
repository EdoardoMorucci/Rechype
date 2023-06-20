import os
import torch
import librosa
import torch
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(
        self,
        dataset_directory: str,
        window_length: int,
    ):
        """
        dataset_directory: Path to directory containing audio and labels
        split: String of either Train, Validation, Test
        transform (callable, optional): Optional transformation to be applied on a sample
        """

        # Store directory containing the audio files
        self.audio_directory = os.path.join(dataset_directory, "Audio")

        # Store window length
        self.window_length = window_length

        # Store formats of audio that can be handled by torch
        self.formats = [
            ".wav",
            ".mp3",
            ".ogg",
            ".vorbis",
            ".amr-nb",
            ".amb",
            ".flac",
            ".sph",
            ".gsm",
            ".htk",
        ]

        # If in inference mode, we cannot use labels to later grab the files
        # Hence, we parse the entire directory looking for admissible files
        self.audio_files = [
            path.resolve()
            for path in Path(self.audio_directory).glob("**/*")
            if path.suffix in self.formats
        ]

    def __len__(self):
        """Return the total number of samples present in the dataset"""

        return len(self.audio_files)

    def __getitem__(self, index):
        """Return item corresponding to a certain index"""

        # Get the path to the audio file
        audio_path = self.audio_files[index]

        # Load waveform and sample rate
        waveform, sample_rate = librosa.load(audio_path, mono=True, sr=None)

        # Conver to torch tensor
        waveform = torch.Tensor(waveform)

        # Convert the window length from seconds to
        # total number of samples, given the sample rate of
        # the audio clip
        window_length_sampled = round(self.window_length * sample_rate)

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
                        window_length_sampled - len(stacked_waveforms[-1]),
                        dtype="float32",
                    ),
                )
            )

            # Actuall stack the waveforms
            batch = torch.tensor(np.vstack(stacked_waveforms))

            # Generate the splits in seconds
            splits = np.append(
                np.arange(0, len(waveform) / sample_rate, self.window_length),
                len(waveform) / sample_rate,
            )

        # If the window is larger, there is no need to do all of that, we just create
        # a dummy batch
        else:
            # Create dummy batch
            batch = torch.unsqueeze(waveform, dim=0)

            # Create dummy split
            splits = np.array([0.0, len(waveform) / sample_rate])

        return batch, Path(audio_path).stem, splits
