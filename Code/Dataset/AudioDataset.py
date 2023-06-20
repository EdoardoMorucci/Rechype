import os
import torch
import torchaudio
import pandas as pd
import torchaudio.transforms as T

from pathlib import Path
from torchvision.transforms import RandomCrop
from torch.utils.data import Dataset
from torchaudio_augmentations import RandomApply, Noise, PitchShift, Compose


class AudioDataset(Dataset):
    def __init__(
        self,
        dataset_directory: str,
        split: str = None,
        inference: bool = False,
        augment: bool = True,
    ):
        """
        dataset_directory: Path to directory containing audio and labels
        split: String of either Train, Validation, Test
        transform (callable, optional): Optional transformation to be applied on a sample
        """

        # Store directory containing the audio files
        self.audio_directory = os.path.join(dataset_directory, "Audio")

        # Store the inference mode
        self.inference = inference

        # Store split
        self.split = split

        # Store data augmentation
        self.augment = augment

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

        # If not in inference mode, load labels
        if not self.inference:
            # Read the CSV file containing the labels for each recording
            self.labels = pd.read_csv(
                os.path.join(dataset_directory, "labels.csv")
            ).query("`split` == @split")

            # Store number of labels
            self.n_labels = self.labels["class"].nunique()
        else:
            # If in inference mode, we cannot use labels to later grab the files
            # Hence, we parse the entire directory looking for admissible files
            self.audio_files = [
                path.resolve()
                for path in Path(self.audio_directory).glob("**/*")
                if path.suffix in self.formats
            ]

        # Add transformations
        if self.augment:
            # List of transformations to apply
            transformations = [
                RandomApply([Noise(min_snr=0.001, max_snr=0.005)], p=0.5),
                # RandomApply([PitchShift(
                #     n_samples=2,
                #     sample_rate=sample_rate
                # )], probability),
            ]

            # Compose them into a function
            self.transform = Compose(transforms=transformations)

    def __len__(self):
        """Return the total number of samples present in the dataset"""

        return len(self.audio_files) if self.inference else len(self.labels)

    def __getitem__(self, index):
        """Return item corresponding to a certain index"""

        if self.inference:
            # In inference mode, only return the waveform without class
            audio_path = self.audio_files[index]

            # Load the audio file
            waveform, _sample_rate = torchaudio.load(audio_path)

            return waveform[0, ...], Path(audio_path).stem

        else:
            # Create the audio name
            audio_path = os.path.join(self.audio_directory, self.labels.iloc[index, 0])

            # Fetch the audio class
            audio_class = self.labels.iloc[index, 1]

            # Load the audio file
            waveform, _sample_rate = torchaudio.load(audio_path)

            if self.split == "Test":
                return waveform[0, ...], audio_class, Path(audio_path).stem

            # Apply data augmentation
            elif self.augment:
                if self.split == "Train":
                    waveform = self.transform(waveform)

                # waveform = RandomCrop(size=(1, 3 * _sample_rate))(waveform)

            return waveform[0, ...], audio_class
