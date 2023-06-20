import os
import torchaudio
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset


class SiameseDataset(Dataset):
    def __init__(
        self,
        dataset_directory: str,
        split: str = None,
    ):
        """
        dataset_directory: Path to directory containing audio and labels
        split: String of either Train, Validation, Test
        transform (callable, optional): Optional transformation to be applied on a sample
        """

        # Store directory containing the audio files
        self.audio_directory = os.path.join(dataset_directory, "Audio")

        # Store split
        self.split = split

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

        # Read the CSV file containing the labels for each recording
        classification_df = pd.read_csv(
            os.path.join(dataset_directory, "labels.csv")
        ).query("`split` == @split")

        # Create labels suitable for siamese
        self.labels = self.create_siamese_labels(classification_df)

    def create_siamese_labels(df: pd.DataFrame) -> pd.DataFrame:
        """Create labels that are suitable for a Siamese network"""

        # If n is the number of rows in n, create a n^2 df by having each row of df
        # coupled with each row from the (copy of) df
        merged_df = pd.merge(df, df, on="split", suffixes=("_1", "_2"))

        # Create a new class label
        merged_df["class"] = (merged_df["class_1"] == merged_df["class_2"]).astype(int)

        # Create the class_name label
        merged_df["class_name"] = merged_df["class"].apply(
            lambda x: x * "Same" + (1 - x) * "Different"
        )

        # Reorder
        merged_df = merged_df[
            [
                "file_1",
                "file_2",
                "class",
                "class_name",
                "split",
                "class_1",
                "class_name_1",
                "class_2",
                "class_name_2",
            ]
        ]

    def __len__(self):
        """Return the total number of samples present in the dataset"""

        return len(self.labels)

    def __getitem__(self, index):
        """Return item corresponding to a certain index"""

        # Create the audio names
        audio_path_1, audio_path_2 = os.path.join(
            self.audio_directory, self.labels.iloc[index, [0, 1]]
        )

        # Fetch the audio class
        audio_class = self.labels.iloc[index, 2]

        # Load the audio files
        waveform_1, _sample_rate_1 = torchaudio.load(audio_path_1)
        waveform_2, _sample_rate_2 = torchaudio.load(audio_path_2)

        if self.split == "Test":
            return (
                waveform_1[0, ...],
                waveform_2[0, ...],
                audio_class,
                Path(audio_path_1).stem,
                Path(audio_path_2).stem,
            )

        else:
            return waveform_1[0, ...], waveform_2[0, ...], audio_class
