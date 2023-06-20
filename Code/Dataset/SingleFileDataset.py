import torchaudio
import torch
import librosa
import torch
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset


class SingleFileDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        window_length: int = None,
    ):
        """
        file_path (str): Path to the file containing the audio
        split: String of either Train, Validation, Test
        """

        # Store directory containing the audio files
        self.file_path = file_path

        # Store window length
        self.window_length = window_length

        # Get the file metadata
        metadata = torchaudio.info(self.file_path)
        
        # Get the sample rate
        self.sr = metadata.sample_rate
        
        # Get the number of frames
        self.num_frames = metadata.num_frames
        
        # Get the total length of the file in seconds
        self.file_length = self.num_frames / self.sr
        
        if self.window_length:
        
            # Compute the number of split
            n_splits = self.num_frames // (self.window_length * self.sr)

            # Adjust it
            if self.num_frames % (self.window_length * self.sr) != 0:
                
                # Store it
                self.n_splits = n_splits + 1
                
        else:
            
            self.n_splits = 1

    def __len__(self):
        """Return the total number of samples present in the dataset"""

        return self.n_splits

    def __getitem__(self, index):
        """Return item corresponding to a certain index"""

        if self.window_length:
            
            # Start of the window
            start = index * self.window_length
            
            # Load waveform and sample rate
            waveform, sample_rate = librosa.load(self.file_path, mono=True, sr=None, offset=start, duration=self.window_length)
                        
            # Pad if necessary
            if len(waveform) < self.window_length * self.sr:
                
                # Pad it
                waveform = np.concatenate(
                    (
                        waveform,
                        np.zeros(
                            self.window_length * self.sr - len(waveform), dtype="float32"
                        ),
                    )
                )
                
            # Generate the splits in seconds
            splits = np.array([start, start + self.window_length])
            
            return torch.Tensor(waveform), Path(self.file_path).stem, splits
            
        else:
            # Load waveform and sample rate
            waveform, sample_rate = librosa.load(self.file_path, mono=True, sr=None)
            
            return torch.Tensor(waveform), Path(self.file_path).stem
