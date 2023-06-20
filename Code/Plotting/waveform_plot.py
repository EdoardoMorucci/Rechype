import librosa
import numpy as np
import matplotlib.pyplot as plt


def plot_waveform(waveform: np.array, sr: int):
    """Display the waveform of a signal."""

    librosa.display.waveshow(waveform, sr=sr)


def compare_waveforms(waveforms: list, sr: int):
    """Display the waveform of a signal."""

    for i, waveform in enumerate(waveforms):
        librosa.display.waveshow(waveform, sr=sr, label=f"Waveform {i}")

    # Show legend
    plt.legend()
