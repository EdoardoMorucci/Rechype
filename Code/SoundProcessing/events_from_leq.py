import numpy as np
from typing import List


def find_events(
    waveform: np.array,
    sr: int,
    chunk_size: float,
    threshold: float,
    enforce_length: bool = True,
):
    """Find the events in a waveform. An event is defined as a chunk of the waveform
    whose Leq is above a given threshold.

    @waveform: the waveform to split
    @sr: the sample rate of the waveform
    @chunk_size: the size of each chunk in seconds
    @threshold: the threshold above which an event is detected. It is expressed in dBFS, hence it is
    a negative number.
    """

    # Split the waveform into chunks
    chunks = split_waveform(waveform, sr, chunk_size)

    # Compute the Leq of each chunk
    # leqs = np.vectorize(compute_Leq)(chunks)
    leqs = np.asarray([compute_Leq(chunk) for chunk in chunks])

    # Get the indices of the chunks. It is the index of the first sample of each chunk
    # plus the length of the chunk
    indices = np.insert(np.cumsum([len(chunk) for chunk in chunks]), 0, 0)

    # Create an array of intervals. Each interval is defined by the index of the first sample and
    # the index of the last sample of the chunk
    intervals = np.stack((indices[:-1], indices[1:]), axis=-1)

    # Find the indices of the chunks whose Leq is above the threshold
    above_threshold_indices = np.argwhere(leqs >= threshold)

    # Find the intervals corresponding to the chunks whose Leq is above the threshold
    above_threshold_intervals = intervals[above_threshold_indices].reshape(-1, 2)

    # Init the waveform with only relevant events
    thresholded_waveform = np.zeros_like(waveform)

    # Init events
    events = []

    # For each interval, add the corresponding chunk to the thresholded waveform
    for interval in above_threshold_intervals:
        # Get the signal
        signal = waveform[interval[0] : interval[1]]

        # Add the interval to the waveform
        thresholded_waveform[interval[0] : interval[1]] = signal

        # Add to the events the corresponding chunk
        if enforce_length:
            events.append(signal[: int(chunk_size * sr)])
        else:
            events.append(signal)

    return thresholded_waveform, events


def compute_Leq(waveform: np.array) -> float:
    """Computes the Leq of a waveform"""

    # Add small number for numerical stability
    return 20 * np.log10(np.sqrt(np.mean(waveform) ** 2) + 1e-12)


def split_waveform(waveform: np.array, sr: int, chunk_size: float) -> List:
    """Split a waveform into chunks of a given size expressed in seconds. The splits
    are non overlapping.

    @waveform: the waveform to split
    @sr: the sample rate of the waveform
    @chunk_size: the size of each chunk in seconds
    """

    return np.array_split(waveform, len(waveform) // (chunk_size * sr))
