import numpy as np
from typing import List, Tuple


def generate_sampling_intervals(
    n_frames: int, sampling_rate: int, hop_size: int = 320
) -> List[Tuple]:
    """Generate the sampling intervals in the form of an array of tuples, where each tuple interpretation is
    (start, end) expressed in seconds"""

    # Compute the number of seconds each frame lasts (usually around .01 seconds)
    seconds_per_frame = hop_size / sampling_rate

    # Compute the start and end of each sample interval
    # sampling_intervals_start = seconds_per_frame * np.arange(0, n_frames - 1)
    # sampling_intervals_end = seconds_per_frame * np.arange(1, n_frames)
    sampling_intervals_start = seconds_per_frame * np.arange(0, n_frames)
    sampling_intervals_end = seconds_per_frame * np.arange(1, n_frames + 1)

    # Join the two together in a list with (start, end) elements
    sampling_intervals = list(zip(sampling_intervals_start, sampling_intervals_end))

    return sampling_intervals
