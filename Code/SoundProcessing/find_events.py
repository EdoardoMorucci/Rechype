import librosa
import typer
import matplotlib.pyplot as plt
import numpy as np


def find_events(
    filename: str,
    pre_max: float = 3,
    post_max: float = 3,
    pre_avg: float = 3,
    post_avg: float = 5,
    delta: float = 0.5,
    wait: float = 10,
    hop_length: int = 512,
    window_left: float = 2,
    window_right: float = 2,
):
    """
    Find events in an audio waveform by first detecting peaks and then creating windows around them.

    Args:
        filename (str): The path to the audio file.
        pre_max (float, optional): The size (in seconds) of the pre-maximum window for peak picking. Defaults to 3.
        post_max (float, optional): The size (in seconds) of the post-maximum window for peak picking. Defaults to 3.
        pre_avg (float, optional): The size (in seconds) of the pre-average window for peak picking. Defaults to 3.
        post_avg (float, optional): The size (in seconds) of the post-average window for peak picking. Defaults to 5.
        delta (float, optional): The minimum difference between a peak and its surrounding values for peak picking. Defaults to 0.5.
        wait (float, optional): The number of samples to wait before accepting a new peak for peak picking. Defaults to 10.
        hop_length (int, optional): The number of samples between successive frames for onset detection. Defaults to 512.
        window_left (float, optional): The length (in seconds) of the window on the left side of each peak. Defaults to 2.
        window_right (float, optional): The length (in seconds) of the window on the right side of each peak. Defaults to 2.

    Returns:
        np.ndarray: A two-dimensional NumPy array containing the windowed data for each event.
            The number of rows corresponds to the number of events, and the number of columns
            corresponds to the size of each window.

        int: The sample rate of the waveform in Hz.

        np.ndarray: An array of times (in seconds) corresponding to the peaks of interest.
    """

    # Find peaks
    waveform, sr, peak_times = find_peak_times(
        filename=filename,
        pre_max=pre_max,
        post_max=post_max,
        pre_avg=pre_avg,
        post_avg=post_avg,
        delta=delta,
        wait=wait,
        hop_length=hop_length,
    )

    # Create events
    events = create_windows(waveform, sr, peak_times, window_left, window_right)

    return events, sr, peak_times


def create_windows(
    waveform, sr, peak_times, window_left: float = 2, window_right: float = 2
):
    """Create a window of waveform data around the peaks of interest.
    If the limits of the waveform are exceeded from the left or the right, the waveform is put first, and then padded with zeros
    on the right

    Args:
        waveform (ndarray): The waveform data.
        sr (int): The sample rate of the waveform data in Hz.
        peak_times (ndarray): An array of times (in seconds) corresponding to the peaks of interest.
        window_left (float, optional): The length (in seconds) of the window on the left side of each peak. Defaults to 2.
        window_right (float, optional): The length (in seconds) of the window on the right side of each peak. Defaults to 2.

    Returns:
        list: A list of NumPy arrays, where each array contains the windowed data for each peak.
    """

    # Get the size of the window
    window_size = int((window_left + window_right) * sr)

    # Get the indices of the peak in the waveform numpy array
    peak_indices = (peak_times * sr).astype(int)

    # Extract the indices of the righ/left-most part of the window
    left_indices = peak_indices - window_left * sr
    right_indices = peak_indices + window_right * sr

    # Ensure the indices do not exceed 0 on the left, and the length
    # of the waveform on the right
    left_indices = np.clip(left_indices, 0, len(waveform)).astype(int)
    right_indices = np.clip(right_indices, 0, len(waveform)).astype(int)

    # Extract the windowed data for each peak using array slicing
    peak_windows = [
        np.concatenate([waveform[start:end], np.zeros(window_size - (end - start))])
        for start, end in zip(left_indices, right_indices)
    ]

    # Return the stack
    return np.vstack(peak_windows)


def find_peak_times(
    filename: str,
    pre_max: float = 3,
    post_max: float = 3,
    pre_avg: float = 3,
    post_avg: float = 5,
    delta: float = 0.5,
    wait: float = 10,
    hop_length: int = 512,
):
    """Find the times of peaks in an audio waveform.

    Args:
        filename (str): The path to the audio file.
        pre_max (float, optional): The size (in seconds) of the pre-maximum window for peak picking. Defaults to 3.
        post_max (float, optional): The size (in seconds) of the post-maximum window for peak picking. Defaults to 3.
        pre_avg (float, optional): The size (in seconds) of the pre-average window for peak picking. Defaults to 3.
        post_avg (float, optional): The size (in seconds) of the post-average window for peak picking. Defaults to 5.
        delta (float, optional): The minimum difference between a peak and its surrounding values for peak picking. Defaults to 0.5.
        wait (float, optional): The number of samples to wait before accepting a new peak for peak picking. Defaults to 10.
        hop_length (int, optional): The number of samples between successive frames for onset detection. Defaults to 512.

    Returns:
        tuple: A tuple containing the waveform as a NumPy array, the sample rate as an integer,
        and an array of times (in seconds) corresponding to the peaks of the waveform.
    """

    # Read waveform from path, sr is sample rate
    # Enforce mono audio, and use native sample rate
    waveform, sr = librosa.load(filename, mono=True, sr=None)

    # Compute onset
    onset_env = librosa.onset.onset_strength(
        y=waveform, sr=sr, hop_length=hop_length, aggregate=np.median
    )

    # Find peak using librosa
    peaks = librosa.util.peak_pick(
        onset_env,
        pre_max=pre_max,
        post_max=post_max,
        pre_avg=pre_avg,
        post_avg=post_avg,
        delta=delta,
        wait=wait,
    )

    # Get all the times for the given hop_length
    times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)

    # Get times corresponding to the peaks
    peak_times = times[peaks]

    return waveform, sr, peak_times


def plot_spectrogram(db, sr):
    """
    Plot the spectrogram of an audio signal.

    Args:
    db (ndarray): The spectrogram of the audio signal.
    sr (int): The sampling rate of the audio signal.
    """

    # Create a new figure with size 10 inches by 5 inches
    plt.figure(figsize=(10, 5))

    # Display the spectrogram using the librosa.display.specshow() function,
    # which takes the spectrogram, y-axis type (logarithmic scale), x-axis type
    # (time), and the sample rate as input arguments.
    librosa.display.specshow(db, y_axis="log", x_axis="time", sr=sr)

    # Add a colorbar to the plot with the format '+2.0f dB'
    plt.colorbar(format="%+2.0f dB")

    # Add a title to the plot
    plt.title("Spectrogram")

    # Add a label to the x-axis
    plt.xlabel("Time")

    # Add a label to the y-axis
    plt.ylabel("Frequency")

    # Adjust the plot layout
    plt.tight_layout()

    # Show the plot
    plt.show()


def print_graph_spectrogram(onset_env, sr, db, peaks, hop_length: int = 512):
    # Compute the time values for each frame of the onset envelope
    # using the librosa.times_like() function
    times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)

    # Create a two-row subplot using the subplots() function
    # from the matplotlib.pyplot library and assign the axes to the variable ax
    _, ax = plt.subplots(nrows=2, sharex=True)

    # Display the spectrogram using the librosa.display.specshow() function
    # which takes the decibel spectrogram, y-axis type (logarithmic scale),
    # x-axis type (time), and the first subplot axis as input arguments
    librosa.display.specshow(db, y_axis="log", x_axis="time", ax=ax[1])

    # Plot the onset strength envelope over time on the first subplot axis using
    # the plot() function from matplotlib.pyplot, which takes the time values and
    # onset envelope as input arguments
    ax[0].plot(times, onset_env, alpha=0.8, label="Onset strength")

    # Plot vertical lines at the positions of the selected peaks on the onset envelope
    # using the vlines() function from matplotlib.pyplot, which takes the time values
    # at the peak indices, 0 as the minimum y-value, the maximum value of the onset envelope
    # as the maximum y-value, the color red, and the first subplot axis as input arguments
    ax[0].vlines(
        times[peaks], 0, onset_env.max(), color="r", alpha=0.8, label="Selected peaks"
    )

    # Add a legend to the first subplot axis using the legend() function from
    # matplotlib.pyplot, which creates a legend with a frame and a transparency of 0.8
    ax[0].legend(frameon=True, framealpha=0.8)

    # Remove the labels of the first subplot axis using the label_outer() method
    ax[0].label_outer()

    # The function does not return any value. Instead, it displays the onset envelope
    # and spectrogram with selected peaks using the show() method from matplotlib.pyplot
    plt.show()


if __name__ == "__main__":
    typer.run(find_events)
