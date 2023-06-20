import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def generate_detection_plot(
    labels: np.array,
    confidences: np.array,
    intervals: np.array,
    sampling_rate: int,
    hop_size: int = 320,
) -> plt.figure:
    """Generate detection heat-map where the value (heat) is given by the confidence
    in the prediction"""

    # Fetch unique tags
    tags = np.unique(labels)

    # Remove the uncertain part
    tags = tags[tags != "Uncertain"]

    # Init figure
    figure, ax = plt.subplots(figsize=(10, 0.3 * len(tags)))

    # Construct 2D array of probabilities, where each row is a non-trivial tag
    predictions = np.vstack(
        [np.where(labels[0, :, 0] == tag, confidences[0, :, 0], 0) for tag in tags]
    )

    # Create labels so that every 2 seconds we have a tick (change 2 to whatever seconds you like)
    xticklabels = round(2 * sampling_rate / hop_size)

    # Generate heatmap
    sns.heatmap(predictions, yticklabels=tags, xticklabels=xticklabels)

    # Set x-axis to be time in seconds
    ax.set(xlabel="Time (s)")

    # Get index for each tick in the old labels for x_axis
    labels_indices = [int(text.get_text()) for text in ax.get_xticklabels()]

    # Extract sampling intervals starts
    sampling_intervals_start, _ = zip(*intervals)
    sampling_intervals_start = np.asarray(sampling_intervals_start)

    # Generate labels, and round to 2 relevant figures
    sampling_labels = np.round(sampling_intervals_start[labels_indices], 2)

    # Fix x-ticks
    ax.set_xticklabels(sampling_labels)

    # Fix title
    ax.set_title("Detection", weight="bold")

    return figure
