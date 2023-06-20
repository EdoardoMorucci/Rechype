import numpy as np
from typing import List, Tuple, Dict


def format_detection_results(
    labels: np.array, confidences: np.array, intervals: List[Tuple]
) -> Dict:
    """Format results in JSON-like file"""

    # Get the relevant indices
    indices = np.where(labels != "Uncertain")

    # Get the relevant info
    relevant_labels = labels[indices].tolist()
    relevant_confidences = confidences[indices].tolist()
    relevant_intervals = np.round(np.array(intervals)[indices[1]], 3).tolist()

    # Generate events dictionary
    events = {
        "events": [
            {
                "tag": label,
                "confidence": float_to_percentage(confidence),
                "start": start,
                "end": end,
            }
            for label, confidence, (start, end) in zip(
                relevant_labels, relevant_confidences, relevant_intervals
            )
        ]
    }

    return events


def float_to_percentage(confidence: float):
    """Convert the confidence expressed as a float between 0 and 1 into
    an (integer) percentage formatted as a string"""

    return f"{round(confidence * 100)}%"
