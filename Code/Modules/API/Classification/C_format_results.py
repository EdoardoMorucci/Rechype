import numpy as np


def format_classification_results(
    results: np.array, confidences: np.array, names: np.array
):
    """Convert the answer given in terms of numpy array in a JSON-like representation
    using Python dictionary"""

    # Creation of JSON-like results in the form of a dictionary
    json_results = {
        "classification": {
            name: {
                "tags": [
                    {
                        "tag": class_name,
                        "confidence": float_to_percentage(probability),
                    }
                    for class_name, probability in zip(result, confidence)
                    if class_name
                ]
            }
            for name, result, confidence in zip(names, results, confidences)
        }
    }

    return json_results


def float_to_percentage(confidence: float):
    """Convert the confidence expressed as a float between 0 and 1 into
    an (integer) percentage formatted as a string"""

    return f"{round(confidence * 100)}%"
