import torch
import numpy as np
from typing import Tuple


def detection_infer(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    device: torch.device,
    index_to_label: dict,
    n_tags: int = 1,
    threshold: float = 0.15,
) -> Tuple[np.array, np.array]:
    """Run inference for sound detection"""

    # Set inference mode
    with torch.no_grad():
        # Predictions are shaped batch_size x n_labels
        predictions = model(waveform.to(device))["framewise_output"]

        # Get confidences and throw away indices
        confidences, indices = torch.sort(predictions, descending=True)

        # Get the actual confidences and indices
        confidences = confidences[..., :n_tags].cpu().numpy()
        indices = indices[..., :n_tags].cpu().numpy()

        # Convert indices to labels, and reshape so that each row corresponds to a file
        labels = np.vectorize(lambda x: index_to_label[x])(indices)

        # Change label to "Undefined" when confidence is below threshold
        labels[confidences < threshold] = "Uncertain"

        return labels, confidences
