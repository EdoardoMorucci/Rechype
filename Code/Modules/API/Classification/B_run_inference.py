import torch
import numpy as np
from typing import Tuple


def infer(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    device: torch.device,
    index_to_label: dict,
    n_tags: int = 3,
) -> Tuple[np.array, np.array]:
    """Run inference"""

    # Set inference mode
    with torch.no_grad():
        # Allow for presence of embeddings in case of pretrained models
        results = model(waveform.to(device))
        if isinstance(results, tuple):
            # Predictions are shaped batch_size x n_labels
            (
                predictions,
                _,
            ) = results  # Unpack both objects (one is the embedding, not useful here)
        else:
            # Predictions are shaped batch_size x n_labels
            predictions = results  # Only one object returned, i.e. th predictions

        # Get confidences and throw away indices
        confidences, indices = torch.sort(predictions, descending=True)

        # Get the actual confidences and indices
        confidences = confidences[..., :n_tags].cpu().numpy()
        indices = indices[..., :n_tags].cpu().numpy()

        # Convert indices to labels, and reshape so that each row corresponds to a file
        labels = np.vectorize(lambda x: index_to_label[x])(indices)

        # Return restricting to first direction since that's the batch direction
        return labels, confidences
