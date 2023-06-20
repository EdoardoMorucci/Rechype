import torch
import numpy as np
from typing import Tuple


def infer(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    device: torch.device,
    index_to_label: dict,
) -> Tuple[np.array, np.array]:
    """Run inference"""

    # Set inference mode
    with torch.no_grad():
        # Get probabilities
        probabilities = (
            model(waveform.to(device)).cpu().numpy().reshape((-1, len(index_to_label)))
        )

        # Convert to predictions
        predictions = (probabilities > 0.5).astype(int)

        # Convert indices to labels, and reshape so that each row corresponds to a file
        labels = np.where(predictions, np.array(list(index_to_label.values())), None)

        return labels, probabilities
