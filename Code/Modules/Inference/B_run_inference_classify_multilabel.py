import torch
import numpy as np
from tqdm import tqdm


def infer(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    index_to_label: dict,
    threshold: float = 0.5,
):
    """Run inference"""

    # Set inference mode
    with torch.no_grad():
        # Initiate indices over batches
        indices_over_batches = []
        confidences_over_batches = []

        # Get names of the files in the batches
        names = []

        # Loop over batches
        for batch, batch_names in tqdm(dataloader):
            # Get the probabilities
            probabilities = model(batch.to(device))

            # Threshold the probabilities to get the predictions
            predictions = (probabilities > threshold).float()

            # Append indices
            indices_over_batches.append(predictions)

            # Append indices
            confidences_over_batches.append(probabilities)

            # Append names
            names.extend(batch_names)

        # Convert to tensor, then move to CPU and convert to numpy array
        indices = (
            torch.stack(indices_over_batches)
            .cpu()
            .numpy()
            .astype(int)
            .reshape((-1, len(index_to_label)))
        )
        confidences = (
            torch.stack(confidences_over_batches)
            .cpu()
            .numpy()
            .reshape((-1, len(index_to_label)))
        )

        # Convert indices to labels, and reshape so that each row corresponds to a file
        labels = np.where(indices, np.array(list(index_to_label.values())), None)

        # Creation of JSON-like results in the form of a dictionary
        json_results = {
            name: {
                "tags": [
                    {
                        "tag": class_name,
                        "confidence": float(probability),
                    }
                    for class_name, probability in zip(label, confidence)
                    if class_name
                ]
            }
            for name, label, confidence in zip(names, labels, confidences)
        }

        return json_results
