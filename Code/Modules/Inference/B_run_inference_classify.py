import torch
import numpy as np
from tqdm import tqdm


def infer(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    index_to_label: dict,
    n_tags: int = 3,
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
            # Predictions are shaped batch_size x n_labels
            predictions = model(batch.to(device))

            # If the model is generic, neglect the embedding
            # and keep the predictions only
            if isinstance(predictions, tuple):
                predictions = predictions[0]

            # Get confidences and throw away indices
            confidences, indices = torch.sort(predictions, descending=True)

            # Get the actual confidences and indices
            confidences = confidences[..., :n_tags]
            indices = indices[..., :n_tags]

            # Append indices
            indices_over_batches.append(indices)

            # Append indices
            confidences_over_batches.append(confidences)

            # Append names
            names.extend(batch_names)

        # Convert to tensor, then move to CPU and convert to numpy array
        indices = torch.stack(indices_over_batches).cpu().numpy()
        confidences = torch.stack(confidences_over_batches).cpu().numpy()

        # Convert indices to labels, and reshape so that each row corresponds to a file
        labels = np.vectorize(lambda x: index_to_label[x])(indices).reshape(
            (-1, n_tags)
        )

        # Reshape confidences
        confidences = confidences.reshape(-1, n_tags)

        # Creation of JSON-like results in the form of a dictionary
        json_results = {
            name: {
                "tags": [
                    {
                        "tag": class_name,
                        "confidence": float(probability),
                    }
                    for class_name, probability in zip(label, confidence)
                ]
            }
            for name, label, confidence in zip(names, labels, confidences)
        }

        return json_results
