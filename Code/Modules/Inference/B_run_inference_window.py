import torch
import numpy as np
from tqdm import tqdm


def infer(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    index_to_label: dict,
    n_tags: int = 2,
):
    """Run inference"""

    # Set inference mode
    with torch.no_grad():
        # Init JSON file
        json_results = {}

        # Loop over batches
        for batch, name, splits in tqdm(dataloader):
            # Fix name
            name = name[0]

            # Fix batch shape
            batch = batch.reshape(batch.shape[1:])
            splits = np.array(splits.reshape(splits.shape[1:]))

            # Predictions are shaped batch_size x n_labels
            predictions = model(batch.to(device))

            # If the model is generic, neglect the embedding
            # and keep the predictions only
            if isinstance(predictions, tuple):
                predictions = predictions[0]

            # Get confidences and throw away indices
            confidences, indices = torch.sort(predictions, descending=True)

            # Get the actual confidences and indices
            confidences = confidences[..., :n_tags].cpu().numpy()
            indices = indices[..., :n_tags].cpu().numpy()

            # Convert indices to labels, and reshape so that each row corresponds to a file
            labels = np.vectorize(lambda x: index_to_label[x])(indices).reshape(
                (-1, n_tags)
            )
            confidences = confidences.reshape(-1, n_tags)

            # Create results
            batch_results = {
                name: [
                    {
                        "start": float(splits[i]),
                        "end": float(splits[i + 1]),
                        "tags": [
                            {
                                "tag": class_name,
                                "confidence": float(probability),
                            }
                            for class_name, probability in zip(result, confidence)
                        ],
                    }
                    for i, (result, confidence) in enumerate(zip(labels, confidences))
                ],
            }

            # Update results
            json_results.update(batch_results)

        return json_results
