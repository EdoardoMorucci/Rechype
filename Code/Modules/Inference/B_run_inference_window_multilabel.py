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
        # Init JSON file
        json_results = {}

        # Loop over batches
        for batch, name, splits in tqdm(dataloader):
            # Fix name
            name = name[0]

            # Fix batch shape
            batch = batch.reshape(batch.shape[1:])
            splits = np.array(splits.reshape(splits.shape[1:]))

            # Get the probabilities
            probabilities = model(batch.to(device)).cpu().numpy()

            # Threshold the probabilities to get the predictions
            predictions = probabilities > threshold

            # Convert indices to labels, and reshape so that each row corresponds to a file
            labels = np.where(
                predictions, np.array(list(index_to_label.values())), None
            )

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
                            if class_name
                        ],
                    }
                    for i, (result, confidence) in enumerate(zip(labels, probabilities))
                ],
            }

            # Update results
            json_results.update(batch_results)

        return json_results
