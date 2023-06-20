import torch
import numpy as np
import polars as pl
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
        # Initiate lists
        names = []
        predictions = []
        ground_truths = []
        confidences = []

        # Loop over batches
        for batch, batch_ground_truths, batch_names in tqdm(dataloader):
            
            # Get batch confidences
            batch_confidences = model(batch.to(device))
            
            # Turn them into predictions
            batch_predictions = (batch_confidences > threshold).float()

            # Append indices
            confidences.append(batch_confidences)
            predictions.append(batch_predictions)
            ground_truths.append(batch_ground_truths)
            names.extend(batch_names)

        # Convert to tensor, then move to CPU and convert to numpy array
        confidences = torch.vstack(confidences).cpu().numpy()
        predictions = torch.vstack(predictions).cpu().numpy()
        ground_truths = torch.vstack(ground_truths).cpu().numpy().astype(np.float32)

        # Convert to actual labels
        labels = np.where(predictions, np.array(list(index_to_label.values())), "")
        true_labels = np.where(
            ground_truths, np.array(list(index_to_label.values())), ""
        )

        # Init df
        df = pl.DataFrame(
            {
                "file": names,
                "confidences": confidences,
                "predictions": predictions,
                "ground_truths": ground_truths,
                "predicted_labels": labels,
                "true_labels": true_labels,
            }
        )

        return df
