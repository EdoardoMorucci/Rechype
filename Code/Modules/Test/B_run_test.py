import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


def infer(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    index_to_label: dict,
):
    """Run inference"""

    # Set inference mode
    with torch.no_grad():
        # Initiate lists
        names = []
        true_indices = []
        predicted_indices = []

        # Loop over batches
        for batch, batch_true_indices, batch_names in tqdm(dataloader):
            # Predictions are shaped batch_size x n_labels
            predictions = model(batch.to(device))

            # Get indices of most probable prediction
            batch_predicted_indices = torch.argsort(predictions, descending=True)[
                ..., 0
            ]

            # Append indices
            predicted_indices.append(batch_predicted_indices)
            true_indices.append(batch_true_indices)
            names.extend(batch_names)

        # Convert to tensor, then move to CPU and convert to numpy array
        true_indices = torch.hstack(true_indices).cpu().numpy()
        predicted_indices = torch.hstack(predicted_indices).cpu().numpy()

        # Convert indices to labels, and reshape so that each row corresponds to a file
        true_labels = np.vectorize(lambda x: index_to_label[x])(true_indices)
        predicted_labels = np.vectorize(lambda x: index_to_label[x])(predicted_indices)

        # Init df
        df = pd.DataFrame(
            {
                "file": names,
                "predicted_index": predicted_indices,
                "true_index": true_indices,
                "predicted_label": predicted_labels,
                "true_label": true_labels,
            }
        )

        return df
