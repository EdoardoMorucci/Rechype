import comet_ml
import torch
import typer
import numpy as np

from decouple import config
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from Code.Dataset.MultiLabelDataset import MultiLabelDataset
from Code.Modules.Test.A_load_model import load_model
from Code.Modules.Test.B_run_multilabel_test import infer


def main(
    model_name: str = "Enfonic",
    batch_size: int = 32,
    project: str = None,
    threshold: float = 0.5,
):
    """Main function to be run with args coming from Typer"""

    # Get Comet key from .env file
    COMET_API_KEY = config("COMET_API_KEY")

    # If project is not specified when calling the script
    if not project:
        # Create automatic project name
        # using the (supposed) name of the dataset
        project = "SoundAPI-" + model_name

    # Start experiment logging
    experiment = comet_ml.Experiment(
        api_key=COMET_API_KEY, project_name=project, auto_metric_logging=True
    )
    experiment.add_tag("Test")

    # Set seeds
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model, decoding from indices to actual labels and path to dataset
    model, index_to_label, dataset_path = load_model(model_name)

    # Initialise dataset in inference mode
    dataset = MultiLabelDataset(dataset_path, split="Test")

    # Create dataloader
    # If on GPU, enable pin memory to speed up inference
    if device == "cuda":
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
    # Else, leave it false
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=False,
        )

    # Run inference and fetch results in the form of pandas df
    results = infer(
        model=model,
        dataloader=dataloader,
        device=device,
        index_to_label=index_to_label,
        threshold=threshold,
    )

    # Get predictions and ground truths
    predictions = np.vstack(results["predictions"].to_numpy())
    ground_truths = np.vstack(results["ground_truths"].to_numpy())

    # Compute the f1 scores
    f1 = f1_score(
        predictions,
        ground_truths,
        average=None,
    )

    # Compute precision
    precision = precision_score(
        predictions,
        ground_truths,
        average=None,
    )

    # Compute recall
    recall = recall_score(
        predictions,
        ground_truths,
        average=None,
    )

    # Compute accuracy
    accuracy = accuracy_score(
        predictions,
        ground_truths,
    )

    # Write to parquet file
    results.write_parquet(f"{model_name}.parquet")

    # For each separate label
    for i in range(ground_truths.shape[1]):
        # Get the label
        label = index_to_label[i]

        # Log the confusion matrices
        experiment.log_confusion_matrix(
            ground_truths[:, i].astype(int),
            predictions[:, i].astype(int),
            title=f"Confusion Matrix, {label}",
            file_name=f"confusion-matrix-{label}.json",
        )

    # Upload the results onto CometML
    for i, f1_label in enumerate(f1):
        experiment.log_metric(f"F1_label_{index_to_label[i]}", f1_label)

    # Upload the results onto CometML
    for i, precision_label in enumerate(precision):
        experiment.log_metric(f"Precision_label_{index_to_label[i]}", precision_label)

    # Upload the results onto CometML
    for i, recall_label in enumerate(recall):
        experiment.log_metric(f"Recall_label_{index_to_label[i]}", recall_label)

    # Upload the results onto CometML
    experiment.log_metric(f"Accuracy", accuracy)


if __name__ == "__main__":
    # Run main
    typer.run(main)
