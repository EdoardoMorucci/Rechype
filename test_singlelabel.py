import comet_ml
import torch
import typer

from decouple import config
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

from Code.Dataset.AudioDataset import AudioDataset
from Code.Modules.Test.A_load_model import load_model
from Code.Modules.Test.B_run_test import infer


def main(
    model_name: str = "Sony",
    batch_size: int = 32,
    project: str = "SoundAPI-Sony",
):
    """Main function to be run with args coming from Typer"""

    # Get Comet key from .env file
    COMET_API_KEY = config("COMET_API_KEY")

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
    dataset = AudioDataset(dataset_path, split="Test")

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
    )

    # Log confusion matrix onto Comet
    experiment.log_confusion_matrix(
        y_true=results["true_index"].to_numpy(),
        y_predicted=results["predicted_index"].to_numpy(),
        labels=list(index_to_label.values()),
        title="Confusion Matrix",
        row_label="Actual Category",
        column_label="Predicted Category",
    )

    # Get our taxonomy
    taxonomy = results["true_label"].unique()

    # Compute scores
    precisions = precision_score(
        results["true_label"], results["predicted_label"], labels=taxonomy, average=None
    )
    recalls = recall_score(
        results["true_label"], results["predicted_label"], labels=taxonomy, average=None
    )
    f1s = f1_score(
        results["true_label"], results["predicted_label"], labels=taxonomy, average=None
    )

    # Log metrics on Comet
    for label, f1, precision, recall in zip(taxonomy, f1s, precisions, recalls):
        experiment.log_metric(f"f1_{label}", f1)
        experiment.log_metric(f"precision_{label}", precision)
        experiment.log_metric(f"recall_{label}", recall)


if __name__ == "__main__":
    # Run main
    typer.run(main)
