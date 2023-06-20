import comet_ml
import torch
import typer

from pathlib import Path
from decouple import config
from torch.utils.data import DataLoader

from Code.Modules.Finetuning.A_train import train
from Code.Networks.wavegram_logmel_cnn14 import Model
from Code.Networks.sound_model import Model as SoundModel
from Code.Dataset.AudioDataset import AudioDataset


def finetune(
    dataset_path: str = "Data/Dataset/Sony",
    freeze_backbone: int = 0,
    batch_size: int = 32,
    epochs: int = 100,
    base_lr: float = 1e-4,
    max_lr: float = 1e-3,
    dropout_rate: float = 0.6,
    augment: bool = True,
    project: str = None,
):
    """Main function to be run with args coming from Typer"""

    # If project is not specified when calling the script
    if not project:
        # Create automatic project name
        # using the (supposed) name of the dataset
        project = "SoundAPI-" + Path(dataset_path).stem

    # Get Comet key from .env file
    COMET_API_KEY = config("COMET_API_KEY")

    # Start experiment logging
    experiment = comet_ml.Experiment(
        api_key=COMET_API_KEY, project_name=project, auto_metric_logging=True
    )
    experiment.add_tag("Train")

    # Log hypers
    experiment.log_parameter("batch_size", batch_size)
    experiment.log_parameter("epochs", epochs)
    experiment.log_parameter("base_lr", base_lr)
    experiment.log_parameter("max_lr", max_lr)
    experiment.log_parameter("dropout_rate", dropout_rate)
    experiment.log_parameter("dataset_path", dataset_path)

    # Set seeds
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialise dataset
    train_dataset = AudioDataset(dataset_path, "Train", augment=augment)
    valid_dataset = AudioDataset(dataset_path, "Validation")

    # Get the number of unique labels present in the dataset
    n_labels = train_dataset.n_labels

    # Dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )

    # Load wavegram model to be used as a backbone
    # This cannot be touched as it depends on the pre-trained model
    backbone = Model(
        sample_rate=32000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        classes_num=527,
    )

    # Load pre-trained weights for backbone
    checkpoint = torch.load(
        "Models/Generic/Wavegram_Logmel_Cnn14_mAP=0.439.pth", map_location=device
    )
    backbone.load_state_dict(checkpoint["model"])

    # Load fine-tuned model
    tuned_model = SoundModel(
        backbone,
        n_labels=n_labels,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
    )
    tuned_model.to(device)

    # Compute class weights
    class_weights = (
        1 - train_dataset.labels["class"].value_counts(normalize=True).sort_index()
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Load cross entropy loss
    loss = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Define the optimizer
    optimizer = torch.optim.AdamW(tuned_model.parameters(), lr=base_lr)

    # Define the scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # Get trainable parameters
    trainable_parameters = sum(
        p.numel() for p in tuned_model.parameters() if p.requires_grad
    )
    print(f"Total trainable parameters: {trainable_parameters}")

    # Finetune model
    train(
        tuned_model,
        train_dataloader,
        valid_dataloader,
        optimizer,
        scheduler,
        loss,
        epochs,
        device,
        experiment,
    )

    return experiment.name


if __name__ == "__main__":
    # Run main
    typer.run(finetune)
