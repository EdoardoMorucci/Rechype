import torch
import typer

from torch.utils.data import DataLoader

from Code.Modules.Inference.A_load_model import load_model
from Code.Modules.Inference.B_run_inference_classify_multilabel import infer
from Code.Dataset.AudioDataset import AudioDataset


def main(
    dataset_path: str = "Data/Dataset/Enfonic",
    model_name: str = "Enfonic",
    batch_size: int = 1,
    threshold: float = 0.5,
    to_json: bool = True,
):
    """Main function to be run with args coming from Typer"""

    # Set seeds
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialise dataset in inference mode
    dataset = AudioDataset(dataset_path, inference=True)

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

    # Load model and decoding from indices to actual labels
    model, index_to_label = load_model(model_name)

    # Run inference and fetch results
    results = infer(
        model=model,
        dataloader=dataloader,
        device=device,
        index_to_label=index_to_label,
        threshold=threshold,
    )

    # If json is True, the results are stored on file, else they are simply
    # printed on screen
    if to_json:
        import json

        # Save dictionary to a JSON file
        with open(f"results_{model_name}.json", "w") as f:
            json.dump(results, f)

    else:
        print(results)


if __name__ == "__main__":
    # Run main
    typer.run(main)
