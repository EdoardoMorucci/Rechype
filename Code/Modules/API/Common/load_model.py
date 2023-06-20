import os
import torch
from collections import OrderedDict
from yaml import safe_load
from importlib import import_module


def load_model(model_name: str, device: torch.device):
    """Load model with weights"""

    # Get list of available models
    MODELS = os.listdir("Models")

    # Ascertain we have the model
    assert model_name in MODELS, "Model name not admissible"

    # Get path to model metadata
    metadata_path = os.path.join("Models", model_name, "metadata.yaml")

    # Get model metadata
    with open(metadata_path, "r") as file:
        metadata = safe_load(file)

    # Get path for decoding indices to labels
    index_to_label_path = os.path.join("Models", model_name, "index_to_label.yaml")

    # Get index to label
    with open(index_to_label_path, "r") as file:
        index_to_label = safe_load(file)

    # Create path to module
    module_path = os.path.join("Code", "Networks", metadata["module"]).replace("/", ".")

    # Import the "Model" class from the selected module
    Model = getattr(import_module(module_path), "Model")

    # Init model
    model = Model(**metadata["parameters"])

    # Load checkpoint
    checkpoint_path = os.path.join("Models", model_name, metadata["weights"])
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load weights to model
    if "model" in checkpoint.keys():
        model.load_state_dict(checkpoint["model"])
    else:
        try:
            model.load_state_dict(checkpoint)
        except:
            # Keys are wrong
            new_checkpoint = OrderedDict()
            for key, value in checkpoint.items():
                # Rewrite the key
                new_key = "model." + key

                # Create new checkpoint object with that key
                new_checkpoint[new_key] = value

            # Load the weights
            model.load_state_dict(new_checkpoint)

    # Put model in evaluation mode
    model.eval()

    # Send model to device
    model.to(device)

    return model, index_to_label
