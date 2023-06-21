import typer
import os

from decouple import config
from comet_ml import API


def main(model_name: str, saving_directory: str, version: str = None):
    """ Download a model from the Comet ML model registry. This scripts download the entire folder \
        comprising of the model's weights, as well as metadata.yaml and index_to_label.yaml files.
    
    @model_name: str - The name of the model. On Comet this is stored in the form <architecture>-<model_name> (e.g. pann-sonyc) 
    
    @saving_directory: str - The path to the directory where the model will be saved on the local machine 
    
    @version: str - The version of the model to download. If None, the latest version will be downloaded."""

    # Get Comet key from .env file
    COMET_API_KEY = config("GEMMO_COMET_API_KEY")

    # Create the api
    api = API(api_key=COMET_API_KEY)

    # Create target directory if it does not exist
    os.makedirs(saving_directory, exist_ok=True)

    # Download the model
    api.download_registry_model(
        workspace="gemmoai",
        registry_name=model_name,
        output_path=saving_directory,
        version=version,
        expand=True,
    )


if __name__ == "__main__":
    typer.run(main)
