import typer
from comet_ml import Experiment
from decouple import config


def main(architecture_name: str, model_name: str, model_directory: str):
    """Log the model to Comet ML, and register it.

    @architecture_name: str - Then name of the architecture used to train the model (e.g. PANN, BEATs, etc.)

    @model_name: str - The name of the model (e.g. Generic, Sonyc, etc.). Usually the name of the client

    @model_directory: str - The path to the model directory on the local machine.\
        Recall that the model directory should contain the model's weights, as well as metadata.yaml and index_to_label.yaml files

    """

    # Get Comet key from .env file
    COMET_API_KEY = config("GEMMO_COMET_API_KEY")

    # Get the full model name
    full_model_name = f"{architecture_name}-{model_name}"

    # Create the actual project name
    project_name = f"sound-api-{full_model_name}"

    # Start experiment logging
    experiment = Experiment(api_key=COMET_API_KEY, project_name=project_name)
    
    # Fix model directory if necessary
    if model_directory[-1] == "/":
        model_directory = model_directory[:-1]

    # Log model
    experiment.log_model(name=full_model_name, file_or_folder=model_directory)

    # Register model
    experiment.register_model(
        model_name=full_model_name, public=False, tags=["sound-api"], status="Staging"
    )

    # End experiment
    experiment.end()


if __name__ == "__main__":
    typer.run(main)
