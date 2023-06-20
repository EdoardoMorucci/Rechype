import os
import modal


def initialise_modal():
    """Initialise Modal stub (necessary for using Modal), and return the image
    it is using, along with the secrets stored on Modal"""

    # Initialise Modal stub (necessary for using Modal)
    stub = modal.Stub("sound-api")

    # Create image (Docker-like) to be used by Modal backend
    image = modal.Image.debian_slim(python_version="3.10")

    # Pip install packages
    image = image.pip_install(
        "numpy",
        "tqdm",
        "pandas",
        "torch",
        "torchaudio",
        "openpyxl",
        "python-decouple",
        "torchlibrosa",
        "pyyaml",
        "fastapi[all]",
        "uvicorn",
        "python-multipart",
        "chronopy",
        "pytz",
        "matplotlib",
        "seaborn",
        "sentry-sdk[fastapi]",
        "boto3",
    )

    # Model directory
    model_directory = "Models"

    # Set the available models to be copied
    AVAILABLE_MODELS = ["Generic", "Enfonic"]

    # Put all the models available for production in the image
    for model in AVAILABLE_MODELS:
        # Copy from local directory
        image = image.copy_local_dir(
            local_path=os.path.join(model_directory, model),
            remote_path=f"/root/Models/{model}",
        )

    # Assign image to stub
    stub.image = image

    # Get the secretes
    secret = modal.Secret.from_name("apiadmin-secret")

    return stub, image, secret
