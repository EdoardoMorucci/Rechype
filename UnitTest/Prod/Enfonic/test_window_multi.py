import requests
import os

from glob import glob
from pathlib import Path
from decouple import config

# Set the URL for the API
URL = "https://gemmoai--sound-api-fastapi-app.modal.run"

# Define the endpoint
ENDPOINT = "window/multi"

# Define the model
MODEL = "Enfonic"

# Construct the endpoint URL
ENDPOINT_URL = os.path.join(URL, ENDPOINT)

# Get all files
ALL_FILES = glob("../../Files/*.mp3") + glob("../../Files/*.wav")


def test_window_multi_fail():
    """Test the API with a single file and one tag"""

    # Get acceptable token
    TOKEN = config("NO_ENFONIC_TOKEN")

    # Get response
    response = call_api(token=TOKEN, window_length=3, files=ALL_FILES[:1])

    # Check
    assert response.status_code == 403, response.text


def test_window_multi_one_file():
    """Test the API with a single file and one tag"""

    # Get acceptable token
    TOKEN = config("ENFONIC_TOKEN")

    # Get response
    response = call_api(token=TOKEN, window_length=3, files=ALL_FILES[:1])

    # Check
    assert response.status_code == 200, response.text


def call_api(token: str, window_length: int, files: list):
    """Define a single call to the api with specified parameters"""

    # Define the payload
    payload = {
        "window_length": str(window_length),
        "model_name": MODEL,
    }

    # Create the files to be opened and passed to the model
    files = [
        ("audio_file", (Path(file).name, open(file, "rb"), "audio/mpeg"))
        for file in files
    ]

    # Compile headers with auth token
    headers = {"Authorization": token}

    # Get the response from the API
    response = requests.request(
        "POST", ENDPOINT_URL, headers=headers, data=payload, files=files
    )

    return response
