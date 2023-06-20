import requests
import os

from glob import glob
from pathlib import Path
from decouple import config

# Set the URL for the API
URL = "https://gemmoai--sound-api-fastapi-app.modal.run"

# Define the endpoint
ENDPOINT = "window/single"

# Define the model
MODEL = "Generic"

# Construct the endpoint URL
ENDPOINT_URL = os.path.join(URL, ENDPOINT)

# Get all files
ALL_FILES = glob("../../Files/*.mp3") + glob("../../Files/*.wav")


def test_window_single_fail():
    """Test the API with a single file and one tag"""

    # Get acceptable token
    TOKEN = config("NO_GENERIC_TOKEN")

    # Get response
    response = call_api(token=TOKEN, window_length=3, n_tags=1, files=ALL_FILES[:1])

    # Check
    assert response.status_code == 403, response.text


def test_window_single_one_file_one_tag():
    """Test the API with a single file and one tag"""

    # Get acceptable token
    TOKEN = config("GENERIC_TOKEN")

    # Get response
    response = call_api(token=TOKEN, window_length=3, n_tags=1, files=ALL_FILES[:1])

    # Check
    assert response.status_code == 200, response.text


def test_window_single_one_file_three_tag():
    """Test the API with a single file and one tag"""

    # Get acceptable token
    TOKEN = config("GENERIC_TOKEN")

    # Get response
    response = call_api(token=TOKEN, window_length=3, n_tags=3, files=ALL_FILES[:1])

    # Check
    assert response.status_code == 200, response.text


def call_api(token: str, window_length: int, n_tags: int, files: list):
    """Define a single call to the api with specified parameters"""

    # Define the payload
    payload = {
        "window_length": str(window_length),
        "model_name": MODEL,
        "n_tags": n_tags,
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
