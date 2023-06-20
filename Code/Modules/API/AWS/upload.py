import boto3
import os
import io
import uuid

from pathlib import Path
from starlette.datastructures import UploadFile


def upload_to_s3(audio_file: UploadFile) -> str:
    # Initialise boto3
    s3 = boto3.client("s3")

    # Get bucket name from environment
    bucket_name = os.environ.get("BUCKET")

    # Generate a UUID to use as a unique identifier
    unique_id = uuid.uuid4().hex

    # Get the original filename
    original_filename = Path(audio_file.filename)

    # Create hashed version
    unique_filename = (
        original_filename.stem + "_" + unique_id + original_filename.suffix
    )

    # Create the saving path on S3
    saving_path = os.path.join("data/collected/audio", unique_filename)

    # Always remove the point to the beginning of the file
    # or it will return files with 0 bytes
    audio_file.file.seek(0)

    # Read the contents of the file into memory as bytes
    file_contents = audio_file.file.read()

    # Upload the file to S3
    s3.put_object(Body=io.BytesIO(file_contents), Bucket=bucket_name, Key=saving_path)

    return os.path.join("s3:/", bucket_name, saving_path)
