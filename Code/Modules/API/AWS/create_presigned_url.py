import requests
import logging
import boto3

from botocore.exceptions import ClientError
from tqdm import tqdm


def get_urls(
    files: list,
    key: str = "data/datasets/Cirrus/Audio",
    expiration: int = 604800,
    bucket_name: str = "gemmoai-soundapi",
):
    # Init list with urls
    urls = []

    for file in tqdm(files):
        # Generate object key
        object_key = key + "/" + file

        # Generate presigned URL
        url = create_presigned_url(bucket_name, object_key, expiration=expiration)

        if url is not None:
            response = requests.get(url)
            urls.append(response.url)
        else:
            print("Error, URL=None, " + file)

    return urls


def create_presigned_url(bucket_name, object_key, expiration=604800):
    """Generate a presigned URL to share an S3 object

    :param bucket_name: string
    :param object_name: string
    :param expiration: Time in seconds for the presigned URL to remain valid
    :return: Presigned URL as string. If error, returns None.
    """

    # Generate a presigned URL for the S3 object
    s3_client = boto3.client("s3")
    try:
        response = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": object_key},
            ExpiresIn=expiration,
        )
    except ClientError as e:
        logging.error(e)
        return None

    # The response contains the presigned URL
    return response
