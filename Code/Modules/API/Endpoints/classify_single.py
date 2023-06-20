import torch

from fastapi import APIRouter
from fastapi import Depends, Form, UploadFile, HTTPException

from Code.Modules.API.Classification.A_load_batch import load_batch
from Code.Modules.API.Common.load_model import load_model
from Code.Modules.API.Classification.B_run_inference import infer
from Code.Modules.API.Classification.C_format_results import (
    format_classification_results,
)
from Code.Modules.API.Common.api_call_metadata import get_call_metadata

from Authentication.Checkers import (
    classify_auth_check,
)
from Authentication.Token_auth_service_interface import TokenAuthServiceInterface
from Authentication.Types import RequestTokenData

from Code.Modules.API.ResponseModels.responses import (
    classify_error_responses,
    ClassifyResponse,
)

from Code.Modules.API.AWS.upload import upload_to_s3


# Initialise router
router = APIRouter()


@router.post(
    "/classify/single",
    response_model=ClassifyResponse,
    responses=classify_error_responses,
)
async def classify(
    audio_files: list[UploadFile],
    request_token_data: RequestTokenData = Depends(classify_auth_check),
    model_name: str = Form(...),
    n_tags: int = Form(...),
):
    """
    Endpoint used to classify the audio source in one or more audio files (up to a maximum of 16 files). Files are always
    processed in batch. There is a hard cut-off on the length of the audio files, at the moment fixed at 5 seconds.
    Content past 5s is not even loaded into memory but simply discarded.

    @audio_files: list of audio files loaded in binary format

    @model_name (string): name of the model to be used for classification

    @n_tags (integer): int specifying the number of tags to be returned for every window
    """

    # Create request metadata (containing the parameters passed to it)
    request_metadata = {
        "num_audio_files": len(audio_files),
        "model_name": model_name,
        "n_tags": n_tags,
    }

    # Check if token has access to /classify endpoint
    if not request_token_data.has_permission("classify"):
        ERROR = {
            "code": "403",
            "type": "forbidden",
            "error": "wrong permissions",
            "message": "wrong permissions - can't access this endpoint",
        }
        TokenAuthServiceInterface.submit_request_history(
            request_token_data.request_uuid, 403, request_metadata, ERROR
        )
        raise HTTPException(status_code=403, detail=ERROR)

    # Check if token has access to the model
    if not request_token_data.has_permission(model_name):
        ERROR = {
            "code": "403",
            "type": "forbidden",
            "error": "wrong permissions",
            "message": "wrong permissions - can't access this model",
        }
        TokenAuthServiceInterface.submit_request_history(
            request_token_data.request_uuid, 403, request_metadata, ERROR
        )
        raise HTTPException(status_code=403, detail=ERROR)

    # Get call metadata
    call_metadata = get_call_metadata()
    call_metadata.update({"endpoint": "/classify/single"})

    # Set seeds
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialise dataset in inference mode
    batch, names = load_batch(audio_files)

    # Load model and decoding from indices to actual labels
    model, index_to_label = load_model(model_name, device)

    # Return error if number of required tags exceeds model capabilities
    if n_tags > len(index_to_label):
        # Create error message
        ERROR = {
            "code": "400",
            "type": "bad request",
            "error": "too many tags required",
            "message": f"the required model allows for a maximum of {len(index_to_label)} tags",
        }

        # Submit error to DB
        TokenAuthServiceInterface.submit_request_history(
            request_token_data.request_uuid, 400, request_metadata, ERROR
        )

        # Return error to API user
        raise HTTPException(status_code=400, detail=ERROR)

    # If there were no problems up to this point, increment the token usage for
    # the given token
    TokenAuthServiceInterface.increment_token_usage(request_token_data.request_uuid)

    # Run inference and fetch results
    ml_results, confidences = infer(
        model=model,
        waveform=batch,
        device=device,
        index_to_label=index_to_label,
        n_tags=n_tags,
    )

    # Create correct JSON-like format for ML results
    ml_results = format_classification_results(ml_results, confidences, names)

    # Save files to S3, unless user has special permission
    if not request_token_data.has_permission("no_save"):
        # Upload files to S3
        uploaded_files_paths = [upload_to_s3(audio_file) for audio_file in audio_files]

        # Update the request metadata with paths of stored files
        request_metadata.update({"s3_path": uploaded_files_paths})

    # Create additional data to be submitted and stored
    additional_metadata = {}
    additional_metadata.update(call_metadata)
    additional_metadata.update({"device": device.type})

    # Submit everything to DB
    TokenAuthServiceInterface.submit_request_history(
        request_uuid=request_token_data.request_uuid,
        response_code=200,
        request_data=request_metadata,
        response_data=ml_results,
        additional_data=additional_metadata,
    )

    # Just return the Machine Learning results
    return ml_results
