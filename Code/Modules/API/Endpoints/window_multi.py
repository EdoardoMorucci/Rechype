import torch

from fastapi import APIRouter
from fastapi import Depends, Form, UploadFile, HTTPException

from Code.Modules.API.Window.A_load_window import load_window
from Code.Modules.API.Window.B_format_results import format_window_results
from Code.Modules.API.Common.load_model import load_model
from Code.Modules.API.Classification.B_run_inference_multi import infer

from Code.Modules.API.Common.api_call_metadata import get_call_metadata

from Authentication.Checkers import (
    window_auth_check,
)
from Authentication.Token_auth_service_interface import TokenAuthServiceInterface
from Authentication.Types import RequestTokenData

from Code.Modules.API.ResponseModels.responses import (
    window_error_responses,
    WindowResponse,
)

from Code.Modules.API.AWS.upload import upload_to_s3


# Initialise router
router = APIRouter()


@router.post(
    "/window/multi", response_model=WindowResponse, responses=window_error_responses
)
async def window(
    audio_file: UploadFile,
    request_token_data: RequestTokenData = Depends(window_auth_check),
    model_name: str = Form(...),
    window_length: float = Form(...),
):
    """
    Endpoint used to classify the audio content of one audio file by subdividing it into non-overlapping windows of length
    specified by the user. That is, if the user uploads a file of 10s and specifies a window of 3s, then the audio
    will be split into four intervals, namely (0, 3), (3, 6), (6, 9), (9, 10). These windows will then be stacked
    one onto the other (possibly addding silence = zero padding at the end of the last one) to create a batch.

    @model_name (string): name of the model to be used for classification

    @window_length (float): float describing the length in seconds of the sliding window

    @n_tags (integer): int specifying the number of tags to be returned for every window

    """

    # Create request metadata (containing the parameters passed to it)
    request_metadata = {
        "num_audio_files": 1,
        "model_name": model_name,
        "window_length": window_length,
    }

    # Check if user has access to this endpoint
    if not request_token_data.has_permission("window"):
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
    call_metadata.update({"endpoint": "/window/multi"})

    # Set seeds
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialise dataset in inference mode
    batch, name, splits = load_window(audio_file, window_length)

    # Load model and decoding from indices to actual labels
    model, index_to_label = load_model(model_name, device)

    # Run inference and fetch results
    ml_results, confidences = infer(
        model=model,
        waveform=batch,
        device=device,
        index_to_label=index_to_label,
    )

    # Create correct JSON-like format for ML results
    ml_results = format_window_results(ml_results, confidences, name, splits)

    # Save files to S3, unless user has special permission
    if not request_token_data.has_permission("no_save"):
        # Upload files to S3
        uploaded_file_path = upload_to_s3(audio_file)

        # Update the request metadata with paths of stored files
        request_metadata.update({"s3_path": uploaded_file_path})

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

    return ml_results
