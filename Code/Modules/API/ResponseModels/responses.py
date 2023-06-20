from pydantic import BaseModel, Extra, Field
from typing import List, Dict


class Tag(BaseModel):
    tag: str = Field(example="Siren")
    confidence: str = Field(example="82%")


class Events(BaseModel):
    tag: str = Field(example="Speech")
    confidence: str = Field(example="91%")
    start: float = Field(example="0.853")
    end: float = Field(example="0.86")


class SampleClassification(BaseModel):
    tags: List[Tag]


class ClassifyResponse(BaseModel):
    classification: Dict[str, SampleClassification]

    class Config:
        extra = Extra.allow


class WindowClassification(BaseModel):
    start: float
    end: float
    tags: List[Tag]


class WindowResponse(BaseModel):
    name: str = Field(example="sample.mp3")
    windows: List[WindowClassification]

    class Config:
        extra = Extra.allow


class DetectResponse(BaseModel):
    events: List[Events]

    class Config:
        extra = Extra.allow


classify_error_responses = {
    400: {
        "description": "Number of tags greater than allowed",
        "content": {
            "application/json": {
                "example": {
                    "code": "400",
                    "type": "bad request",
                    "error": "too many tags required",
                    "message": "the required model allows for a maximum of 526 tags",
                }
            }
        },
    },
    413: {
        "description": "More than 16 files uploaded",
        "content": {
            "application/json": {
                "example": {
                    "code": "413",
                    "type": "payload too large",
                    "error": "too many files uploaded at once",
                    "message": "resubmit request with no more than 16 files",
                }
            }
        },
    },
    500: {
        "description": "Unexpected server error",
        "content": {
            "application/json": {
                "example": {
                    "code": "500",
                    "type": "server error",
                    "error": "Internal Server Error",
                    "message": "Internal Server Error",
                }
            }
        },
    },
}

window_error_responses = {
    400: {
        "description": "Too many tags required",
        "content": {
            "application/json": {
                "example": {
                    "code": "400",
                    "type": "bad request",
                    "error": "too many tags required",
                    "message": "the required model allows for a maximum of 526 tags",
                }
            }
        },
    },
    413: {
        "description": "Window too large",
        "content": {
            "application/json": {
                "example": {
                    "code": "413",
                    "type": "bad request",
                    "error": "window too large",
                    "message": "the required window is larger than max duration of file",
                }
            }
        },
    },
    500: {
        "description": "Unexpected server error",
        "content": {
            "application/json": {
                "example": {
                    "code": "500",
                    "type": "server error",
                    "error": "Internal Server Error",
                    "message": "Internal Server Error",
                }
            }
        },
    },
}

detect_error_responses = {
    400: {
        "description": "Invalid threshold value",
        "content": {
            "application/json": {
                "example": {
                    "code": "400",
                    "type": "bad request",
                    "error": "invalid threshold value",
                    "message": "the threshold parameter must be between 0 and 1",
                }
            }
        },
    },
    500: {
        "description": "Unexpected server error",
        "content": {
            "application/json": {
                "example": {
                    "code": "500",
                    "type": "server error",
                    "error": "Internal Server Error",
                    "message": "Internal Server Error",
                }
            }
        },
    },
}
