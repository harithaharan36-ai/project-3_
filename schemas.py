"""Pydantic request / response schemas for the CIFAR-10 API."""

from typing import List
from pydantic import BaseModel, Field, HttpUrl


class Prediction(BaseModel):
    class_: str = Field(..., alias="class", description="Predicted class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Softmax probability")

    model_config = {"populate_by_name": True}


class PredictResponse(BaseModel):
    filename: str | None = None
    predictions: List[Prediction]


class URLRequest(BaseModel):
    url: HttpUrl
    topk: int = Field(3, ge=1, le=10)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    num_classes: int
