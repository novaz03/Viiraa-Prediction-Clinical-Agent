from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RawFeatureBuildRequest(BaseModel):
    meal_info: Dict[str, Any] = Field(default_factory=dict)
    pre_glucose_series: List[Any] = Field(default_factory=list)


class RawInputPayload(RawFeatureBuildRequest):
    pass


class PredictRequest(BaseModel):
    raw_input: RawInputPayload
    include_expected_columns: bool = False
    user_id: str = "anonymous"


class PredictBatchRequest(BaseModel):
    items: List[RawInputPayload] = Field(default_factory=list)
    include_expected_columns: bool = False


class PredictResponse(BaseModel):
    request_id: str
    predictions: Dict[str, float]
    prediction_intervals: Optional[Dict[str, Dict[str, List[float]]]] = None
    ci_metadata: Optional[Dict[str, Any]] = None
    cohort_comparison: Optional[Dict[str, Any]] = None
    personal_comparison: Optional[Dict[str, Any]] = None
    analysis_text: Optional[Dict[str, str]] = None
    model_family: str
    notes: List[str]
    features_used: Optional[Dict[str, Any]] = None
    expected_columns: Optional[Dict[str, List[str]]] = None


class PredictBatchResponse(BaseModel):
    request_id: str
    model_family: str
    notes: List[str]
    predictions: List[Dict[str, float]]
    expected_columns: Optional[Dict[str, List[str]]] = None


class RawFeatureBuildResponse(BaseModel):
    features: Dict[str, Any]
