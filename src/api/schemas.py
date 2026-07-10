from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TransactionRequest(BaseModel):
    transaction: Dict[str, Any] = Field(
        ...,
        description="Feature name -> value. Unknown keys are ignored; missing known "
                    "features are treated as missing (LightGBM handles NaN natively).",
        examples=[{"TransactionAmt": 125.0, "ProductCD": "W", "card4": "visa"}],
    )


class TopFeature(BaseModel):
    feature: str
    shap_value: float
    feature_value: Optional[Any]


class PredictionResponse(BaseModel):
    fraud_probability: float
    decision: str
    threshold: float
    top_features: List[TopFeature]


class HealthResponse(BaseModel):
    status: str
    model: str
    threshold: float
    n_features: int
    best_iteration: int


class ExampleSummary(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    transaction_id: int
    category: str
    actual_label: str
    model_proba_at_export: float


class ExampleDetail(ExampleSummary):
    features: Dict[str, Any]
