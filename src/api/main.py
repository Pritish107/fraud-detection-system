"""FastAPI service for the fraud detection model.

Endpoints:
    GET  /health              liveness + model metadata
    GET  /examples             list of precomputed example transactions
    GET  /examples/{id}        full feature payload for one example (feed straight into /predict)
    POST /predict               fraud probability, decision, and top contributing features

Run:
    .venv/Scripts/python.exe -m uvicorn src.api.main:app --reload --port 8000
"""

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

import pandas as pd
from fastapi import FastAPI, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.schemas import (  # noqa: E402
    ExampleDetail,
    ExampleSummary,
    HealthResponse,
    PredictionResponse,
    TransactionRequest,
)
from explainability.explainer import FraudExplainer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_PATH = ROOT / "data" / "processed" / "example_transactions.json"

_explainer: FraudExplainer = None
_examples: list = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _explainer, _examples
    _explainer = FraudExplainer()
    if EXAMPLES_PATH.exists():
        _examples = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))
    yield


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud scoring with SHAP-based explanations.",
    version="1.0.0",
    lifespan=lifespan,
)


def build_dataframe(payload: Dict, fe: FraudExplainer) -> pd.DataFrame:
    """Dtype coercion (numeric parsing, categorical NaN handling) happens centrally in
    FraudExplainer._prepare — this just aligns the payload to the expected columns."""
    row = {col: payload.get(col, None) for col in fe.feature_cols}
    return pd.DataFrame([row])


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model="lightgbm_main",
        threshold=_explainer.threshold,
        n_features=len(_explainer.feature_cols),
        best_iteration=_explainer.meta["best_iteration"],
    )


@app.get("/examples", response_model=list[ExampleSummary])
def list_examples() -> list:
    return [
        ExampleSummary(
            transaction_id=e["transaction_id"],
            category=e["category"],
            actual_label=e["actual_label"],
            model_proba_at_export=e["model_proba_at_export"],
        )
        for e in _examples
    ]


@app.get("/examples/{transaction_id}", response_model=ExampleDetail)
def get_example(transaction_id: int) -> ExampleDetail:
    for e in _examples:
        if e["transaction_id"] == transaction_id:
            return ExampleDetail(**e)
    raise HTTPException(status_code=404, detail=f"No example with transaction_id={transaction_id}")


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest) -> PredictionResponse:
    if not isinstance(request.transaction, dict) or len(request.transaction) == 0:
        raise HTTPException(status_code=422, detail="transaction must be a non-empty object")

    df = build_dataframe(request.transaction, _explainer)
    try:
        result = _explainer.explain_row(df, top_n=5)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to score transaction: {exc}") from exc

    return PredictionResponse(**result)
