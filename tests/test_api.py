import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_features"] > 0


def test_examples_list(client):
    r = client.get("/examples")
    assert r.status_code == 200
    examples = r.json()
    assert len(examples) > 0
    assert "transaction_id" in examples[0]


def test_predict_known_fraud_example(client):
    examples = client.get("/examples").json()
    fraud_example = next(e for e in examples if e["category"] == "high_risk_synthetic")
    detail = client.get(f"/examples/{fraud_example['transaction_id']}").json()

    r = client.post("/predict", json={"transaction": detail["features"]})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "fraud"
    assert body["fraud_probability"] > body["threshold"]
    assert len(body["top_features"]) == 5


def test_predict_partial_payload(client):
    r = client.post("/predict", json={"transaction": {"TransactionAmt": 50.0, "ProductCD": "W"}})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0


def test_predict_empty_payload_rejected(client):
    r = client.post("/predict", json={"transaction": {}})
    assert r.status_code == 422


def test_predict_unknown_example_404(client):
    r = client.get("/examples/999999999")
    assert r.status_code == 404
