from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "dashboard" / "app.py")
REAL_EXAMPLES_PATH = (Path(__file__).resolve().parents[1] / "data" / "processed" /
                       "example_transactions_real_LOCAL_ONLY.json")


def test_overview_renders_without_exceptions():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not list(at.exception)
    assert len(at.main.markdown) > 0


def test_transaction_explorer_renders_without_exceptions():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("Transaction Explorer").run()
    assert not list(at.exception)
    assert len(at.main.get("plotly_chart")) == 2  # risk gauge + SHAP contribution chart


@pytest.mark.skipif(not REAL_EXAMPLES_PATH.exists(),
                     reason="real example data not generated locally (needs Kaggle access)")
def test_transaction_explorer_real_data_toggle():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("Transaction Explorer").run()
    at.main.radio[0].set_value("Real IEEE-CIS transactions (local only)").run()
    assert not list(at.exception)
    assert len(at.main.selectbox[0].options) > 0
    assert "actual:" in at.main.selectbox[0].options[0]


def test_drift_monitoring_renders_without_exceptions():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("Drift Monitoring").run()
    assert not list(at.exception)
    assert len(at.main.tabs) == 3
    assert len(at.main.get("plotly_chart")) == 1  # feature-drift PSI chart
