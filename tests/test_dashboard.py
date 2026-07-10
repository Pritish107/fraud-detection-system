from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "dashboard" / "app.py")


def test_transaction_explorer_renders_without_exceptions():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not list(at.exception)
    assert len(at.main.metric) == 3


def test_drift_monitoring_renders_without_exceptions():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("Drift Monitoring").run()
    assert not list(at.exception)
    assert len(at.main.metric) == 3
