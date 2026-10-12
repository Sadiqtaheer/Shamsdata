"""Exercise dashboard views and key filtering behavior against trained reports."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

import pandas as pd
from streamlit.testing.v1 import AppTest

from auto_training import AutoTrainer

root = Path(__file__).resolve().parent
at = AppTest.from_file(str(root / "app.py"), default_timeout=60).run()
assert not at.exception, [e.message for e in at.exception]
assert [tab.label for tab in at.tabs] == ["Operations", "Churn risk", "Business projection", "More analytics"]
assert at.metric[0].value == "56,036"
at.multiselect(key="services").set_value([]).run()
assert not at.exception
assert any("No transactions" in i.value for i in at.info)
at.multiselect(key="services").set_value(["data"]).run()
assert not at.exception
at.multiselect(key="services").set_value(sorted(pd.read_csv(root / "reports/transaction_success_predictions.csv").service.unique())).run()
at.number_input(key="churn_floor_percent").set_value(75).run()
assert not at.exception
assert at.tabs[1].dataframe[0].value["Churn risk (%)"].ge(75).all()
at.number_input(key="churn_floor_percent").set_value(25).run()
assert at.number_input(key="churn_floor_percent").value == 25
assert not at.exception
at.number_input(key="churn_floor_percent").set_value(0).run()
at.number_input(key="churn_display_count").set_value(20).run()
assert len(at.tabs[1].dataframe[0].value) == 20
at.number_input(key="churn_display_count").set_value(10).run()
assert len(at.tabs[1].dataframe[0].value) == 10
at.button(key="churn_page_next").click().run()
assert any("Page 2" in caption.value for caption in at.tabs[1].caption)
at.text_input(key="churn_search").set_value("[nonexistent customer]").run()
assert at.tabs[1].dataframe[0].value.empty
at.selectbox(key="projection_service").set_value("data").run()
assert not at.exception

for section in ["Forecasts", "Data plan performance", "Customers", "Product trends", "Anomalies", "Model performance"]:
    at.selectbox(key="analytics_section").set_value(section).run()
    assert not at.exception, (section, [e.message for e in at.exception])
    print(section + ": PASS")
    if section == "Forecasts":
        at.number_input(key="forecast_horizon_count").set_value(90).run()
        assert not at.exception and len(at.tabs[3].dataframe[0].value) == 90
        at.number_input(key="forecast_horizon_count").set_value(7).run()
        assert not at.exception and len(at.tabs[3].dataframe[0].value) == 7
        at.selectbox(key="forecast_measure").set_value("Data transactions").run()
        assert not at.exception
        at.selectbox(key="forecast_measure").set_value("Active users").run()
        assert not at.exception
    elif section == "Data plan performance":
        assert at.number_input(key="data_plan_minimum_volume").value == 30
        assert len(at.tabs[3].dataframe[0].value) > 0
        at.number_input(key="data_plan_minimum_volume").set_value(100).run()
        assert not at.exception
    elif section == "Customers":
        at.checkbox(key="only_high").set_value(False).run()
        assert not at.exception
        assert len(at.tabs[3].dataframe[0].value) == len(pd.read_csv(root / "reports/customer_churn_and_value.csv"))
    elif section == "Product trends":
        at.selectbox(key="product_metric").set_value("Transacting users").run()
        assert not at.exception
    elif section == "Anomalies":
        at.multiselect(key="anomaly_services").set_value([]).run()
        assert not at.exception and at.tabs[3].metric[0].value == "0"
        assert not at.tabs[3].dataframe

f = pd.read_csv(root / "reports/company_daily_forecast.csv")
assert len(f) == 90
assert (f.successful_transactions + f.failed_transactions <= f.transactions + 1e-8).all()
assert (f.data_transactions <= f.transactions + 1e-8).all()
assert (f.revenue <= f.transaction_value + 1e-8).all()
assert f.select_dtypes("number").ge(0).all().all()
models = list((root / "saved_models").glob("*.joblib"))
assert len(models) == 12
assert "data_transactions" in f
plan_performance = pd.read_csv(root / "reports/data_plan_performance.csv")
assert not plan_performance.empty
assert plan_performance.success_rate.between(0, 1).all()
manifest = json.loads((root / "reports/run_manifest.json").read_text())
assert manifest["transaction_rows"] == 56036
print("Operations filters, churn pagination, projections, forecast horizons and consistency: PASS")

with tempfile.TemporaryDirectory() as temporary_directory:
    project_dir = Path(temporary_directory)
    data_dir = project_dir / "data"
    reports_dir = project_dir / "reports"
    data_dir.mkdir()
    reports_dir.mkdir()
    source = data_dir / "transactions.csv"
    source.write_text("id,value\n1,10\n", encoding="utf-8")
    manifest = reports_dir / "run_manifest.json"
    manifest.write_text(json.dumps({
        "source_sha256": {
            source.name: hashlib.sha256(source.read_bytes()).hexdigest()
        }
    }), encoding="utf-8")
    fake_trainer = project_dir / "fake_train.py"
    fake_trainer.write_text(
        "import hashlib, json, sys\n"
        "from pathlib import Path\n"
        "source = Path(sys.argv[1])\n"
        "reports = Path(sys.argv[2])\n"
        "digest = hashlib.sha256(source.read_bytes()).hexdigest()\n"
        "(reports / 'run_manifest.json').write_text(json.dumps({"
        "'source_sha256': {source.name: digest}}), encoding='utf-8')\n",
        encoding="utf-8",
    )
    trainer = AutoTrainer(
        project_dir,
        data_dir,
        reports_dir,
        {"transactions": "transactions.csv"},
        poll_seconds=.02,
        settle_seconds=.05,
        retry_seconds=.2,
        command=[sys.executable, str(fake_trainer), str(source), str(reports_dir)],
    ).start()
    try:
        source.write_text("id,value\n1,10\n2,20\n", encoding="utf-8")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and trainer.snapshot()["generation"] == 0:
            time.sleep(.02)
        state = trainer.snapshot()
        assert state["generation"] == 1 and state["state"] == "idle", state
        updated_manifest = json.loads(manifest.read_text(encoding="utf-8"))
        assert updated_manifest["source_sha256"][source.name] == hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
    finally:
        trainer.stop()
print("Automatic retraining after source changes: PASS")
