"""Unit tests for data generation, feature engineering, and API feature builders."""

from mlops_platform.data.feature_pipeline import (
    engineer_churn_features,
    engineer_fraud_features,
    prepare_churn,
    prepare_fraud,
)
from mlops_platform.data.generate_data import make_churn_data, make_fraud_data, make_drift_data
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class TestDataGeneration:
    def test_churn_data_shape(self):
        df = make_churn_data(n=100)
        assert len(df) == 100
        assert "churn" in df.columns

    def test_fraud_data_shape(self):
        df = make_fraud_data(n=200)
        assert len(df) == 200
        assert "fraud" in df.columns

    def test_churn_label_is_binary(self):
        df = make_churn_data(n=500)
        assert set(df["churn"].unique()).issubset({0, 1})

    def test_fraud_label_is_binary(self):
        df = make_fraud_data(n=500)
        assert set(df["fraud"].unique()).issubset({0, 1})

    def test_drift_data_differs_from_original(self):
        df = make_churn_data(n=500)
        drifted = make_drift_data(df, drift_factor=0.5)
        # Drifted data should have different mean for at least one numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "churn"]
        diffs = [abs(df[c].mean() - drifted[c].mean()) for c in numeric_cols]
        assert max(diffs) > 0.01, "Drifted data should differ from original"

    def test_no_nan_in_generated_data(self):
        df = make_churn_data(n=100)
        assert not df.isnull().any().any()


class TestFeatureEngineering:
    def test_churn_features_added(self):
        df = make_churn_data(n=100)
        engineered = engineer_churn_features(df)
        assert "charges_per_month" in engineered.columns
        assert "engagement_score" in engineered.columns
        assert "at_risk" in engineered.columns

    def test_fraud_features_added(self):
        df = make_fraud_data(n=100)
        engineered = engineer_fraud_features(df)
        assert "log_amount" in engineered.columns
        assert "is_night" in engineered.columns
        assert "suspicious_online" in engineered.columns

    def test_prepare_churn_returns_correct_shape(self):
        df = make_churn_data(n=200)
        X, y, cols, scaler = prepare_churn(df)
        assert X.shape[0] == 200
        assert len(y) == 200
        assert X.shape[1] == len(cols)

    def test_prepare_fraud_returns_correct_shape(self):
        df = make_fraud_data(n=200)
        X, y, cols, scaler = prepare_fraud(df)
        assert X.shape[0] == 200
        assert len(y) == 200
        assert X.shape[1] == len(cols)

    def test_scaler_transform_works(self):
        df_train = make_churn_data(n=200)
        df_test = make_churn_data(n=50)
        _, _, _, scaler = prepare_churn(df_train, fit_scaler=True)
        X_test, _, _, _ = prepare_churn(df_test, scaler=scaler, fit_scaler=False)
        assert X_test.shape[0] == 50

    def test_at_risk_flag_logic(self):
        df = make_churn_data(n=1000)
        engineered = engineer_churn_features(df)
        # at_risk should only be 1 when contract_type==0 AND support_calls > 3
        at_risk_rows = engineered[engineered["at_risk"] == 1]
        assert (at_risk_rows["contract_type"] == 0).all()
        assert (at_risk_rows["support_calls"] > 3).all()


class TestAPIFeatureBuilders:
    """Test the feature builders used in the FastAPI serving layer."""

    def test_churn_feature_builder(self):
        from mlops_platform.serving.api import ChurnRequest, build_churn_features
        req = ChurnRequest(
            tenure=24, monthly_charges=60.0, total_charges=1440.0,
            num_products=2, support_calls=1, payment_delays=0,
            contract_type=1, internet_service=1, online_security=1,
            tech_support=0, paperless_billing=1, senior_citizen=0,
        )
        X = build_churn_features(req)
        assert X.shape == (1, 19)

    def test_fraud_feature_builder(self):
        from mlops_platform.serving.api import FraudRequest, build_fraud_features
        req = FraudRequest(
            amount=150.0, hour=14, day_of_week=2, merchant_category=5,
            distance_from_home=15.0, distance_from_last_txn=5.0,
            ratio_to_median=1.0, used_chip=1, used_pin=1,
            online_order=0, num_txn_last_hour=1,
        )
        X = build_fraud_features(req)
        assert X.shape == (1, 19)
