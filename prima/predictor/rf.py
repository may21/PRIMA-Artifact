"""Predictor model loader.

The current `rf_model.pkl` artifact is a scikit-learn Pipeline whose final
estimator is `RandomForestRegressor`. `RFClient` is kept as a compatibility
alias for the existing Orin1/master scripts and paper notes.
"""

import os, joblib, json
import pandas as pd


class PredictorModelClient:
    def __init__(self, local_model_path, feature_order_path=None):
        self.model = joblib.load(local_model_path)
        self.feature_order = None
        if feature_order_path and os.path.exists(feature_order_path):
            with open(feature_order_path) as f:
                data = json.load(f)
                # Accept either the current dict format or the older list format.
                if isinstance(data, dict) and "feature_order" in data:
                    self.feature_order = data["feature_order"]
                elif isinstance(data, list):
                    self.feature_order = data
                else:
                    raise ValueError("feature_order.json format invalid")

    def predict_max_mem_mb(self, features: dict) -> int:
        if not self.feature_order:
            raise RuntimeError("feature_order not loaded")

        # Align features to feature_order.
        X = [features.get(k, 0) for k in self.feature_order]

        # Preserve column names expected by the sklearn pipeline.
        df = pd.DataFrame([X], columns=self.feature_order)

        pred = self.model.predict(df)[0]
        return int(pred)

    def describe_model(self) -> dict[str, str]:
        estimator = self.model
        if hasattr(self.model, "named_steps") and "clf" in self.model.named_steps:
            estimator = self.model.named_steps["clf"]
        return {
            "container_type": f"{type(self.model).__module__}.{type(self.model).__name__}",
            "estimator_type": f"{type(estimator).__module__}.{type(estimator).__name__}",
        }


RFClient = PredictorModelClient
