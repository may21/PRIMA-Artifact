"""Paper-facing Predictor wrapper."""

from __future__ import annotations

from dataclasses import dataclass

from prima.features.extract_features import extract_global_features
from prima.predictor.rf import RFClient


@dataclass
class Predictor:
    rf: RFClient

    @classmethod
    def from_files(cls, model_path: str, feature_order_path: str) -> "Predictor":
        return cls(RFClient(local_model_path=model_path, feature_order_path=feature_order_path))

    def predict_memory_mb(
        self,
        onnx_path: str,
        workload: str,
        *,
        batch: int = 1,
        height: int = 640,
        width: int = 640,
        precision: str = "fp32",
    ) -> int:
        features = extract_global_features(
            onnx_path,
            workload,
            batch,
            height,
            width,
            precision,
        )
        return self.rf.predict_max_mem_mb(features)

