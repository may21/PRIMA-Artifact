"""Docker-oriented PRIMA workflow for the paper reproduction path.

This module mirrors the Figure 5 Manager role at a lightweight level:
it receives requested workload roles, asks the Predictor for PMU, asks the
Calculator for memory budgets, and builds Docker launch specs for the
Workload Launcher.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from prima.calculator import calculate_budgets
from prima.metrics_collector import read_available_memory_mb
from prima.predictor import Predictor
from prima.workload_launcher import WorkloadSpec


ROLE_TO_ONNX = {
    "CLS": "yolov8n-cls.onnx",
    "DET": "yolov8n.onnx",
    "EST": "yolov8n-pose.onnx",
    "SEG": "yolov8n-seg.onnx",
    "OBB": "yolov8n-obb.onnx",
}

ROLE_TO_SCRIPT = {
    "CLS": "classify.py",
    "DET": "detect.py",
    "EST": "pose.py",
    "SEG": "segment.py",
    "OBB": "obb.py",
}


@dataclass(frozen=True)
class DockerWorkflowConfig:
    image: str
    host_path: str
    ctr_path: str = "/usr/src/ultralytics/yolo_new"
    cache_dir: str = "cache"
    batch: int = 1
    height: int = 640
    width: int = 640
    precision: str = "fp32"
    margin_mb: int = 100
    margin_overrides_mb: dict[str, int] = field(default_factory=lambda: {"SEG": -300})
    tegra_log: str = "/tmp/tegrastats.log"


def build_workload_specs(
    roles: Iterable[str],
    config: DockerWorkflowConfig,
    predictor: Predictor,
) -> dict[str, WorkloadSpec]:
    """Build Docker workload specs with PRIMA memory budgets."""

    normalized_roles = [role.upper() for role in roles]
    predicted_with_margin: dict[str, int] = {}

    for role in normalized_roles:
        onnx_name = ROLE_TO_ONNX.get(role)
        if not onnx_name:
            raise ValueError(f"unknown role: {role}")
        onnx_path = Path(config.cache_dir) / onnx_name
        predicted_mb = predictor.predict_memory_mb(
            str(onnx_path),
            role,
            batch=config.batch,
            height=config.height,
            width=config.width,
            precision=config.precision,
        )
        margin_mb = config.margin_overrides_mb.get(role, config.margin_mb)
        predicted_with_margin[role] = predicted_mb + margin_mb

    available_mb = read_available_memory_mb(config.tegra_log)
    budgets_mib = calculate_budgets(predicted_with_margin, available_mb)

    specs: dict[str, WorkloadSpec] = {}
    for role in normalized_roles:
        script = ROLE_TO_SCRIPT[role]
        specs[role] = WorkloadSpec(
            role=role,
            image=config.image,
            command=["python3", os.path.join(config.ctr_path, script)],
            mem_mib=budgets_mib[role],
            host_path=config.host_path,
            ctr_path=config.ctr_path,
        )
    return specs
