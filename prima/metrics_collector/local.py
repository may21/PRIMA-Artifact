"""Local Metrics Collector for Jetson/Orin Docker experiments."""

from __future__ import annotations

from prima.runtime.resource_reader import get_available_mb


def read_available_memory_mb(tegra_log: str) -> int:
    return get_available_mb(tegra_log)

