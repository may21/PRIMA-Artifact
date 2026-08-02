"""Docker Workload Launcher adapter."""

from __future__ import annotations

from prima.runtime.docker_controller import DockerController, WorkloadRunSpec


WorkloadSpec = WorkloadRunSpec


class DockerWorkloadLauncher:
    def __init__(self) -> None:
        self.controller = DockerController()

    def apply(self, specs: dict[str, WorkloadSpec]) -> None:
        self.controller.apply(specs)

