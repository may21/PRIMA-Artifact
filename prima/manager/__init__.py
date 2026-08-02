"""Manager entrypoints for PRIMA workflows."""

from prima.manager.docker_workflow import DockerWorkflowConfig, build_workload_specs

__all__ = ["DockerWorkflowConfig", "build_workload_specs"]
