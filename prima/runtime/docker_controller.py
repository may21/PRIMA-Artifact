# app/docker_controller.py
import os
import json
import time
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional

import docker


@dataclass
class WorkloadRunSpec:
    role: str
    image: str
    command: List[str]
    mem_mib: int
    host_path: str
    ctr_path: str


class DockerController:
    """
    Reconcile per-role Docker workloads to the desired state on one SDC node.
    - apply(desired): create, remove, or replace role containers.
    - changes use per-role blue/green replacement.
    """

    def __init__(self, network_name: str = "pmu-net", active_dir: str = "/tmp/pmu_active"):
        self.client = docker.from_env()
        self.network_name = network_name
        self.active_dir = active_dir
        os.makedirs(self.active_dir, exist_ok=True)
        self._ensure_network()

    def _ensure_network(self) -> None:
        nets = {n.name: n for n in self.client.networks.list()}
        if self.network_name not in nets:
            self.client.networks.create(self.network_name, driver="bridge")

    def _spec_hash(self, s: WorkloadRunSpec) -> str:
        payload = json.dumps(
            {"image": s.image, "command": s.command, "mem_mib": s.mem_mib, "host_path": s.host_path, "ctr_path": s.ctr_path},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]

    def _name(self, role: str, color: str, h: str) -> str:
        return f"pmu-{role.lower()}-{color}-{h}"

    def _active_file(self, role: str) -> str:
        return os.path.join(self.active_dir, f"{role.lower()}.txt")

    def _read_active_name(self, role: str) -> Optional[str]:
        p = self._active_file(role)
        if not os.path.exists(p):
            return None
        return open(p, "r", encoding="utf-8").read().strip() or None

    def _write_active_name(self, role: str, name: str) -> None:
        open(self._active_file(role), "w", encoding="utf-8").write(name)

    def _list_role_containers(self, role: str, all_: bool = True):
        return self.client.containers.list(
            all=all_,
            filters={"label": [f"pmu.managed=true", f"pmu.role={role.lower()}"]},
        )

    def _get_container(self, name: str):
        try:
            return self.client.containers.get(name)
        except docker.errors.NotFound:
            return None

    def _is_running_stable(self, name: str, stable_sec: float = 2.0) -> bool:
        """
        Readiness substitute: the container must stay running for stable_sec.
        """
        c = self._get_container(name)
        if not c:
            return False
        c.reload()
        if c.status != "running":
            return False
        t0 = time.time()
        while time.time() - t0 < stable_sec:
            time.sleep(0.2)
            c = self._get_container(name)
            if not c:
                return False
            c.reload()
            if c.status != "running":
                return False
        return True

    def _start_container(self, s: WorkloadRunSpec, color: str) -> str:
        h = self._spec_hash(s)
        name = self._name(s.role, color, h)

        # Remove any stale container with the same generated name.
        old = self._get_container(name)
        if old:
            old.remove(force=True)

        volumes = {
            os.path.abspath(s.host_path): {"bind": s.ctr_path, "mode": "ro"}
        }

        self.client.containers.run(
            image=s.image,
            name=name,
            command=s.command,
            detach=True,
            network=self.network_name,
            volumes=volumes,
            working_dir=s.ctr_path,
            mem_limit=f"{int(s.mem_mib)}m",  # MiB
            labels={
                "pmu.managed": "true",
                "pmu.role": s.role.lower(),
                "pmu.hash": h,
                "pmu.color": color,
            },
            restart_policy={"Name": "unless-stopped"},
        )
        return name

    def apply(self, desired: Dict[str, WorkloadRunSpec]) -> None:
        """
        desired: {role: WorkloadRunSpec}
        - remove managed role containers that are no longer desired
        - deploy and switch to a new container when the active hash differs
        """
        desired_roles = set(r.lower() for r in desired.keys())

        # 1) Remove roles that are not in the desired state.
        managed = self.client.containers.list(all=True, filters={"label": ["pmu.managed=true"]})
        for c in managed:
            labels = c.labels or {}
            role = labels.get("pmu.role")
            if role and role not in desired_roles:
                try:
                    c.remove(force=True)
                    print(f"[remove] role={role} name={c.name}")
                except Exception as e:
                    print(f"[remove-fail] role={role} name={c.name} err={e}")

        # 2) Reconcile each desired role.
        for role, spec in desired.items():
            role_l = role.lower()
            desired_hash = self._spec_hash(spec)

            active_name = self._read_active_name(role_l)
            active = self._get_container(active_name) if active_name else None
            active_hash = (active.labels or {}).get("pmu.hash") if active else None
            active_ok = active is not None and self._is_running_stable(active.name, 0.2)

            if active_ok and active_hash == desired_hash:
                print(f"[ok] role={role_l} active={active.name}")
                continue

            # Start the new green container.
            print(f"[deploy] role={role_l} -> green")
            new_name = self._start_container(spec, color="green")

            # readiness
            if not self._is_running_stable(new_name, stable_sec=2.0):
                # Preserve the previous active container when the new one fails.
                c = self._get_container(new_name)
                if c:
                    logs = ""
                    try:
                        logs = c.logs(tail=50).decode("utf-8", errors="ignore")
                    except Exception:
                        pass
                    c.remove(force=True)
                raise RuntimeError(f"[{role_l}] new container failed to stay running.\nlogs:\n{logs}")

            # Switch active pointer.
            self._write_active_name(role_l, new_name)
            print(f"[switch] role={role_l} active={new_name}")

            # Remove previous role containers except the new active one.
            for c in self._list_role_containers(role_l, all_=True):
                if c.name != new_name:
                    try:
                        c.remove(force=True)
                        print(f"[cleanup] role={role_l} removed={c.name}")
                    except Exception as e:
                        print(f"[cleanup-fail] role={role_l} name={c.name} err={e}")

        print("apply complete")
