# app/tegrastats_reader.py
import os
import re
import time
import subprocess
from dataclasses import dataclass
from typing import Optional

_RAM_RE = re.compile(r"RAM\s+(\d+)\s*/\s*(\d+)MB", re.IGNORECASE)

@dataclass
class TegrastatsSample:
    used_mb: int
    total_mb: int

    @property
    def avail_mb(self) -> int:
        return max(0, self.total_mb - self.used_mb)


def parse_tegrastats_line(line: str) -> Optional[TegrastatsSample]:
    """
    Parse RAM used/total in MB from one tegrastats line.
    Example: "RAM 3182/7770MB ..."
    """
    m = _RAM_RE.search(line)
    if not m:
        return None
    used = int(m.group(1))
    total = int(m.group(2))
    return TegrastatsSample(used_mb=used, total_mb=total)


def ensure_tegrastats_logging(log_path: str, interval_ms: int = 1000) -> None:
    """
    Ensure a background tegrastats process is writing to log_path.
    - Reuse an existing matching process when present.
    - Start a new process otherwise.
    """
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    # Conservatively reuse any tegrastats process already writing this logfile.
    try:
        out = subprocess.check_output(["bash", "-lc", "pgrep -af tegrastats || true"], text=True)
        for ln in out.splitlines():
            if log_path in ln:
                return
    except Exception:
        pass

    # Start logging. JetPack commonly supports --logfile; interval is in ms.
    subprocess.Popen(
        ["tegrastats", "--interval", str(interval_ms), "--logfile", log_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def read_latest_sample_from_file(log_path: str, wait_sec: float = 2.0) -> TegrastatsSample:
    """
    Return the latest valid RAM used/total sample from the logfile.
    Retry briefly when the logfile has not been written yet.
    """
    t0 = time.time()
    last_good = None

    while time.time() - t0 < wait_sec:
        if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
            try:
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    # Search backward for the most recent valid line.
                    lines = f.readlines()[-50:]
                for line in reversed(lines):
                    s = parse_tegrastats_line(line)
                    if s:
                        last_good = s
                        break
                if last_good:
                    return last_good
            except Exception:
                pass
        time.sleep(0.2)

    raise RuntimeError(f"failed to read tegrastats sample from {log_path}")


def get_available_mb(log_path: str) -> int:
    s = read_latest_sample_from_file(log_path)
    return s.avail_mb
