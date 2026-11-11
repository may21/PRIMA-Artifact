"""Tolerant parsers for the logs written by the PRIMA runners.

Runners now force LC_ALL=C, but results collected earlier on machines with a
non-English locale (e.g. ko_KR) and workload scripts that print a different
per-image line are still parsed here, so old and new results stay comparable.
"""

from __future__ import annotations

import re

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Ultralytics verbose line printed by the original workload scripts:
#   image 1/100 /path/000000000139.jpg: 480x640 1 person, ..., 12.3ms
ULTRALYTICS_IMAGE_RE = re.compile(
    r"^image\s+(?P<index>\d+)/(?P<count>\d+)\s+.*?(?P<ms>\d+(?:\.\d+)?)ms\s*$"
)
# Explicit per-stage line printed by re-created workload scripts:
#   [1/100] 000000000139.jpg preprocess=3.1ms inference=12.3ms postprocess=2.0ms
STAGED_IMAGE_RE = re.compile(
    r"^\[(?P<index>\d+)/(?P<count>\d+)\]\s+.*?inference=(?P<ms>\d+(?:\.\d+)?)ms"
)


def parse_inference_line(line: str):
    """Return (image_index, image_count, inference_ms) or None.

    Only the model inference time is returned for both formats, which is the
    quantity reported in the paper (Ultralytics speed['inference'])."""
    clean = ANSI_RE.sub("", line).strip()
    for pattern in (ULTRALYTICS_IMAGE_RE, STAGED_IMAGE_RE):
        match = pattern.match(clean)
        if match:
            return int(match.group("index")), int(match.group("count")), float(match.group("ms"))
    return None


_KO_TIME_RE = re.compile(r"^(?P<ampm>오전|오후)\s+(?P<h>\d{1,2})시\s+(?P<m>\d{1,2})분\s+(?P<s>\d{1,2})초")
_KO_CLOCK_RE = re.compile(r"^(?P<ampm>오전|오후)\s+(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2})")
_EN_AMPM_RE = re.compile(r"^(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2})\s+(?P<ampm>AM|PM)\b")


def _to_24h(hour: int, pm: bool) -> int:
    hour %= 12
    return hour + 12 if pm else hour


def normalize_time_prefix(line: str) -> str:
    """Rewrite a leading localized sysstat timestamp as 24-hour HH:MM:SS.

    Handles '오후 03시 12분 01초', '오후 03:12:01' and '03:12:01 PM'; other
    lines are returned unchanged."""
    stripped = line.lstrip()
    for pattern in (_KO_TIME_RE, _KO_CLOCK_RE, _EN_AMPM_RE):
        match = pattern.match(stripped)
        if match:
            pm = match.group("ampm") in ("오후", "PM")
            hhmmss = f"{_to_24h(int(match.group('h')), pm):02d}:{int(match.group('m')):02d}:{int(match.group('s')):02d}"
            return hhmmss + stripped[match.end():]
    return line
