# app/extract_features.py
import os, re
import onnx
import numpy as np
from onnx import numpy_helper, shape_inference
from collections import Counter

# Enable YOLO-head-aware output correction for dynamic estimation.
USE_YOLO_AWARE = True

# Use fixed output-byte estimates by default to match the training CSV.
FORCE_FIXED_OUTPUT = True
FIXED_OUTPUT_BYTES_BY_WL_DEFAULT = {
    "cls":  32_000,
    "det":  5_644_800,
    "obb":  1_344_000,
    "pose": 3_763_200,  # Treat EST as pose.
    "seg":  14_348_800,
}

# Normalize workload aliases to canonical lowercase names.
WORKLOAD_ALIASES = {
    "cls": "cls",
    "det": "det",
    "obb": "obb",
    "seg": "seg",
    "pose": "pose",
    "est": "pose",
}

def _precision_to_elem_bytes(precision: str | None) -> int | None:
    if not precision:
        return None
    p = str(precision).lower().strip()
    if p in ("fp16", "float16", "16"):
        return 2
    if p in ("fp32", "float32", "32"):
        return 4
    if p in ("int8", "i8", "8"):
        return 1
    return None

ONNX_DTYPE_BYTES = {1:4,2:1,3:1,4:2,5:2,6:4,7:8,9:2,10:8,16:2}

def _prod(xs):
    n = 1
    for v in xs:
        n *= int(v)
    return int(n)

def _elem_bytes(vi):
    return ONNX_DTYPE_BYTES.get(vi.type.tensor_type.elem_type, 4)

def _dims_from_vi(vi):
    tt = vi.type.tensor_type
    if not tt.HasField("shape"):
        return []
    dims = []
    for d in tt.shape.dim:
        if d.HasField("dim_value"):
            dims.append(int(d.dim_value))
        else:
            dims.append(None)
    return dims

def _yolo_grid(H, W):
    # Approximate a 3-scale YOLO grid: P3/8, P4/16, and P5/32.
    return (H // 8) * (W // 8) + (H // 16) * (W // 16) + (H // 32) * (W // 32)

def _tensor_bytes_with_override(vi, batch, H, W, default_c=3, override_elem_bytes: int | None = None):
    """Infer tensor bytes for image/feature-map tensors, assuming NCHW."""
    dims = _dims_from_vi(vi)
    if not dims:
        return 0
    ebytes = override_elem_bytes if override_elem_bytes is not None else _elem_bytes(vi)
    filled = []
    for i, d in enumerate(dims):
        if d is not None and d > 0:
            filled.append(int(d))
        else:
            if i == 0:
                filled.append(int(batch))
            elif i == 1:
                filled.append(int(default_c))
            elif i == 2:
                filled.append(int(H))
            elif i == 3:
                filled.append(int(W))
            else:
                filled.append(1)
    return int(_prod(filled) * ebytes)

def _tensor_bytes_output_yolo_aware(vi, batch, H, W, override_elem_bytes: int | None = None):
    """Estimate YOLO-head output bytes when dynamic estimation is enabled."""
    dims = _dims_from_vi(vi)
    if not dims:
        return 0
    ebytes = override_elem_bytes if override_elem_bytes is not None else _elem_bytes(vi)
    rank = len(dims)
    filled = []
    for i, d in enumerate(dims):
        if d is not None and d > 0:
            filled.append(int(d))
        else:
            if i == 0:
                filled.append(int(batch))
            elif rank == 4 and i in (2, 3):
                filled.append(int(H if i == 2 else W))
            elif rank == 3 and i == 1:
                filled.append(int(_yolo_grid(H, W)))
            else:
                filled.append(1)
    return int(_prod(filled) * ebytes)

def _normalize_workload(name: str) -> str:
    if not isinstance(name, str):
        return "det"
    key = name.strip().lower()
    return WORKLOAD_ALIASES.get(key, key)

def extract_global_features(
    onnx_path,
    workload, batch, H, W, precision,
    *,
    force_fixed_output: bool = FORCE_FIXED_OUTPUT,
    fixed_output_bytes_by_wl: dict | None = None,
    return_workload_lower: bool = True,
):
    """
    force_fixed_output=True applies workload-specific fixed output bytes that
    match the training CSV distribution. fixed_output_bytes_by_wl can replace
    the default map. return_workload_lower=True returns a lowercase workload.
    """
    wl_norm = _normalize_workload(workload)
    fixed_map = (fixed_output_bytes_by_wl or FIXED_OUTPUT_BYTES_BY_WL_DEFAULT)
    override_b = _precision_to_elem_bytes(precision)

    # Load ONNX and enrich missing shapes when possible.
    m = onnx.load(onnx_path)
    try:
        m = shape_inference.infer_shapes(m)
    except Exception:
        pass

    g = m.graph
    init_map = {init.name: init for init in g.initializer}
    vi_map = {vi.name: vi for vi in list(g.input) + list(g.value_info) + list(g.output)}

    # Real graph inputs and outputs.
    real_inputs  = [vi for vi in g.input if vi.name not in init_map]
    real_outputs = [vi_map.get(vo.name, vo) for vo in g.output]

    # Total input bytes, adjusted by precision when provided.
    total_input_bytes = 0
    for vi in real_inputs:
        total_input_bytes += _tensor_bytes_with_override(vi, batch, H, W, override_elem_bytes=override_b)

    # Output bytes: fixed defaults or dynamic estimates.
    if force_fixed_output:
        # EST is normalized to pose.
        wl_key = "pose" if wl_norm == "est" else wl_norm
        total_output_bytes = int(fixed_map.get(wl_key, 0))
    else:
        total_output_bytes = 0
        if USE_YOLO_AWARE and wl_norm in {"det", "obb", "pose"}:
            for vi in real_outputs:
                total_output_bytes += _tensor_bytes_output_yolo_aware(vi, batch, H, W, override_elem_bytes=override_b)
        else:
            for vi in real_outputs:
                total_output_bytes += _tensor_bytes_with_override(vi, batch, H, W, override_elem_bytes=override_b)

    # Weights and parameters.
    weight_bytes = 0
    param_count  = 0
    for w in g.initializer:
        arr = numpy_helper.to_array(w)
        weight_bytes += int(arr.nbytes)
        param_count  += int(arr.size)

    # Operator counts.
    op_cnt = Counter(n.op_type for n in g.node)

    # Model size, derived from the filename.
    base = os.path.basename(onnx_path).lower()
    mm = re.search(r"yolov8([nsm lx])".replace(" ", ""), base)
    model_size = mm.group(1) if mm else "n"

    ret_wl = wl_norm if return_workload_lower else workload
    return {
        "workload": str(ret_wl),
        "model_size": model_size,
        "batch": int(batch),
        "H": int(H),
        "W": int(W),
        "precision": str(precision).lower(),
        "num_inputs": int(len(real_inputs)),
        "num_outputs": int(len(real_outputs)),
        "total_input_bytes": int(total_input_bytes),
        "total_output_bytes": int(total_output_bytes),
        "weight_bytes": int(weight_bytes),
        "param_count": int(param_count),
        "op_Add": int(op_cnt.get("Add", 0)),
        "op_Concat": int(op_cnt.get("Concat", 0)),
        "op_Constant": int(op_cnt.get("Constant", 0)),
        "op_Conv": int(op_cnt.get("Conv", 0)),
        "op_GlobalAveragePool": int(op_cnt.get("GlobalAveragePool", 0)),
        "op_Mul": int(op_cnt.get("Mul", 0)),
        "op_Reshape": int(op_cnt.get("Reshape", 0)),
        "op_Sigmoid": int(op_cnt.get("Sigmoid", 0)),
        "op_Split": int(op_cnt.get("Split", 0)),
    }
