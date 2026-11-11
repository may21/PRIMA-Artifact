#!/usr/bin/env python3
"""Leakage-free evaluation of the PRIMA memory predictor (reviewer comment D).

The training CSV holds repeated measurements: 1,547 rows but only 368
distinct configurations (feature vectors). A random row split therefore puts
copies of the same configuration in both the training and the test set.
This script evaluates the paper's regressors and two simple baselines under
splits that keep (a) a configuration and (b) a whole model out of training.

  python3 scripts/analysis/evaluate_predictor.py \
      --csv datasets/predictor_memory_features.csv \
      --output-dir results/predictor_evaluation

Outputs in <output-dir>:
  dataset_summary.md / .csv   models, configurations, repeats, target range
  results.md / results.csv    metrics per subset, protocol and regressor
  per_model_rf.csv            random forest, leave-one-model-out, per model
  predictions.csv             every out-of-fold prediction
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

FEATURES = [
    "workload", "model_size", "batch", "H", "W", "precision", "num_inputs", "num_outputs",
    "total_input_bytes", "total_output_bytes", "weight_bytes", "param_count",
    "op_Add", "op_Concat", "op_Constant", "op_Conv", "op_GlobalAveragePool", "op_Mul",
    "op_Reshape", "op_Sigmoid", "op_Split",
]
CATEGORICAL = ["workload", "model_size", "precision"]
YOLO_SCALES = {"n", "s", "m", "l", "x"}
TASK_SUFFIX = {"cls": "-cls", "det": "", "pose": "-pose", "seg": "-seg", "obb": "-obb"}

# The CSV does not name the non-YOLO models. These names are inferred from
# orin1's profiling scripts (profile_non_yolo_dataset.py: TensorRT/trtexec;
# profile_pytorch_fallback.py: torchvision detection models in PyTorch) and
# from the ONNX parameter counts. Confirm with the authors.
NON_YOLO = {
    5267540: ("EfficientNet-B0", "trtexec"),
    25530472: ("ResNet-50", "trtexec"),
    86567656: ("ViT-B/16", "trtexec"),
    41975470: ("DeepLabV3-ResNet50", "trtexec"),
    34068355: ("RetinaNet-R50-FPN", "PyTorch"),
    41808636: ("Faster R-CNN-R50-FPN", "PyTorch"),
    59190527: ("Keypoint R-CNN-R50-FPN", "PyTorch"),
    44454754: ("Mask R-CNN-R50-FPN", "PyTorch"),
}


def load(csv_path: Path, target: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"model_size": str})
    df["family"] = np.where(df.model_size.isin(YOLO_SCALES), "YOLOv8", "non-YOLO")
    yolo = df.family == "YOLOv8"
    df["model"] = ""
    df.loc[yolo, "model"] = "yolov8" + df.loc[yolo, "model_size"] + df.loc[yolo, "workload"].map(TASK_SUFFIX)
    df.loc[~yolo, "model"] = df.loc[~yolo, "param_count"].map(lambda p: NON_YOLO.get(int(p), (f"unknown-{p}", "?"))[0])
    df["runtime"] = np.where(yolo, "TensorRT (Ultralytics)", df.param_count.map(lambda p: NON_YOLO.get(int(p), ("", "?"))[1]))
    df["config"] = df.groupby(FEATURES, sort=False).ngroup()
    df["y"] = df[target].astype(float)
    return df


def make(estimator, numeric=None, categorical=True) -> Pipeline:
    numeric = numeric or [c for c in FEATURES if c not in CATEGORICAL]
    transformers = [("num", StandardScaler(with_mean=False), numeric)]
    if categorical:
        transformers.insert(0, ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL))
    return Pipeline([("prep", ColumnTransformer(transformers, remainder="drop", sparse_threshold=1.0)), ("reg", estimator)])


def regressors(seed: int) -> dict:
    """The paper's four regressors (settings of the original compare_models.py) and two baselines."""
    models = {
        "Random forest": lambda: make(RandomForestRegressor(n_estimators=500, max_depth=None, n_jobs=-1, random_state=seed)),
        "SVR": lambda: make(SVR(C=10.0, epsilon=5.0, kernel="rbf", gamma="scale")),
        "Mean (baseline)": lambda: make(DummyRegressor(strategy="mean")),
        "Linear on weight size (baseline)": lambda: make(LinearRegression(), numeric=["weight_bytes"], categorical=False),
    }
    try:
        import xgboost as xgb
        models["XGBoost"] = lambda: make(xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", tree_method="hist", random_state=seed))
    except ImportError:
        print("[WARN] xgboost not installed; skipped")
    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = lambda: make(LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            random_state=seed, verbose=-1))
    except ImportError:
        print("[WARN] lightgbm not installed; skipped")
    order = ["Random forest", "XGBoost", "LightGBM", "SVR", "Linear on weight size (baseline)", "Mean (baseline)"]
    return {name: models[name] for name in order if name in models}


def splits(df: pd.DataFrame, protocol: str, seed: int):
    """Yield (train_index, test_index) positional index arrays."""
    idx = np.arange(len(df))
    if protocol == "random rows 8:2":
        yield tuple(train_test_split(idx, test_size=0.2, random_state=seed))
    elif protocol == "grouped by configuration":
        yield from GroupKFold(n_splits=5, shuffle=True, random_state=seed).split(idx, groups=df.config)
    elif protocol == "leave one model out":
        yield from LeaveOneGroupOut().split(idx, groups=df.model)
    elif protocol == "leave one task out":
        yield from LeaveOneGroupOut().split(idx, groups=df.workload)
    elif protocol == "leave one scale out":
        yield from LeaveOneGroupOut().split(idx, groups=df.model_size)
    elif protocol == "train YOLOv8, test non-YOLO":
        yield idx[(df.family == "YOLOv8").values], idx[(df.family != "YOLOv8").values]
    elif protocol == "train non-YOLO, test YOLOv8":
        yield idx[(df.family != "YOLOv8").values], idx[(df.family == "YOLOv8").values]
    else:
        raise ValueError(protocol)


def metrics(y, p) -> dict:
    err = p - y
    ape = np.abs(err) / y * 100
    ss_res, ss_tot = float(np.sum(err ** 2)), float(np.sum((y - y.mean()) ** 2))
    return {
        "n_test": len(y),
        "MAE_MB": float(np.mean(np.abs(err))),
        "RMSE_MB": float(np.sqrt(np.mean(err ** 2))),
        "R2": 1 - ss_res / ss_tot if ss_tot else float("nan"),
        "MAPE_pct": float(np.mean(ape)),
        "P90_APE_pct": float(np.percentile(ape, 90)),
        "under_pct": float(np.mean(err < 0) * 100),
        "mean_signed_MB": float(np.mean(err)),
    }


def repeat_noise(df: pd.DataFrame) -> dict:
    """Error of predicting a measurement by the mean of the other repeats of the same configuration."""
    g = df.groupby("config")["y"]
    n, s = g.transform("count"), g.transform("sum")
    keep = n > 1
    pred = (s[keep] - df.y[keep]) / (n[keep] - 1)
    return metrics(df.y[keep].values, pred.values)


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, model), g in df.groupby(["family", "model"], sort=False):
        rows.append({
            "family": family, "model": model, "task": g.workload.iloc[0], "runtime": g.runtime.iloc[0],
            "parameters_M": round(g.param_count.iloc[0] / 1e6, 1), "rows": len(g), "configurations": g.config.nunique(),
            "repeats_per_config": f"{g.groupby('config').size().min()}-{g.groupby('config').size().max()}",
            "input_px": "/".join(str(v) for v in sorted(g.H.unique())), "batch": "/".join(str(v) for v in sorted(g.batch.unique())),
            "precision": "/".join(sorted(g.precision.unique())),
            "target_min_MB": int(g.y.min()), "target_max_MB": int(g.y.max()),
        })
    return pd.DataFrame(rows).sort_values(["family", "task", "parameters_M"], ascending=[False, True, True])


def md_table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=Path("datasets/predictor_memory_features.csv"))
    ap.add_argument("--target", default="Max_cpu_mem_MB")
    ap.add_argument("--output-dir", type=Path, default=Path("results/predictor_evaluation"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    df = load(args.csv, args.target)
    summary = dataset_summary(df)
    summary.to_csv(out / "dataset_summary.csv", index=False)

    subsets = {
        "All rows (YOLOv8 + non-YOLO)": (df, ["random rows 8:2", "grouped by configuration", "leave one model out",
                                              "train YOLOv8, test non-YOLO", "train non-YOLO, test YOLOv8"]),
        "YOLOv8 rows only": (df[df.family == "YOLOv8"].reset_index(drop=True),
                             ["random rows 8:2", "grouped by configuration", "leave one model out",
                              "leave one task out", "leave one scale out"]),
    }
    models = regressors(args.seed)
    results, predictions = [], []
    for subset, (data, protocols) in subsets.items():
        noise = repeat_noise(data)
        results.append({"subset": subset, "protocol": "repeat-measurement noise (reference)", "regressor": "mean of other repeats", **noise})
        for protocol in protocols:
            for name, build in models.items():
                y_true, y_pred, rows = [], [], []
                for train, test in splits(data, protocol, args.seed):
                    model = build()
                    model.fit(data.iloc[train][FEATURES], data.iloc[train].y)
                    y_pred.append(model.predict(data.iloc[test][FEATURES]))
                    y_true.append(data.iloc[test].y.values)
                    rows.append(data.iloc[test][["family", "model", "workload", "batch", "H", "precision", "config"]])
                y, p = np.concatenate(y_true), np.concatenate(y_pred)
                results.append({"subset": subset, "protocol": protocol, "regressor": name, **metrics(y, p)})
                frame = pd.concat(rows).assign(subset=subset, protocol=protocol, regressor=name, actual_MB=y, predicted_MB=p)
                predictions.append(frame)
                print(f"{subset[:8]:8} | {protocol:28} | {name:34} MAE {results[-1]['MAE_MB']:8.1f} MB  MAPE {results[-1]['MAPE_pct']:6.1f}%  R2 {results[-1]['R2']:7.3f}", flush=True)

    res = pd.DataFrame(results)
    res.to_csv(out / "results.csv", index=False)
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(out / "predictions.csv", index=False)

    rf = pred[(pred.regressor == "Random forest") & (pred.protocol == "leave one model out") & (pred.subset.str.startswith("All"))]
    per_model = rf.assign(err=rf.predicted_MB - rf.actual_MB).groupby(["family", "model"]).apply(
        lambda g: pd.Series({"rows": len(g), "actual_mean_MB": g.actual_MB.mean(), "predicted_mean_MB": g.predicted_MB.mean(),
                             "MAE_MB": g.err.abs().mean(), "MAPE_pct": (g.err.abs() / g.actual_MB).mean() * 100,
                             "under_pct": (g.err < 0).mean() * 100})).reset_index().round(1)
    per_model.to_csv(out / "per_model_rf.csv", index=False)

    n_cfg = df.config.nunique()
    lines = ["# Predictor evaluation", "",
             f"Data: `{args.csv}`, target `{args.target}`. {len(df)} rows, {n_cfg} distinct configurations "
             f"({(df.family == 'YOLOv8').sum()} YOLOv8 rows in {df[df.family == 'YOLOv8'].config.nunique()} configurations, "
             f"{(df.family != 'YOLOv8').sum()} non-YOLO rows in {df[df.family != 'YOLOv8'].config.nunique()} configurations).", "",
             "## Dataset", "", md_table(summary), "",
             "Non-YOLO model names are inferred from the profiling scripts and parameter counts.", ""]
    for subset in subsets:
        lines += [f"## {subset}", ""]
        for metric, title, fmt in [("MAE_MB", "MAE (MB)", "{:.1f}"), ("MAPE_pct", "MAPE (%)", "{:.1f}"), ("R2", "R²", "{:.3f}"),
                                   ("under_pct", "Share of under-predictions (%)", "{:.0f}")]:
            sub = res[res.subset == subset]
            table = sub.pivot(index="protocol", columns="regressor", values=metric)
            table = table.reindex(index=list(dict.fromkeys(sub.protocol)), columns=[c for c in ["mean of other repeats", *models] if c in table.columns])
            table = table.map(lambda v: "" if pd.isna(v) else fmt.format(v)).reset_index()
            lines += [f"**{title}**", "", md_table(table), ""]
    lines += ["## Random forest, leave one model out, per held-out model (all rows)", "", md_table(per_model), ""]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    (out / "dataset_summary.md").write_text(md_table(summary) + "\n")
    print(f"[written] {out}/results.md, results.csv, per_model_rf.csv, predictions.csv, dataset_summary.md")


if __name__ == "__main__":
    main()
