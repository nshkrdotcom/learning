from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from construct_mismatch.datasets import CONSTRUCTS, DECOUPLING_AXES, artifact_path

METHODS = ("direction", "probe", "patching")
EVALUATION_AXES = (
    "ordinary_heldout",
    "lexical_reversal",
    "negation",
    "quotation",
    "contrast",
    "format_shift",
    "causal_steering",
    "specificity",
)
STATUS_LABELS = (
    "pass",
    "weak",
    "fail",
    "not_applicable",
    "behavior_absent",
    "not_run",
)
OBJECT_LABELS = (
    "robust_construct_variable",
    "ordinary_only_proxy",
    "causal_but_nonspecific_handle",
    "predictive_noncausal_detector",
    "prompt_local_dependency",
    "no_reliable_object",
    "behavior_absent_or_weak",
)


def accuracy_status(value: float | None) -> str:
    if value is None or np.isnan(value):
        return "not_run"
    if value >= 0.70:
        return "pass"
    if value >= 0.55:
        return "weak"
    return "fail"


def effect_status(value: float | None, weak: float, passing: float) -> str:
    if value is None or np.isnan(value):
        return "not_run"
    if value >= passing:
        return "pass"
    if value >= weak:
        return "weak"
    return "fail"


def specificity_status(kl_value: float | None) -> str:
    if kl_value is None or np.isnan(kl_value):
        return "not_run"
    if kl_value <= 0.05:
        return "pass"
    if kl_value <= 0.20:
        return "weak"
    return "fail"


def patching_stability_status(stability: float | None) -> str:
    if stability is None or np.isnan(stability):
        return "not_run"
    if stability > 0.66:
        return "pass"
    if stability >= 0.50:
        return "weak"
    return "fail"


def construct_behavior_absent(root: Path, construct: str) -> bool:
    path = artifact_path(root) / "behavior" / "behavior_summary.csv"
    if not path.exists():
        return False
    df = pd.read_csv(path)
    rows = df[
        (df["construct"] == construct)
        & (df["split"] == "heldout")
        & (df["decoupling_axis"] == "ordinary")
    ]
    if rows.empty:
        return False
    return str(rows.iloc[0]["behavior_status"]) != "usable"


def selected_direction_layer(df: pd.DataFrame) -> int | None:
    if "accuracy" not in df:
        return None
    heldout = df[
        (df["baseline_type"] == "direction")
        & (df["split"] == "heldout")
        & (df["decoupling_axis"] == "ordinary")
    ]
    if heldout.empty:
        return None
    return int(heldout.sort_values("accuracy", ascending=False).iloc[0]["layer"])


def selected_probe_layer(df: pd.DataFrame) -> int | None:
    if "accuracy" not in df:
        return None
    heldout = df[(df["split"] == "heldout") & (df["decoupling_axis"] == "ordinary")]
    if heldout.empty:
        return None
    return int(heldout.sort_values("accuracy", ascending=False).iloc[0]["layer"])


def direction_rows(root: Path, construct: str) -> list[dict[str, object]]:
    path = artifact_path(root) / "directions" / f"{construct}_direction_metrics.csv"
    steering_path = artifact_path(root) / "directions" / f"{construct}_steering_metrics.csv"
    rows: list[dict[str, object]] = []
    if not path.exists():
        return [
            {"construct": construct, "method": "direction", "evaluation_axis": axis, "status": "not_run"}
            for axis in EVALUATION_AXES
        ]
    df = pd.read_csv(path)
    if "status" in df.columns and "accuracy" not in df.columns:
        return [
            {
                "construct": construct,
                "method": "direction",
                "evaluation_axis": axis,
                "status": "behavior_absent" if "behavior_absent" in set(df["status"]) else "not_run",
            }
            for axis in EVALUATION_AXES
        ]
    layer = selected_direction_layer(df)
    for axis in ("ordinary_heldout", *DECOUPLING_AXES):
        if layer is None:
            status = "not_run"
            value = np.nan
        elif axis == "ordinary_heldout":
            match = df[
                (df["baseline_type"] == "direction")
                & (df["split"] == "heldout")
                & (df["decoupling_axis"] == "ordinary")
                & (df["layer"] == layer)
            ]
            value = float(match.iloc[0]["accuracy"]) if not match.empty else np.nan
            status = accuracy_status(value)
        else:
            match = df[
                (df["baseline_type"] == "direction")
                & (df["split"] == "decoupling")
                & (df["decoupling_axis"] == axis)
                & (df["layer"] == layer)
            ]
            value = float(match.iloc[0]["accuracy"]) if not match.empty else np.nan
            status = accuracy_status(value)
        rows.append(
            {
                "construct": construct,
                "method": "direction",
                "evaluation_axis": axis,
                "status": status,
                "value": value,
                "selected_layer": layer,
            }
        )
    if steering_path.exists():
        steering = pd.read_csv(steering_path)
    else:
        steering = pd.DataFrame()
    if not steering.empty and "mean_abs_raw_logit_shift" in steering:
        candidate = steering[
            (steering["baseline_type"] == "direction")
            & (steering["alpha"].abs() == steering["alpha"].abs().max())
        ]
        if layer is not None:
            candidate = candidate[candidate["layer"] == layer]
        effect = float(candidate["mean_abs_raw_logit_shift"].max()) if not candidate.empty else np.nan
        kl = float(candidate["mean_kl_divergence"].mean()) if not candidate.empty else np.nan
        rows.append(
            {
                "construct": construct,
                "method": "direction",
                "evaluation_axis": "causal_steering",
                "status": effect_status(effect, weak=0.20, passing=0.50),
                "value": effect,
                "selected_layer": layer,
            }
        )
        rows.append(
            {
                "construct": construct,
                "method": "direction",
                "evaluation_axis": "specificity",
                "status": specificity_status(kl),
                "value": kl,
                "selected_layer": layer,
            }
        )
    else:
        rows.extend(
            [
                {
                    "construct": construct,
                    "method": "direction",
                    "evaluation_axis": "causal_steering",
                    "status": "not_run",
                    "value": np.nan,
                    "selected_layer": layer,
                },
                {
                    "construct": construct,
                    "method": "direction",
                    "evaluation_axis": "specificity",
                    "status": "not_run",
                    "value": np.nan,
                    "selected_layer": layer,
                },
            ]
        )
    return rows


def probe_rows(root: Path, construct: str) -> list[dict[str, object]]:
    path = artifact_path(root) / "probes" / f"{construct}_probe_metrics.csv"
    if not path.exists():
        return [
            {"construct": construct, "method": "probe", "evaluation_axis": axis, "status": "not_run"}
            for axis in EVALUATION_AXES
        ]
    df = pd.read_csv(path)
    if "status" in df.columns and "accuracy" not in df.columns:
        return [
            {
                "construct": construct,
                "method": "probe",
                "evaluation_axis": axis,
                "status": "behavior_absent" if "behavior_absent" in set(df["status"]) else "not_run",
            }
            for axis in EVALUATION_AXES
        ]
    layer = selected_probe_layer(df)
    rows: list[dict[str, object]] = []
    for axis in ("ordinary_heldout", *DECOUPLING_AXES):
        if layer is None:
            status = "not_run"
            value = np.nan
        elif axis == "ordinary_heldout":
            match = df[
                (df["split"] == "heldout")
                & (df["decoupling_axis"] == "ordinary")
                & (df["layer"] == layer)
            ]
            value = float(match.iloc[0]["accuracy"]) if not match.empty else np.nan
            status = accuracy_status(value)
        else:
            match = df[
                (df["split"] == "decoupling")
                & (df["decoupling_axis"] == axis)
                & (df["layer"] == layer)
            ]
            value = float(match.iloc[0]["accuracy"]) if not match.empty else np.nan
            status = accuracy_status(value)
        rows.append(
            {
                "construct": construct,
                "method": "probe",
                "evaluation_axis": axis,
                "status": status,
                "value": value,
                "selected_layer": layer,
            }
        )
    rows.append(
        {
            "construct": construct,
            "method": "probe",
            "evaluation_axis": "causal_steering",
            "status": "not_applicable",
            "value": np.nan,
            "selected_layer": layer,
        }
    )
    rows.append(
        {
            "construct": construct,
            "method": "probe",
            "evaluation_axis": "specificity",
            "status": "not_applicable",
            "value": np.nan,
            "selected_layer": layer,
        }
    )
    return rows


def patching_rows(root: Path, construct: str) -> list[dict[str, object]]:
    path = artifact_path(root) / "patching" / f"{construct}_patching_metrics.csv"
    top_path = artifact_path(root) / "patching" / f"{construct}_top_sites.csv"
    if not path.exists():
        return [
            {"construct": construct, "method": "patching", "evaluation_axis": axis, "status": "not_run"}
            for axis in EVALUATION_AXES
        ]
    df = pd.read_csv(path)
    if "status" in df.columns and "recovery" not in df.columns:
        return [
            {
                "construct": construct,
                "method": "patching",
                "evaluation_axis": axis,
                "status": "behavior_absent" if "behavior_absent" in set(df["status"]) else "not_run",
            }
            for axis in EVALUATION_AXES
        ]
    rows: list[dict[str, object]] = []
    for axis in ("ordinary_heldout", *DECOUPLING_AXES):
        raw_axis = "ordinary" if axis == "ordinary_heldout" else axis
        match = df[df["decoupling_axis"] == raw_axis]
        if match.empty:
            value = np.nan
            status = "not_run"
        else:
            per_pair = match.groupby("pair_id")["recovery"].apply(
                lambda values: values.clip(lower=0.0, upper=1.0).max()
            )
            value = float(per_pair.mean())
            status = effect_status(value, weak=0.10, passing=0.20)
        rows.append(
            {
                "construct": construct,
                "method": "patching",
                "evaluation_axis": axis,
                "status": status,
                "value": value,
                "selected_layer": np.nan,
            }
        )
    rows.append(
        {
            "construct": construct,
            "method": "patching",
            "evaluation_axis": "causal_steering",
            "status": "not_applicable",
            "value": np.nan,
            "selected_layer": np.nan,
        }
    )
    if top_path.exists():
        top = pd.read_csv(top_path)
        stability = (
            float(top["axis_top_site_stability"].mean())
            if "axis_top_site_stability" in top and not top.empty
            else np.nan
        )
        status = patching_stability_status(stability)
    else:
        stability = np.nan
        status = "not_run"
    rows.append(
        {
            "construct": construct,
            "method": "patching",
            "evaluation_axis": "specificity",
            "status": status,
            "value": stability,
            "selected_layer": np.nan,
        }
    )
    return rows


def classify_object(rows: pd.DataFrame, construct: str, method: str) -> str:
    subset = rows[(rows["construct"] == construct) & (rows["method"] == method)]
    statuses = {row.evaluation_axis: row.status for row in subset.itertuples()}
    if any(status == "behavior_absent" for status in statuses.values()):
        return "behavior_absent_or_weak"
    ordinary = statuses.get("ordinary_heldout", "not_run")
    decoupling = [statuses.get(axis, "not_run") for axis in DECOUPLING_AXES]
    if ordinary in {"fail", "not_run"}:
        return "no_reliable_object"
    if method == "probe":
        return "predictive_noncausal_detector"
    if method == "patching":
        if statuses.get("specificity") in {"fail", "weak"} and ordinary in {"pass", "weak"}:
            return "prompt_local_dependency"
        if ordinary == "pass" and decoupling.count("pass") >= 3 and statuses.get("specificity") == "pass":
            return "robust_construct_variable"
        return "no_reliable_object"
    if method == "direction":
        if statuses.get("causal_steering") in {"pass", "weak"} and statuses.get("specificity") == "fail":
            return "causal_but_nonspecific_handle"
        if ordinary == "pass" and any(status == "fail" for status in decoupling):
            return "ordinary_only_proxy"
        if ordinary == "pass" and decoupling.count("pass") + decoupling.count("weak") >= 4:
            if statuses.get("causal_steering") in {"pass", "weak"} and statuses.get("specificity") in {
                "pass",
                "weak",
            }:
                return "robust_construct_variable"
        return "no_reliable_object"
    return "no_reliable_object"


def write_matrix_heatmap(matrix: pd.DataFrame, output_path: Path) -> None:
    status_to_value = {
        "pass": 4,
        "weak": 3,
        "fail": 2,
        "not_applicable": 1,
        "behavior_absent": 0,
        "not_run": 0,
    }
    matrix = matrix.copy()
    matrix["row_label"] = matrix["construct"] + " / " + matrix["method"]
    pivot = matrix.pivot(index="row_label", columns="evaluation_axis", values="status")
    pivot = pivot.reindex(columns=EVALUATION_AXES)
    values = pivot.replace(status_to_value).astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    image = ax.imshow(values, aspect="auto", vmin=0, vmax=4, cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, str(pivot.iloc[i, j]), ha="center", va="center", color="white", fontsize=7)
    ax.set_title("Construct mismatch matrix")
    fig.colorbar(image, ax=ax, ticks=list(status_to_value.values()))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def generate_scoring_artifacts(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for construct in CONSTRUCTS:
        if construct_behavior_absent(root, construct):
            for method in METHODS:
                rows.extend(
                    {
                        "construct": construct,
                        "method": method,
                        "evaluation_axis": axis,
                        "status": "behavior_absent",
                        "value": np.nan,
                        "selected_layer": np.nan,
                    }
                    for axis in EVALUATION_AXES
                )
            continue
        rows.extend(direction_rows(root, construct))
        rows.extend(probe_rows(root, construct))
        rows.extend(patching_rows(root, construct))
    matrix = pd.DataFrame(rows)
    for status in matrix["status"]:
        if status not in STATUS_LABELS:
            raise ValueError(f"Invalid scoring status: {status}")

    classifications = pd.DataFrame(
        [
            {
                "construct": construct,
                "method": method,
                "object_classification": classify_object(matrix, construct, method),
            }
            for construct in CONSTRUCTS
            for method in METHODS
        ]
    )
    for label in classifications["object_classification"]:
        if label not in OBJECT_LABELS:
            raise ValueError(f"Invalid object label: {label}")

    output_dir = artifact_path(root) / "scoring"
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_dir / "construct_mismatch_matrix.csv", index=False)
    classifications.to_csv(output_dir / "object_classifications.csv", index=False)
    write_matrix_heatmap(matrix, output_dir / "construct_mismatch_matrix.png")
    return matrix, classifications
