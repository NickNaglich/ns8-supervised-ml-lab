from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
import hashlib


def load_metrics_file(path: Path) -> Dict[str, Any]:
    """Load metrics JSON safely."""
    record = _evidence_json(path)
    return record["value"] if record["status"] == "recorded" else {}


def load_dataset_signature_from_run(run_path: Path | str) -> Optional[str]:
    """Attempt to load a dataset signature stored alongside a run."""
    p = Path(run_path)
    metrics_path = p / "metrics.json"
    metrics = load_metrics_file(metrics_path) if metrics_path.exists() else {}
    sig = metrics.get("dataset_signature") or metrics.get("dataset_sig")
    if sig:
        return str(sig)
    for candidate in ["dataset_signature.txt", "dataset_sig.txt", "dataset_signature.json"]:
        cand_path = p / candidate
        if not cand_path.exists():
            continue
        try:
            if cand_path.suffix == ".json":
                data = json.loads(cand_path.read_text())
                sig_val = data.get("signature") or data.get("dataset_signature")
                if sig_val:
                    return str(sig_val)
            else:
                sig_val = cand_path.read_text().strip()
                if sig_val:
                    return sig_val
        except Exception:
            continue
    return None


def load_ui_settings() -> Dict[str, Any]:
    cfg_path = Path("ui/config.yaml")
    settings: Dict[str, Any] = {"mode": "read-only"}
    if cfg_path.exists():
        try:
            import yaml
            settings.update(yaml.safe_load(cfg_path.read_text()) or {})
        except Exception:
            pass
    # Environment override for mode
    env_mode = os.getenv("NS8_UI_MODE")
    if env_mode:
        settings["mode"] = env_mode
    return settings


def load_results_snapshot() -> Optional[List[Dict[str, Any]]]:
    path = Path("reports/results/results.md")
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    rows: List[Dict[str, Any]] = []
    header_index = None
    for idx, line in enumerate(lines):
        if line.startswith("| Task "):
            header_index = idx
            break
    if header_index is None:
        return None
    for row_line in lines[header_index + 2 :]:
        if not row_line.startswith("|"):
            continue
        parts = [p.strip() for p in row_line.strip("|").split("|")]
        if len(parts) < 6:
            continue
        task, grouping, seed, model, metrics, config = parts[:6]
        rows.append(
            {
                "task": task,
                "grouping": grouping,
                "seed": seed,
                "model": model,
                "metrics": metrics,
                "config": config,
            }
        )
    return rows or None


def list_configs() -> List[str]:
    cfg_dir = Path("configs")
    if not cfg_dir.exists():
        return []
    return sorted(str(p) for p in cfg_dir.glob("*.yaml"))


PRIMARY_METRIC = {
    "task_a_view": "f1_macro",
    "task_b_kbucket": "f1_macro",
    "task_c_regress_n": "r2",
}


def index_runs(experiments_path: str = "reports/experiments", runs_path: str = "runs") -> List[Dict[str, Any]]:
    """Scan experiment and run folders and normalize basic run metadata."""
    entries: List[Dict[str, Any]] = []

    exp_root = Path(experiments_path)
    if exp_root.exists():
        for ts_dir in sorted(exp_root.iterdir(), reverse=True):
            if not ts_dir.is_dir():
                continue
            for model_dir in ts_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                metrics = load_metrics_file(model_dir / "metrics.json")
                run_id = model_dir.name
                task = metrics.get("task")
                metric_name = PRIMARY_METRIC.get(task, "f1_macro")
                primary_metric = metrics.get(metric_name)
                badge = {
                    "metrics": (model_dir / "metrics.json").exists(),
                    "config": (model_dir / "config.yaml").exists(),
                    "cv": (model_dir / "cv_results.csv").exists(),
                    "figures": (model_dir / "confusion_matrix.png").exists(),
                }
                entries.append(
                    {
                        "source": "experiments",
                        "timestamp": ts_dir.name,
                        "run_id": f"{ts_dir.name}_{run_id}",
                        "task": task,
                        "model": run_id,
                        "primary_metric_name": metric_name,
                        "primary_metric": primary_metric,
                        "metrics_path": str(model_dir / "metrics.json"),
                        "config_path": str(model_dir / "config.yaml"),
                        "cv_path": str(model_dir / "cv_results.csv"),
                        "figures": [str(model_dir / "confusion_matrix.png")] if badge["figures"] else [],
                        "run_path": str(model_dir),
                        "artifact_badges": badge,
                        "dataset_signature": load_dataset_signature_from_run(model_dir),
                    }
                )

    runs_root = Path(runs_path)
    if runs_root.exists():
        for run_dir in sorted(runs_root.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            metrics = load_metrics_file(run_dir / "metrics.json")
            task = metrics.get("task")
            metric_name = PRIMARY_METRIC.get(task, "f1_macro")
            primary_metric = metrics.get(metric_name)
            badge = {
                "metrics": (run_dir / "metrics.json").exists(),
                "config": (run_dir / "config.json").exists() or (run_dir / "config.yaml").exists(),
                "cv": (run_dir / "cv_results.csv").exists(),
                "figures": (run_dir / "confusion_matrix.png").exists(),
            }
            entries.append(
                {
                    "source": "runs",
                    "timestamp": run_dir.name.split("_")[0] if "_" in run_dir.name else run_dir.name,
                    "run_id": run_dir.name,
                    "task": task,
                    "model": run_dir.name.split("_")[-1],
                    "primary_metric_name": metric_name,
                    "primary_metric": primary_metric,
                    "metrics_path": str(run_dir / "metrics.json"),
                    "config_path": str(run_dir / "config.json") if (run_dir / "config.json").exists() else str(run_dir / "config.yaml"),
                    "cv_path": str(run_dir / "cv_results.csv"),
                    "figures": [str(run_dir / "confusion_matrix.png")] if badge["figures"] else [],
                    "run_path": str(run_dir),
                    "artifact_badges": badge,
                    "dataset_signature": load_dataset_signature_from_run(run_dir),
                }
            )

    return entries


def run_command(cmd: List[str]) -> List[str]:
    """Run a command and return stdout lines. Intended for gated run mode."""
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except Exception as exc:
        return [f"Failed to start command: {exc}"]
    lines: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line.rstrip())
    proc.wait()
    lines.append(f"Exit code: {proc.returncode}")
    return lines


def stream_command(cmd: List[str]):
    """Generator to stream command output line by line."""
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except Exception as exc:
        yield f"Failed to start command: {exc}"
        return
    assert proc.stdout is not None
    for line in proc.stdout:
        yield line.rstrip()
    yield f"Exit code: {proc.returncode}"


def stream_command_with_log(cmd: List[str], log_dir: str = "runs/ui_logs") -> List[str]:
    """Run command, stream lines, and persist to a log file."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"{ts}_ui.log"
    lines: List[str] = []
    with log_path.open("w", encoding="utf-8") as fh:
        for line in stream_command(cmd):
            fh.write(line + "\n")
            lines.append(line)
    lines.append(f"Log saved to {log_path}")
    return lines


def load_text_file(path: str) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return p.read_text()
    except Exception:
        return None


def load_figures(fig_dirs: Optional[List[str]] = None, patterns: Optional[List[str]] = None, recursive: bool = False) -> List[str]:
    """Load figure paths from one or more directories."""
    dirs = fig_dirs or ["reports/figures"]
    pats = patterns or ["*.png", "*.jpg"]
    images: List[str] = []
    for d in dirs:
        p = Path(d)
        if not p.exists():
            continue
        globber = p.rglob if recursive else p.glob
        for pat in pats:
            for img in globber(pat):
                if img.is_file():
                    images.append(str(img))
    return sorted(images)


def load_dataset_sample(path: str) -> Optional[Any]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        import pandas as pd

        if p.suffix.lower() in {".parquet"}:
            return pd.read_parquet(p)
        if p.suffix.lower() in {".csv"}:
            return pd.read_csv(p)
    except Exception:
        return None
    return None


def render_grid_from_row(row: Dict[str, Any]) -> Optional[Any]:
    try:
        from ns8lab.grids import generate_grid
        import numpy as np
    except Exception:
        return None
    N = int(row.get("N"))
    k = int(row.get("k"))
    view = str(row.get("view"))
    grid = generate_grid(N, k, view=view)
    return grid


def compute_feature_importance(
    model_path: str,
    dataset_path: str,
    feature_cols: List[str],
    target_col: str,
    save_path: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    try:
        import joblib
    except Exception:
        return None
    model_file = Path(model_path)
    data_file = Path(dataset_path)
    if not model_file.exists() or not data_file.exists():
        return None
    try:
        model = joblib.load(model_file)
        df = pd.read_csv(data_file) if data_file.suffix.lower() == ".csv" else pd.read_parquet(data_file)
    except Exception:
        return None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        df = pd.DataFrame({"feature": feature_cols, "importance": importances})
        dest = save_path or (model_file.parent / "feature_importance.csv")
        if dest:
            out_path = Path(dest)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                df.to_csv(out_path, index=False)
            except Exception:
                pass
        return df
    return None


def health_info(settings: Dict[str, Any]) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        import numpy as np

        info["numpy_version"] = np.__version__
    except Exception:
        info["numpy_version"] = "unavailable"
    try:
        import sklearn

        info["sklearn_version"] = sklearn.__version__
    except Exception:
        info["sklearn_version"] = "unavailable"
    try:
        import streamlit as st

        info["streamlit_version"] = st.__version__
    except Exception:
        info["streamlit_version"] = "unavailable"

    paths = {
        "experiments_path": settings.get("experiments_path", "reports/experiments"),
        "figures_path": settings.get("figures_path", "reports/figures"),
        "results_path": settings.get("results_path", "reports/results/results.md"),
        "runs_path": settings.get("runs_path", "runs"),
    }
    info["paths"] = {k: {"path": v, "exists": Path(v).exists()} for k, v in paths.items()}

    # Fingerprint placeholder
    info["fingerprint"] = {
        "computed": False,
        "dataset_sig": settings.get("dataset_sig", None),
    }
    info["mode"] = settings.get("mode", "read-only")
    return info


def clear_artifacts(paths: Optional[List[str]] = None) -> List[str]:
    """Dangerous: delete run/report artifacts. Returns list of cleared paths."""
    import shutil

    targets = paths or ["runs", "reports/experiments", "reports/figures"]
    cleared: List[str] = []
    for p in targets:
        path = Path(p)
        if path.exists():
            shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
            cleared.append(str(path))
    return cleared


def compute_dataset_signature(path: str) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        hasher = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None


def compute_run_fingerprint(config_path: str, dataset_sig: Optional[str]) -> Optional[str]:
    """Hash config content + dataset signature for a simple fingerprint."""
    if not dataset_sig:
        return None
    cfg = Path(config_path)
    if not cfg.exists():
        return None
    try:
        cfg_bytes = cfg.read_bytes()
        payload = cfg_bytes + dataset_sig.encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
    except Exception:
        return None


def layout_profile(device: str = "desktop", density: Optional[str] = None) -> Dict[str, Any]:
    """Return simple layout defaults per device class."""
    device = device.lower()
    base: Dict[str, Any]
    if device == "mobile":
        base = {
            "filters_in_columns": False,
            "table_height": 320,
            "card_mode": True,
            "default_topn": 20,
            "density": "tight",
            "padding_px": 10,
            "font_px": 14,
        }
    elif device == "tablet" or device == "phablet":
        base = {
            "filters_in_columns": True,
            "table_height": 420,
            "card_mode": False,
            "default_topn": 0,
            "density": "normal",
            "padding_px": 20,
            "font_px": 15,
        }
    else:
        # desktop / default
        base = {
            "filters_in_columns": True,
            "table_height": 520,
            "card_mode": False,
            "default_topn": 0,
            "density": "compact",
            "padding_px": 12,
            "font_px": 15,
        }
    if density:
        dens = density.lower()
        if dens == "compact":
            base["padding_px"] = max(8, base["padding_px"] - 4)
            base["font_px"] = max(13, base["font_px"] - 1)
            base["density"] = "compact"
        elif dens == "comfortable":
            base["padding_px"] = base["padding_px"] + 6
            base["font_px"] = base["font_px"] + 1
            base["density"] = "comfortable"
    return base


def load_recent_runs(limit: int = 5) -> List[Dict[str, Any]]:
    exp_dir = Path("reports/experiments")
    runs: List[Dict[str, Any]] = []
    if not exp_dir.exists():
        return runs
    for ts_dir in sorted(exp_dir.iterdir(), reverse=True):
        if not ts_dir.is_dir():
            continue
        for model_dir in ts_dir.iterdir():
            if not model_dir.is_dir():
                continue
            metrics_path = model_dir / "metrics.json"
            config_path = model_dir / "config.yaml"
            figures = []
            if metrics_path.exists():
                try:
                    metrics = json.loads(metrics_path.read_text())
                except Exception:
                    metrics = {}
            else:
                metrics = {}
            fig_path = model_dir / "confusion_matrix.png"
            if fig_path.exists():
                figures.append(str(fig_path))
            runs.append(
                {
                    "timestamp": ts_dir.name,
                    "model": model_dir.name,
                    "task": metrics.get("task"),
                    "metrics": metrics,
                    "config_text": config_path.read_text() if config_path.exists() else None,
                    "config_path": str(config_path) if config_path.exists() else None,
                    "figures": figures,
                    "run_path": str(model_dir),
                }
            )
            if len(runs) >= limit:
                return runs
    return runs


def diff_configs(config_a: str, config_b: str) -> Dict[str, Any]:
    """Compute a simple diff between two YAML config texts."""
    try:
        import yaml
        a = yaml.safe_load(config_a) if config_a else {}
        b = yaml.safe_load(config_b) if config_b else {}
    except Exception:
        return {}
    changes = {}
    keys = set(a.keys()) | set(b.keys())
    for k in keys:
        if a.get(k) != b.get(k):
            changes[k] = {"a": a.get(k), "b": b.get(k)}
    return changes


def load_run_details(run_path: str) -> Dict[str, Any]:
    """Load metrics/config/cv/figures for a specific run directory."""
    p = Path(run_path)
    details: Dict[str, Any] = {"run_path": run_path}
    metrics_path = p / "metrics.json"
    config_yaml = p / "config.yaml"
    config_json = p / "config.json"
    cv_path = p / "cv_results.csv"
    fig_path = p / "confusion_matrix.png"

    details["evidence"] = load_run_evidence(p)
    details["metrics"] = load_metrics_file(metrics_path)
    if config_yaml.exists():
        details["config"] = config_yaml.read_text()
        details["config_path"] = str(config_yaml)
    elif config_json.exists():
        details["config"] = config_json.read_text()
        details["config_path"] = str(config_json)
    else:
        details["config"] = None
    details["cv_results_path"] = str(cv_path) if cv_path.exists() else None
    if cv_path.exists():
        try:
            details["cv_results_df"] = pd.read_csv(cv_path)
        except Exception:
            details["cv_results_df"] = None
    details["figures"] = [str(fig_path)] if fig_path.exists() else []
    details["config_yaml"] = config_yaml.read_text() if config_yaml.exists() else None
    details["config_json"] = config_json.read_text() if config_json.exists() else None
    return details


# Evidence inspection deliberately reads JSON, never model.joblib or training code.
_EVIDENCE_LIMIT = 2 * 1024 * 1024


def _evidence_json(path: Path) -> dict:
    if not path.is_file():
        return {"status": "missing", "value": None}
    try:
        if path.stat().st_size > _EVIDENCE_LIMIT:
            raise ValueError("file exceeds 2 MiB evidence limit")
        def reject_constant(value):
            raise ValueError(f"nonfinite JSON number: {value}")
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return {"status": "recorded", "value": value}
    except (OSError, ValueError, UnicodeError) as exc:
        return {"status": "invalid", "value": None, "reason": str(exc)}


def _partition_evidence(value, mode):
    if not isinstance(value, dict):
        raise ValueError("partition must be an object")
    rows, ids, groups = (value.get(k) for k in ("row_positions", "sample_ids", "groups"))
    if not isinstance(rows, list) or not rows or any(type(x) is not int or x < 0 for x in rows):
        raise ValueError("nonempty nonnegative integer row_positions required")
    if len(set(rows)) != len(rows):
        raise ValueError("duplicate row positions")
    if not isinstance(ids, list) or len(ids) != len(rows):
        raise ValueError("sample_ids must align with row_positions")
    if any(type(x) not in (int, str) for x in ids):
        raise ValueError("sample_ids must be strings or integers")
    if mode == "none":
        if groups is not None:
            raise ValueError("row split groups must be null")
    elif not isinstance(groups, list) or len(groups) != len(rows) or any(type(x) not in (int, str) for x in groups):
        raise ValueError("groups must align with row_positions")
    return {"rows": len(rows), "groups": None if groups is None else len(set(groups))}


def _validate_split(value):
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported split schema_version")
    mode = value.get("group_mode")
    if mode not in ("none", "n", "nk"):
        raise ValueError("unsupported or missing group_mode")
    columns = value.get("feature_columns")
    if (not isinstance(columns, list) or not columns or
            any(not isinstance(c, str) or not c.startswith("feat_") for c in columns) or
            len(set(columns)) != len(columns)):
        raise ValueError("ordered unique feat_ columns required")
    if not isinstance(value.get("outer"), dict):
        raise ValueError("outer split required")
    outer = value["outer"]
    summary = {part: _partition_evidence(outer.get(part), mode) for part in ("train", "test")}
    train, test = outer["train"], outer["test"]
    if set(train["row_positions"]) & set(test["row_positions"]):
        raise ValueError("outer row overlap")
    if mode != "none" and set(train["groups"]) & set(test["groups"]):
        raise ValueError("outer group overlap")
    inner = value.get("inner")
    if not isinstance(inner, list) or len(inner) < 2:
        raise ValueError("at least two recorded inner folds required")
    original = {row: (sid, None if mode == "none" else train["groups"][i])
                for i, (row, sid) in enumerate(zip(train["row_positions"], train["sample_ids"]))}
    folds = []
    for fold in inner:
        if not isinstance(fold, dict):
            raise ValueError("inner fold must be an object")
        counts = {part: _partition_evidence(fold.get(part), mode) for part in ("train", "validation")}
        left, right = fold["train"], fold["validation"]
        if set(left["row_positions"]) & set(right["row_positions"]):
            raise ValueError("inner row overlap")
        if set(left["row_positions"]) | set(right["row_positions"]) != set(original):
            raise ValueError("inner fold does not partition outer training rows")
        if mode != "none" and set(left["groups"]) & set(right["groups"]):
            raise ValueError("inner group overlap")
        for part in (left, right):
            for i, (row, sid) in enumerate(zip(part["row_positions"], part["sample_ids"])):
                if original[row] != (sid, None if mode == "none" else part["groups"][i]):
                    raise ValueError("inner row identity/group disagrees with outer training")
        folds.append(counts)
    summary.update({"group_mode": mode, "feature_count": len(columns), "inner_folds": folds})
    return summary


def load_run_evidence(run_path: Path | str) -> dict:
    """Inspect recorded facts; missing/malformed artifacts never imply a match."""
    p = Path(run_path)
    config = _evidence_json(p / "config.json")
    if config["status"] == "missing" and (p / "config.yaml").is_file():
        try:
            import yaml
            path = p / "config.yaml"
            if path.stat().st_size > _EVIDENCE_LIMIT:
                raise ValueError("file exceeds 2 MiB evidence limit")
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("expected a YAML object")
            config = {"status": "recorded", "value": value}
        except (ImportError, OSError, ValueError, UnicodeError) as exc:
            config = {"status": "invalid", "value": None, "reason": str(exc)}
        except Exception as exc:  # optional YAML parser errors; never execute a fallback
            config = {"status": "invalid", "value": None, "reason": str(exc)}
    task = (config.get("value") or {}).get("task")
    if task not in ("task_a_view", "task_b_kbucket", "task_c_regress_n"):
        task = None
    signatures = []
    signature_problems = []
    metrics = _evidence_json(p / "metrics.json")
    for name in ("dataset_signature", "dataset_sig"):
        value = (metrics.get("value") or {}).get(name)
        if value is not None:
            if isinstance(value, str) and value.strip():
                signatures.append(value.strip())
            else:
                signature_problems.append("invalid metrics dataset signature")
    for name in ("dataset_signature.txt", "dataset_sig.txt", "dataset_signature.json"):
        path = p / name
        if not path.is_file():
            continue
        try:
            if path.stat().st_size > _EVIDENCE_LIMIT:
                raise ValueError("file exceeds 2 MiB evidence limit")
            if path.suffix == ".json":
                record = _evidence_json(path)
                if record["status"] != "recorded":
                    raise ValueError(record.get("reason", "invalid signature JSON"))
                value = record["value"].get("signature") or record["value"].get("dataset_signature")
            else:
                value = path.read_text(encoding="utf-8").strip()
            if not isinstance(value, str) or not value.strip():
                raise ValueError("nonempty signature string required")
            signatures.append(value.strip())
        except (OSError, ValueError, UnicodeError) as exc:
            signature_problems.append(str(exc))
    unique = sorted(set(signatures))
    identity = {"status": "missing", "value": None, "problems": signature_problems}
    if len(unique) > 1:
        identity["status"] = "conflicting"
    elif signature_problems:
        identity["status"] = "invalid"
    elif unique:
        identity.update({"status": "recorded", "value": unique[0]})
    split = _evidence_json(p / "split.json")
    summary = None
    if split["status"] == "recorded":
        try:
            summary = _validate_split(split["value"])
        except (ValueError, TypeError, KeyError) as exc:
            split = {"status": "invalid", "value": None, "reason": str(exc)}
    baseline = _evidence_json(p / "baseline.json")
    if baseline["status"] == "recorded":
        value = baseline["value"]
        import math
        valid_metrics = isinstance(value.get("metrics"), dict) and bool(value["metrics"]) and all(
            type(x) in (int, float) and math.isfinite(x) for x in value["metrics"].values())
        if (not isinstance(value.get("name"), str) or not value["name"] or
                not isinstance(value.get("prediction"), list) or not valid_metrics or
                (summary is not None and len(value["prediction"]) != summary["test"]["rows"])):
            baseline = {"status": "invalid", "value": None, "reason": "invalid baseline name, metrics or prediction length"}
    return {"task": task, "config_status": config["status"], "dataset_identity": identity,
            "split": split, "split_summary": summary, "baseline": baseline}


def compare_run_evidence(a: dict, b: dict) -> dict:
    """A fact-by-fact recorded comparison, not source authentication or gain."""
    facts = []
    def fact(name, left, right):
        status = "unverified" if left is None or right is None else ("match" if left == right else "mismatch")
        facts.append({"dimension": name, "status": status})
    fact("task", a.get("task"), b.get("task"))
    def identity(record):
        value = record.get("dataset_identity", {})
        return value.get("value") if value.get("status") == "recorded" else None
    fact("recorded_dataset_identity", identity(a), identity(b))
    def split(record):
        value = record.get("split", {})
        return value.get("value") if value.get("status") == "recorded" else None
    sa, sb = split(a), split(b)
    for name in ("feature_columns", "group_mode"):
        fact(name, sa.get(name) if sa else None, sb.get(name) if sb else None)
    for part in ("train", "test"):
        def membership(s):
            if not s:
                return None
            value = s["outer"][part]
            groups = value["groups"] or [None] * len(value["row_positions"])
            return sorted(zip(value["row_positions"], value["sample_ids"], groups))
        fact(f"outer_{part}_membership", membership(sa), membership(sb))
    states = {v["status"] for v in facts}
    status = "incompatible" if "mismatch" in states else ("unverified" if "unverified" in states else "matched_recorded_evidence")
    return {"status": status, "facts": facts,
            "scope": "Recorded facts only; matching reported identity is not authenticated provenance or evidence of model gain."}
