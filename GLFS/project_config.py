"""Shared configuration helpers for the generic Geo/Engineering MLOps project.

The repository uses ``params.yaml`` as the *active* configuration so all legacy
programs can keep using ``--params params.yaml``.  Any number of alternative
YAML profiles may live beside it and can be activated by the GUI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import shutil
import yaml


DEFAULT_PATHS = {
    "processed_data": "data/processed",
    "reports_regression": "regression_report",
    "reports_classification": "classification_report",
    "models_regression": "regression_models",
    "models_classification": "classification_models",
    "dq_report": "data_processing_report/dq_report",
    "multicollinearity_report": "data_processing_report/multicollinearity",
    "summary_report": "summary_report",
}


def load_params(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Parameter file must contain a YAML mapping: {path}")
    return loaded


def save_params(path: str | Path, params: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(params, handle, sort_keys=False, allow_unicode=True)


def _path(root: Path, value: Any, fallback: str) -> Path:
    text = str(value).strip() if value is not None else ""
    candidate = Path(text or fallback).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def portable_path(root: str | Path, value: str | Path) -> str:
    """Store project-local directories as relative paths when possible."""
    root = Path(root).resolve()
    path = Path(value).expanduser().resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def resolve_project_paths(root: str | Path, params: dict[str, Any]) -> dict[str, Path]:
    root = Path(root).resolve()
    data_cfg = params.get("data", {}) or {}
    paths_cfg = params.get("paths", {}) or {}

    raw_csv = _path(root, data_cfg.get("input_csv"), "data/raw/samples.csv")
    processed = _path(root, paths_cfg.get("processed_data"), DEFAULT_PATHS["processed_data"])
    reports_reg = _path(root, paths_cfg.get("reports_regression"), DEFAULT_PATHS["reports_regression"])
    reports_cls = _path(root, paths_cfg.get("reports_classification"), DEFAULT_PATHS["reports_classification"])
    models_reg = _path(root, paths_cfg.get("models_regression"), DEFAULT_PATHS["models_regression"])
    models_cls = _path(root, paths_cfg.get("models_classification"), DEFAULT_PATHS["models_classification"])

    return {
        "input_csv": raw_csv,
        "processed_data": processed,
        "reports_regression": reports_reg,
        "reports_classification": reports_cls,
        "models_regression": models_reg,
        "models_classification": models_cls,
        "dq_report": _path(root, paths_cfg.get("dq_report"), DEFAULT_PATHS["dq_report"]),
        "multicollinearity_report": _path(
            root, paths_cfg.get("multicollinearity_report"), DEFAULT_PATHS["multicollinearity_report"]
        ),
        "summary_report": _path(root, paths_cfg.get("summary_report"), DEFAULT_PATHS["summary_report"]),
    }



def cleanup_legacy_report_model_dirs(paths: dict[str, Path]) -> list[Path]:
    """Remove obsolete *empty* model/models folders nested under report directories.

    Models are intentionally stored only in models_regression/models_classification.
    This function is conservative: it removes a legacy directory only when it is
    completely empty, so no trained artifact can be deleted accidentally.
    """
    removed: list[Path] = []
    for key in ("reports_regression", "reports_classification"):
        report_dir = paths.get(key)
        if not report_dir:
            continue
        for name in ("model", "models"):
            candidate = Path(report_dir) / name
            try:
                if candidate.is_dir() and not any(candidate.iterdir()):
                    candidate.rmdir()
                    removed.append(candidate)
            except OSError:
                # Never make cleanup a reason for the pipeline to fail.
                pass
    return removed

def update_paths_in_params(root: str | Path, params_path: str | Path, selected: dict[str, str | Path]) -> dict[str, Any]:
    """Write confirmed output directories into the active YAML configuration."""
    params = load_params(params_path)
    cfg = params.setdefault("paths", {})
    if not isinstance(cfg, dict):
        cfg = {}
        params["paths"] = cfg
    for key, value in selected.items():
        if key in DEFAULT_PATHS and str(value).strip():
            cfg[key] = portable_path(root, value)
    save_params(params_path, params)
    return params


def activate_params_profile(root: str | Path, source: str | Path, active_name: str = "params.yaml") -> Path:
    """Copy a selected YAML profile to the canonical active params.yaml."""
    root = Path(root).resolve()
    source = Path(source).expanduser().resolve()
    target = root / active_name
    if source != target.resolve():
        shutil.copy2(source, target)
    return target


def classification_targets(params: dict[str, Any]) -> list[str]:
    """Return active classification targets. Empty/omitted means every TARGETS key."""
    targets = list((params.get("TARGETS", {}) or {}).keys())
    cfg = params.get("classification", {}) or {}
    requested = cfg.get("targets")
    if requested in (None, [], "all", "ALL", "*"):
        return targets
    if isinstance(requested, str):
        requested = [requested]
    requested_set = {str(x) for x in requested}
    return [name for name in targets if name in requested_set]


def write_install_txt(root: str | Path, params_path: str | Path, output_file: str | Path | None = None) -> Path:
    """Generate install.txt with commands matching the active YAML paths.

    This keeps the familiar explicit program arguments while ensuring that the
    command examples always match directories selected in the GUI.
    """
    root = Path(root).resolve()
    params_path = Path(params_path).resolve()
    params = load_params(params_path)
    paths = resolve_project_paths(root, params)
    output = Path(output_file) if output_file else root / "install.txt"
    if not output.is_absolute():
        output = root / output

    def q(path: Path) -> str:
        try:
            return path.resolve().relative_to(root).as_posix()
        except ValueError:
            return f'"{path}"'

    p = q(params_path)
    input_csv = q(paths["input_csv"])
    processed = q(paths["processed_data"])
    reg_models = q(paths["models_regression"])
    cls_models = q(paths["models_classification"])
    reg_reports = q(paths["reports_regression"])
    cls_reports = q(paths["reports_classification"])
    dq = q(paths["dq_report"])
    multi = q(paths["multicollinearity_report"])
    summary = q(paths["summary_report"])

    lines = [
        "# Generic Geo/Engineering MLOps setup",
        "# This file is regenerated by the GUI after output paths are confirmed.",
        "python -m venv .venv",
        "# Windows PowerShell: ./.venv/Scripts/Activate.ps1",
        "# Windows cmd.exe:    .venv/Scripts/activate.bat",
        "# Linux/macOS:        source .venv/bin/activate",
        "python -m pip install --upgrade pip",
        "pip install -r requirements.txt",
        "",
        "# Active configuration",
        f"# params: {p}",
        "",
        "# Full configurable pipeline",
        f"python run_pipeline.py --params {p}",
        "",
        "# Individual programs with explicit arguments",
        f"python src/data_quality.py --data {input_csv} --outdir {dq} --params {p}",
        f"python src/Multicollinearity.py --data {input_csv} --out {multi} --params {p}",
        f"python src/features.py --input {input_csv} --out {processed} --params {p}",
        f"python src/train.py --data {processed} --models_dir {reg_models} --reports_dir {reg_reports} --params {p}",
        f"python src/evaluate_all.py --data {processed} --models_dir {reg_models} --reports_dir {reg_reports} --params {p}",
        f"python src/best_model_of_each_set_all_evaluation.py --input {reg_reports}/all_evaluations.csv --output {reg_reports}",
        f"python src/trainclass.py --data {processed} --models_dir {cls_models} --reports_dir {cls_reports} --params {p}",
        f"python -m src.evaluate_class --data {processed} --models_dir {cls_models} --reports_dir {cls_reports} --params {p}",
        f"python -m src.evaluate_all_class --data {processed} --models_dir {cls_models} --reports_dir {cls_reports} --params {p}",
        f"python src/best_model_of_each_set_all_evaluation_class.py --input {cls_reports}/all_evaluations_class.csv --output {cls_reports}",
        "",
        "# Final aggregation: run only after all enabled evaluations above",
        f"python src/generate_summary_report.py --params {p} --output_dir {summary}",
        "",
        "# Optional interactive prediction GUIs (not part of the automatic pipeline)",
        f"python src/predict_Reg.py --gui --data {processed} --models_dir {reg_models} --reports_dir {reg_reports}",
        f"python src/predict_class.py --gui --data {processed} --models_dir {cls_models} --reports_dir {cls_reports} --params {p}",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
