"""Config-driven pipeline launcher used by soil_mlops_gui.py.

Study-specific paths and optional stages are controlled from params.yaml.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from project_config import load_params, resolve_project_paths, cleanup_legacy_report_model_dirs


def run(command: list[str], *, cwd: Path, required: bool = True) -> bool:
    print("\n[PIPELINE]", subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        if required:
            raise subprocess.CalledProcessError(completed.returncode, command)
        print(
            f"[PIPELINE WARNING] Optional stage exited with code {completed.returncode}; "
            "the completed training outputs are preserved.",
            flush=True,
        )
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the configurable MLOps pipeline.")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--data", default=None, help="Optional CSV override.")
    parser.add_argument(
        "--tasks", nargs="+", choices=["regression", "classification"], default=None,
        help="Optional per-run task override. If omitted, pipeline.regression/classification from params.yaml are used.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    params_path = Path(args.params).expanduser()
    if not params_path.is_absolute():
        params_path = root / params_path
    params = load_params(params_path)
    paths = resolve_project_paths(root, params)
    removed_legacy = cleanup_legacy_report_model_dirs(paths)
    for legacy_dir in removed_legacy:
        print(f"[CONFIG] Removed obsolete empty report-model directory: {legacy_dir}", flush=True)
    input_csv = Path(args.data).expanduser() if args.data else paths["input_csv"]
    if not input_csv.is_absolute():
        input_csv = root / input_csv

    if not input_csv.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_csv}")

    py = sys.executable
    stages = params.get("pipeline", {}) or {}
    selected_tasks = set(args.tasks or [])

    def enabled(name: str, default: bool = True) -> bool:
        if name in {"regression", "classification"} and args.tasks is not None:
            return name in selected_tasks
        return bool(stages.get(name, default))

    paths["processed_data"].mkdir(parents=True, exist_ok=True)
    if enabled("regression"):
        paths["reports_regression"].mkdir(parents=True, exist_ok=True)
        paths["models_regression"].mkdir(parents=True, exist_ok=True)

    classification_cfg = params.get("classification", {}) or {}
    if enabled("classification"):
        if not bool(classification_cfg.get("enabled", False)):
            if args.tasks is not None and "classification" in selected_tasks:
                raise ValueError(
                    "Classification was selected for this run, but classification.enabled is false in params.yaml."
                )
        else:
            paths["reports_classification"].mkdir(parents=True, exist_ok=True)
            paths["models_classification"].mkdir(parents=True, exist_ok=True)

    if enabled("summary_report", True):
        paths["summary_report"].mkdir(parents=True, exist_ok=True)

    if enabled("data_quality"):
        run([py, "src/data_quality.py", "--data", str(input_csv), "--outdir", str(paths["dq_report"]), "--params", str(params_path)], cwd=root)
    if enabled("multicollinearity"):
        run([py, "src/Multicollinearity.py", "--data", str(input_csv), "--out", str(paths["multicollinearity_report"]), "--params", str(params_path)], cwd=root)
    if enabled("features"):
        run([py, "src/features.py", "--input", str(input_csv), "--out", str(paths["processed_data"]), "--params", str(params_path)], cwd=root)
    # ------------------------------------------------------------------
    # Training and post-training evaluation
    # ------------------------------------------------------------------
    # The research summary must be generated *after* these stages because
    # it aggregates all_evaluations*, best-per-feature-set rankings, nested
    # CV, learning curves, permutation analyses, and final holdout results.
    if enabled("regression"):
        run([py, "src/train.py", "--data", str(paths["processed_data"]), "--models_dir", str(paths["models_regression"]), "--reports_dir", str(paths["reports_regression"]), "--params", str(params_path)], cwd=root)

        if enabled("regression_evaluation", True):
            run([
                py, "src/evaluate_all.py",
                "--data", str(paths["processed_data"]),
                "--models_dir", str(paths["models_regression"]),
                "--reports_dir", str(paths["reports_regression"]),
                "--params", str(params_path),
            ], cwd=root)
            run([
                py, "src/best_model_of_each_set_all_evaluation.py",
                "--input", str(paths["reports_regression"] / "all_evaluations.csv"),
                "--output", str(paths["reports_regression"]),
            ], cwd=root)

    if enabled("classification") and bool(classification_cfg.get("enabled", False)):
        run([py, "src/trainclass.py", "--data", str(paths["processed_data"]), "--models_dir", str(paths["models_classification"]), "--reports_dir", str(paths["reports_classification"]), "--params", str(params_path)], cwd=root)

        if enabled("classification_evaluation", True):
            # Evaluate the selected classifier first, then all ordinary
            # classifier packages. Both outputs are useful to the final report.
            run([
                py, "-m", "src.evaluate_class",
                "--data", str(paths["processed_data"]),
                "--models_dir", str(paths["models_classification"]),
                "--reports_dir", str(paths["reports_classification"]),
                "--params", str(params_path),
            ], cwd=root)
            run([
                py, "-m", "src.evaluate_all_class",
                "--data", str(paths["processed_data"]),
                "--models_dir", str(paths["models_classification"]),
                "--reports_dir", str(paths["reports_classification"]),
                "--params", str(params_path),
            ], cwd=root)
            run([
                py, "src/best_model_of_each_set_all_evaluation_class.py",
                "--input", str(paths["reports_classification"] / "all_evaluations_class.csv"),
                "--output", str(paths["reports_classification"]),
            ], cwd=root)

    # IMPORTANT: summary_report is deliberately the final non-interactive
    # stage. Prediction GUIs are user-driven tools and are not launched by
    # the automated training pipeline because they would block execution.
    summary_cfg = params.get("summary_report", {}) or {}
    if enabled("summary_report", True) and bool(summary_cfg.get("enabled", True)):
        summary_required = bool(summary_cfg.get("fail_pipeline_on_error", False))
        run(
            [py, "src/generate_summary_report.py", "--params", str(params_path), "--output_dir", str(paths["summary_report"])],
            cwd=root,
            required=summary_required,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
