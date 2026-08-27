from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

# PDF support is optional at runtime.  The rest of the summary (CSV/JSON/Markdown)
# must still be generated even when ReportLab has not yet been installed into an
# existing virtual environment.  ``requirements.txt`` includes reportlab, so a
# normal ``pip install -r requirements.txt`` enables the PDF automatically.
try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    REPORTLAB_AVAILABLE = True
    REPORTLAB_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - environment dependent
    REPORTLAB_AVAILABLE = False
    REPORTLAB_IMPORT_ERROR = exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_config import load_params, resolve_project_paths  # noqa: E402


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _first_existing(paths: Iterable[Path]) -> Path | None:
    return next((p for p in paths if p.exists() and p.is_file()), None)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (list, dict, tuple, set)) else False:
        return None
    return value


def _hash_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _column(df: pd.DataFrame, names: list[str]) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _numeric(df: pd.DataFrame, names: list[str]) -> pd.Series:
    col = _column(df, names)
    if not col:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _task_model_summary(report_dir: Path, task: str) -> pd.DataFrame:
    candidates = (
        [
            # Prefer the post-training evaluation table because it contains
            # the complete model set and is produced immediately before the
            # final summary stage.
            report_dir / "all_evaluations.csv",
            report_dir / "best_model_per_featureset_selection.csv",
            report_dir / "selected_models_final_test_evaluation.csv",
            report_dir / "regression_summary.csv",
            report_dir / "Regrassion_summary.csv",
            report_dir / "regression_ranking.csv",
        ]
        if task == "regression"
        else [
            report_dir / "all_evaluations_class.csv",
            report_dir / "best_classifier_per_featureset_selection.csv",
            report_dir / "evaluate_class_summary.csv",
            report_dir / "selected_models_final_test_evaluation.csv",
            report_dir / "classification_summary.csv",
            report_dir / "classification_ranking.csv",
            report_dir / "ClassificationEnhancedFinalRanking.csv",
        ]
    )
    source = _first_existing(candidates)
    if source is None:
        return pd.DataFrame()
    df = _safe_read_csv(source)
    if df.empty:
        return df

    model_col = _column(df, ["model", "Model"])
    fs_col = _column(df, ["feature_set", "fs", "Feature_Set"])
    target_col = _column(df, ["target", "Target"])

    if task == "regression":
        cv = _numeric(df, ["cv_r2_mean", "nested_cv_r2_mean", "outer_cv_r2", "mean_outer_r2"])
        test = _numeric(df, ["r2_test", "test_r2", "Test_R2"])
        sd = _numeric(df, ["cv_r2_std", "cv_r2_sd", "nested_cv_r2_std", "outer_cv_r2_std"])
        gap = _numeric(df, ["train_cv_gap", "cv_test_gap"])
        metric = "R2"
    else:
        cv = _numeric(df, ["cv_macro_f1", "cv_macro_f1_mean", "nested_cv_macro_f1", "outer_cv_macro_f1"])
        test = _numeric(df, ["test_macro_f1", "macro_f1_test", "Test_Macro_F1"])
        sd = _numeric(df, ["cv_macro_f1_std", "cv_macro_f1_sd", "nested_cv_macro_f1_std"])
        gap = _numeric(df, ["train_cv_gap", "cv_test_gap", "val_test_gap"])
        metric = "Macro-F1"

    out = pd.DataFrame({
        "task": task,
        "target": df[target_col].astype(str) if target_col else "unspecified",
        "feature_set": df[fs_col].astype(str) if fs_col else "unspecified",
        "model": df[model_col].astype(str) if model_col else "unspecified",
        "selection_metric": metric,
        "cv_score": cv,
        "cv_sd": sd,
        "test_score": test,
        "generalization_gap": gap,
        "source_file": str(source),
    })
    # Keep one representative row per target/feature set where possible.
    sort_score = out["cv_score"].where(out["cv_score"].notna(), out["test_score"])
    out = out.assign(_sort=sort_score)
    out = out.sort_values(["target", "feature_set", "_sort"], ascending=[True, True, False])
    out = out.drop_duplicates(["target", "feature_set"], keep="first").drop(columns="_sort")
    return out.reset_index(drop=True)


def _nested_summary(report_dir: Path, task: str) -> pd.DataFrame:
    nested_root = report_dir / "nested_cv"
    if not nested_root.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for path in sorted(nested_root.rglob("*_nested_cv_folds.csv")):
        df = _safe_read_csv(path)
        if df.empty:
            continue
        model = path.stem.replace("_nested_cv_folds", "")
        feature_set = path.parent.name
        target = path.parent.parent.name if path.parent.parent != nested_root else "unspecified"
        if task == "regression":
            score = _numeric(df, ["outer_r2", "r2", "validation_r2", "test_r2", "score"])
            secondary = _numeric(df, ["outer_rmse", "rmse", "validation_rmse"])
            metric = "R2"
            secondary_name = "RMSE"
        else:
            score = _numeric(df, ["outer_macro_f1", "macro_f1", "validation_macro_f1", "f1_macro", "score"])
            secondary = _numeric(df, ["outer_balanced_accuracy", "balanced_accuracy"])
            metric = "Macro-F1"
            secondary_name = "Balanced accuracy"
        rows.append({
            "task": task,
            "target": target,
            "feature_set": feature_set,
            "model": model,
            "folds": int(len(df)),
            "metric": metric,
            "mean_score": float(score.mean()) if score.notna().any() else np.nan,
            "sd_score": float(score.std(ddof=1)) if score.notna().sum() > 1 else np.nan,
            "secondary_metric": secondary_name,
            "secondary_mean": float(secondary.mean()) if secondary.notna().any() else np.nan,
            "source_file": str(path),
        })
    return pd.DataFrame(rows)


def _learning_summary(report_dir: Path, task: str) -> pd.DataFrame:
    root = report_dir / "learning_curves"
    preferred = root / "learning_curve_all_models_summary.csv"
    frames: list[pd.DataFrame] = []
    if preferred.exists():
        df = _safe_read_csv(preferred)
        if not df.empty:
            frames.append(df)
    else:
        for path in sorted(root.rglob("learning_curve_model_summary.csv")) if root.exists() else []:
            df = _safe_read_csv(path)
            if not df.empty:
                df["source_file"] = str(path)
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    if "task" not in df.columns:
        df.insert(0, "task", task)
    else:
        df["task"] = task
    if "source_file" not in df.columns:
        df["source_file"] = str(preferred)
    return df


def _permutation_summary(report_dir: Path, task: str) -> pd.DataFrame:
    root = report_dir / "permutation_sensitivity"
    if not root.exists():
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*_permutation_sensitivity.csv")):
        df = _safe_read_csv(path)
        if df.empty:
            continue
        feature_col = _column(df, ["feature", "Feature", "variable"])
        mean_col = _column(df, ["mean_importance", "importance_mean", "mean_permutation_importance"])
        rel_col = _column(df, ["relative_contribution_percent", "relative_importance_percent", "contribution_percent"])
        if not feature_col or not mean_col:
            continue
        work = df.copy()
        work[mean_col] = pd.to_numeric(work[mean_col], errors="coerce")
        work["_abs"] = work[mean_col].abs()
        work = work.sort_values("_abs", ascending=False).head(5)
        model = path.stem.replace("_permutation_sensitivity", "")
        feature_set = path.parent.name
        target = path.parent.parent.name if path.parent.parent != root else "unspecified"
        for rank, (_, row) in enumerate(work.iterrows(), start=1):
            records.append({
                "task": task,
                "target": target,
                "feature_set": feature_set,
                "model": model,
                "rank": rank,
                "feature": row[feature_col],
                "mean_importance": row[mean_col],
                "relative_contribution_percent": row[rel_col] if rel_col else np.nan,
                "source_file": str(path),
            })
    return pd.DataFrame(records)


def _analysis_inventory(paths: dict[str, Path], params: dict[str, Any]) -> pd.DataFrame:
    checks = [
        ("Data quality", paths["dq_report"], "*.csv"),
        ("Multicollinearity", paths["multicollinearity_report"], "*.csv"),
        ("Regression nested CV", paths["reports_regression"] / "nested_cv", "*_nested_cv_folds.csv"),
        ("Regression learning curves", paths["reports_regression"] / "learning_curves", "*_learning_curve.csv"),
        ("Regression permutation", paths["reports_regression"] / "permutation_sensitivity", "*_permutation_sensitivity.csv"),
        ("Classification nested CV", paths["reports_classification"] / "nested_cv", "*_nested_cv_folds.csv"),
        ("Classification learning curves", paths["reports_classification"] / "learning_curves", "*_learning_curve.csv"),
        ("Classification permutation", paths["reports_classification"] / "permutation_sensitivity", "*_permutation_sensitivity.csv"),
    ]
    rows = []
    for name, folder, pattern in checks:
        files = list(folder.rglob(pattern)) if folder.exists() else []
        rows.append({
            "analysis": name,
            "status": "Available" if files else "Not performed / not available",
            "files_found": len(files),
            "location": str(folder),
        })
    return pd.DataFrame(rows)


def _package_versions() -> dict[str, str]:
    packages = [
        "numpy", "pandas", "scipy", "scikit-learn", "xgboost", "statsmodels",
        "matplotlib", "joblib", "PyYAML", "reportlab",
    ]
    result = {}
    for pkg in packages:
        try:
            result[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            result[pkg] = "not installed"
    return result


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def _df_to_table(df: pd.DataFrame, columns: list[tuple[str, str]], max_rows: int = 18) -> Table:
    data = [[label for _, label in columns]]
    for _, row in df.head(max_rows).iterrows():
        current = []
        for key, _ in columns:
            val = row.get(key, "")
            if isinstance(val, float):
                val = _fmt(val)
            text = str(val)
            current.append(text if len(text) <= 36 else text[:33] + "...")
        data.append(current)
    table = Table(data, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EDF3")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.2),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def _markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    """Return a lightweight Markdown table without requiring optional packages."""
    if df is None or df.empty:
        return "_Not performed / not available._\n"
    view = df.head(max_rows).copy()
    # Keep cells compact and single-line for a readable fallback report.
    for col in view.columns:
        view[col] = view[col].map(lambda v: str(_json_safe(v)).replace("|", "\\|").replace("\n", " "))
    header = "| " + " | ".join(map(str, view.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in view.itertuples(index=False, name=None)]
    suffix = "\n\n_Only the first %d rows are shown here; see the CSV files for the complete table._" % max_rows if len(df) > max_rows else ""
    return "\n".join([header, sep, *rows]) + suffix + "\n"


def _build_markdown(
    output: Path,
    params: dict[str, Any],
    paths: dict[str, Path],
    model_summary: pd.DataFrame,
    nested: pd.DataFrame,
    learning: pd.DataFrame,
    permutation: pd.DataFrame,
    inventory: pd.DataFrame,
    provenance: dict[str, Any],
    pdf_status: str,
) -> None:
    """Write a dependency-light human-readable report that always succeeds."""
    split = params.get("split", {}) or {}
    classification = params.get("classification", {}) or {}
    project = params.get("project", {}) or {}
    targets = ", ".join((params.get("TARGETS", {}) or {}).keys()) or "-"
    lines = [
        f"# {project.get('name', 'MLOps Research Study')}",
        "",
        "## Automated Research Summary",
        "",
        f"- Generated: {provenance['generated_at']}",
        f"- Dataset: `{provenance.get('dataset', '-')}`",
        f"- Dataset SHA256: `{provenance.get('dataset_sha256') or '-'}`",
        f"- Active params: `{provenance.get('params_file', '-')}`",
        f"- Targets: {targets}",
        f"- Grouping enabled: {bool(split.get('grouping_enabled', False))}",
        f"- Group column: {split.get('group_column') or '-'}",
        f"- Train/test: {split.get('train', '-')}/{split.get('test', '-')}",
        f"- Classification enabled: {bool(classification.get('enabled', False))}",
        f"- PDF status: {pdf_status}",
        "",
        "## Analysis availability",
        "",
        _markdown_table(inventory),
        "## Selected / representative model results",
        "",
        _markdown_table(model_summary),
        "## Nested cross-validation",
        "",
        "Nested CV summarizes saved outer-fold generalization estimates; hyperparameter optimization remains inside the inner CV loop.",
        "",
        _markdown_table(nested),
        "## Learning-curve analysis",
        "",
        _markdown_table(learning),
        "## Permutation sensitivity / feature importance",
        "",
        "Permutation results quantify predictive dependence on each feature and are not causal effects.",
        "",
        _markdown_table(permutation),
        "## Reproducibility",
        "",
        f"- Python: `{provenance['python']}`",
        f"- Platform: `{provenance['platform']}`",
        f"- Regression reports: `{paths['reports_regression']}`",
        f"- Classification reports: `{paths['reports_classification']}`",
        "",
        "### Package versions",
        "",
    ]
    for name, version in provenance["package_versions"].items():
        lines.append(f"- {name}: {version}")
    lines += [
        "",
        "## Interpretation note",
        "",
        "The independent test partition is final holdout evidence and must not be used as a tuning criterion. Nested CV, learning curves, and permutation sensitivity answer different questions and should not be collapsed into one score.",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def _build_pdf(
    output: Path,
    params: dict[str, Any],
    paths: dict[str, Path],
    model_summary: pd.DataFrame,
    nested: pd.DataFrame,
    learning: pd.DataFrame,
    permutation: pd.DataFrame,
    inventory: pd.DataFrame,
    provenance: dict[str, Any],
) -> None:
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError(f"ReportLab is unavailable: {REPORTLAB_IMPORT_ERROR}")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, leading=22))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, leading=11))
    styles.add(ParagraphStyle(name="H2Compact", parent=styles["Heading2"], spaceBefore=8, spaceAfter=5))

    doc = SimpleDocTemplate(
        str(output), pagesize=landscape(A4), rightMargin=12 * mm, leftMargin=12 * mm,
        topMargin=12 * mm, bottomMargin=12 * mm, title="MLOps Research Summary",
    )
    story = []
    project = params.get("project", {}) or {}
    story += [
        Paragraph(str(project.get("name", "MLOps Research Study")), styles["TitleCenter"]),
        Paragraph("Automated Research Summary Report", styles["Heading2"]),
        Paragraph(
            "This compact report collects the principal configuration, validation, generalization, learning-curve, and permutation-sensitivity outputs produced by the current MLOps run. Detailed CSV/JSON files remain the authoritative source for full results.",
            styles["Small"],
        ),
        Spacer(1, 6),
    ]

    split = params.get("split", {}) or {}
    classification = params.get("classification", {}) or {}
    overview = pd.DataFrame([
        ["Generated", provenance["generated_at"]],
        ["Dataset", provenance.get("dataset", "-")],
        ["Dataset SHA256", provenance.get("dataset_sha256", "-") or "-"],
        ["Targets", ", ".join((params.get("TARGETS", {}) or {}).keys()) or "-"],
        ["Grouping enabled", str(bool(split.get("grouping_enabled", False)))],
        ["Group column", str(split.get("group_column") or "-")],
        ["Train / test", f"{split.get('train', '-')}/{split.get('test', '-')}"],
        ["CV splits / repeats", f"{split.get('cv_splits', params.get('cv_splits', '-'))}/{split.get('cv_repeats', '-')}"],
        ["Classification enabled", str(bool(classification.get("enabled", False)))],
        ["Class boundaries", json.dumps(classification.get("class_boundaries", {}), ensure_ascii=True)],
    ], columns=["Setting", "Value"])
    story.append(_df_to_table(overview, [("Setting", "Setting"), ("Value", "Value")], 20))

    story += [Spacer(1, 8), Paragraph("Analysis availability", styles["H2Compact"])]
    story.append(_df_to_table(inventory, [("analysis", "Analysis"), ("status", "Status"), ("files_found", "Files")], 30))

    story += [Spacer(1, 8), Paragraph("Selected / representative model results", styles["H2Compact"])]
    if model_summary.empty:
        story.append(Paragraph("Not performed / not available.", styles["Small"]))
    else:
        story.append(_df_to_table(model_summary, [
            ("task", "Task"), ("target", "Target"), ("feature_set", "Feature set"), ("model", "Model"),
            ("selection_metric", "Metric"), ("cv_score", "CV"), ("cv_sd", "CV SD"), ("test_score", "Test"),
            ("generalization_gap", "Gap"),
        ], 24))

    story += [PageBreak(), Paragraph("Nested cross-validation", styles["Heading1"])]
    story.append(Paragraph(
        "Nested CV is summarized from the saved outer-fold files. Hyperparameter optimization remains inside the inner CV loop in the training programs; the table below reports the outer-fold generalization estimates when available.",
        styles["Small"],
    ))
    if nested.empty:
        story.append(Paragraph("Not performed / not available.", styles["Small"]))
    else:
        best_nested = nested.sort_values("mean_score", ascending=False).groupby(["task", "target", "feature_set"], as_index=False).head(1)
        story.append(_df_to_table(best_nested, [
            ("task", "Task"), ("target", "Target"), ("feature_set", "Feature set"), ("model", "Model"),
            ("folds", "Outer folds"), ("metric", "Metric"), ("mean_score", "Mean"), ("sd_score", "SD"),
            ("secondary_metric", "Secondary"), ("secondary_mean", "Value"),
        ], 28))

    story += [Spacer(1, 10), Paragraph("Learning-curve analysis", styles["Heading1"])]
    if learning.empty:
        story.append(Paragraph("Not performed / not available.", styles["Small"]))
    else:
        # Keep generic because regression/classification summary schemas differ slightly.
        preferred = [
            ("task", "Task"), ("target", "Target"), ("feature_set", "Feature set"), ("model", "Model"),
            ("final_cv_score", "Final CV"), ("final_cv_r2", "Final CV R2"), ("final_cv_macro_f1", "Final CV F1"),
            ("final_train_cv_gap", "Final gap"), ("validation_trend", "Trend"),
            ("more_data_likely_helpful", "More-data signal"), ("interpretation", "Interpretation"),
        ]
        cols = [(c, label) for c, label in preferred if c in learning.columns]
        if len(cols) < 3:
            cols = [(str(c), str(c).replace("_", " ").title()) for c in list(learning.columns)[:9]]
        story.append(_df_to_table(learning, cols, 24))

    story += [PageBreak(), Paragraph("Permutation sensitivity / feature importance", styles["Heading1"])]
    story.append(Paragraph(
        "Permutation sensitivity is computed by perturbing one feature at a time and measuring the decrease in the configured predictive score. Large positive magnitudes indicate greater dependence of the fitted model on that predictor; negative values can occur and should not be interpreted as beneficial causal effects.",
        styles["Small"],
    ))
    if permutation.empty:
        story.append(Paragraph("Not performed / not available.", styles["Small"]))
    else:
        top = permutation[permutation["rank"] <= 3].copy()
        story.append(_df_to_table(top, [
            ("task", "Task"), ("target", "Target"), ("feature_set", "Feature set"), ("model", "Model"),
            ("rank", "Rank"), ("feature", "Feature"), ("mean_importance", "Mean importance"),
            ("relative_contribution_percent", "Relative %"),
        ], 36))

    story += [Spacer(1, 10), Paragraph("Reproducibility record", styles["Heading1"])]
    repro = pd.DataFrame([
        ["Python", provenance["python"]],
        ["Platform", provenance["platform"]],
        ["Active params", provenance["params_file"]],
        ["Random state", params.get("random_state", split.get("random_state", "-"))],
        ["Regression reports", paths["reports_regression"]],
        ["Classification reports", paths["reports_classification"]],
    ] + [[f"pkg:{k}", v] for k, v in provenance["package_versions"].items()], columns=["Item", "Value"])
    story.append(_df_to_table(repro, [("Item", "Item"), ("Value", "Value")], 30))

    story += [Spacer(1, 10), Paragraph("Interpretation note", styles["Heading2"])]
    story.append(Paragraph(
        "The independent test partition should be interpreted as final holdout evidence and not as a tuning criterion. Nested CV, repeated/group-aware CV, learning curves, and permutation sensitivity answer different questions and should be considered jointly rather than collapsed into a single performance number.",
        styles["Small"],
    ))
    doc.build(story)


def generate(params_path: Path, output_dir: Path | None = None) -> dict[str, Path]:
    params_path = params_path.resolve()
    params = load_params(params_path)
    paths = resolve_project_paths(ROOT, params)
    summary_cfg = params.get("summary_report", {}) or {}
    if output_dir is None:
        configured = (params.get("paths", {}) or {}).get("summary_report", "summary_report")
        candidate = Path(str(configured)).expanduser()
        output_dir = candidate if candidate.is_absolute() else (ROOT / candidate)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reg_models = _task_model_summary(paths["reports_regression"], "regression")
    cls_models = _task_model_summary(paths["reports_classification"], "classification") if (params.get("classification", {}) or {}).get("enabled", False) else pd.DataFrame()
    model_summary = pd.concat([reg_models, cls_models], ignore_index=True, sort=False) if not reg_models.empty or not cls_models.empty else pd.DataFrame()

    reg_nested = _nested_summary(paths["reports_regression"], "regression")
    cls_nested = _nested_summary(paths["reports_classification"], "classification")
    nested = pd.concat([reg_nested, cls_nested], ignore_index=True, sort=False) if not reg_nested.empty or not cls_nested.empty else pd.DataFrame()

    reg_learning = _learning_summary(paths["reports_regression"], "regression")
    cls_learning = _learning_summary(paths["reports_classification"], "classification")
    learning = pd.concat([reg_learning, cls_learning], ignore_index=True, sort=False) if not reg_learning.empty or not cls_learning.empty else pd.DataFrame()

    reg_perm = _permutation_summary(paths["reports_regression"], "regression")
    cls_perm = _permutation_summary(paths["reports_classification"], "classification")
    permutation = pd.concat([reg_perm, cls_perm], ignore_index=True, sort=False) if not reg_perm.empty or not cls_perm.empty else pd.DataFrame()

    inventory = _analysis_inventory(paths, params)
    input_csv = paths["input_csv"]
    provenance = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "dataset": str(input_csv),
        "dataset_sha256": _hash_file(input_csv),
        "params_file": str(params_path),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "package_versions": _package_versions(),
    }

    outputs = {
        "pdf": output_dir / str(summary_cfg.get("pdf_filename", "MLOps_Research_Summary.pdf")),
        "json": output_dir / "MLOps_Research_Summary.json",
        "markdown": output_dir / "MLOps_Research_Summary.md",
        "model_csv": output_dir / "model_summary.csv",
        "nested_csv": output_dir / "nested_cv_summary.csv",
        "learning_csv": output_dir / "learning_curve_summary.csv",
        "permutation_csv": output_dir / "permutation_summary.csv",
        "inventory_csv": output_dir / "analysis_inventory.csv",
    }

    model_summary.to_csv(outputs["model_csv"], index=False, encoding="utf-8-sig")
    nested.to_csv(outputs["nested_csv"], index=False, encoding="utf-8-sig")
    learning.to_csv(outputs["learning_csv"], index=False, encoding="utf-8-sig")
    permutation.to_csv(outputs["permutation_csv"], index=False, encoding="utf-8-sig")
    inventory.to_csv(outputs["inventory_csv"], index=False, encoding="utf-8-sig")

    payload = {
        "project": params.get("project", {}),
        "provenance": provenance,
        "analysis_inventory": inventory.to_dict(orient="records"),
        "model_summary": model_summary.to_dict(orient="records"),
        "nested_cv_summary": nested.to_dict(orient="records"),
        "learning_curve_summary": learning.to_dict(orient="records"),
        "permutation_summary": permutation.to_dict(orient="records"),
    }
    outputs["json"].write_text(json.dumps(payload, indent=2, default=_json_safe), encoding="utf-8")

    pdf_requested = bool(summary_cfg.get("pdf", True))
    pdf_status = "disabled by configuration"
    if pdf_requested:
        if REPORTLAB_AVAILABLE:
            try:
                _build_pdf(outputs["pdf"], params, paths, model_summary, nested, learning, permutation, inventory, provenance)
                pdf_status = "generated successfully"
            except Exception as exc:
                pdf_status = f"skipped after PDF-generation error: {exc}"
                print(f"[SUMMARY WARNING] PDF could not be generated: {exc}", file=sys.stderr)
        else:
            pdf_status = "skipped - ReportLab is not installed in this Python environment"
            print(
                "[SUMMARY WARNING] PDF skipped because ReportLab is unavailable. "
                "Install it with: python -m pip install reportlab  "
                "(or rerun: python -m pip install -r requirements.txt)",
                file=sys.stderr,
            )

    _build_markdown(
        outputs["markdown"], params, paths, model_summary, nested, learning,
        permutation, inventory, provenance, pdf_status
    )

    # Record the PDF status in JSON as well, so downstream automation knows
    # whether the human-readable PDF exists without treating it as a failed run.
    payload["summary_report"] = {"pdf_requested": pdf_requested, "pdf_status": pdf_status}
    outputs["json"].write_text(json.dumps(payload, indent=2, default=_json_safe), encoding="utf-8")

    print("[SUMMARY] Research summary generated:")
    for key, path in outputs.items():
        if path.exists():
            print(f"  {key:16s} {path}")
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a compact MLOps research summary report.")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    params_path = Path(args.params).expanduser()
    if not params_path.is_absolute():
        params_path = ROOT / params_path
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else None
    if output_dir is not None and not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    generate(params_path, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
