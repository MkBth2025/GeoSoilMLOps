from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from evaluate import _load_params, evaluate_model_file, add_ranks


def run(
    data_dir: str,
    models_dir: str,
    reports_dir: str,
    params_path: str = "params.yaml",
    out_name: str = "all_evaluations",
    skip_best: bool = True,
) -> None:
    data_dir, models_dir, reports_dir = Path(data_dir), Path(models_dir), Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    params = _load_params(params_path)

    model_paths = sorted(models_dir.glob("*.pkl"))
    if skip_best:
        model_paths = [p for p in model_paths if not p.name.startswith("best")]

    rows, predictions, failures = [], [], []
    for i, model_path in enumerate(model_paths, start=1):
        print(f"[EVAL_ALL] {i}/{len(model_paths)}: {model_path.name}")
        try:
            row, model_predictions = evaluate_model_file(model_path, data_dir, params)
            rows.append(row)
            predictions.extend(model_predictions)
        except Exception as error:
            failures.append({"file": model_path.name, "error": str(error)})
            print(f"[EVAL_ALL][WARN] {model_path.name}: {error}")

    if not rows:
        raise RuntimeError("No models were evaluated successfully.")

    df = add_ranks(pd.DataFrame(rows))
    df.to_csv(reports_dir / f"{out_name}.csv", index=False)
    df.to_json(reports_dir / f"{out_name}.json", orient="records", indent=2)
    df.to_csv(reports_dir / "regression_evaluation_ranking.csv", index=False)
    pd.DataFrame(predictions).to_csv(reports_dir / "predictions_long.csv", index=False)
    pd.DataFrame(failures).to_csv(reports_dir / "evaluation_failures.csv", index=False)

    paper_columns = [
        "Selection Rank", "Test Performance Rank", "target", "model", "feature_set",
        "r2_train", "cv_r2_mean", "cv_r2_std", "r2_test",
        "train_cv_gap", "cv_test_gap", "rmse_test", "generalization_diagnosis",
    ]
    df[[c for c in paper_columns if c in df.columns]].to_csv(
        reports_dir / "Results_regression_evaluation_summary.csv",
        index=False, encoding="utf-8-sig"
    )
    print("[EVAL_ALL] Selection ranking uses repeated CV only; test metrics are descriptive.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all regression models with CV-only selection.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--models_dir", required=True)
    parser.add_argument("--reports_dir", required=True)
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--out_name", default="all_evaluations")
    parser.add_argument("--include_best", action="store_true")
    args = parser.parse_args()
    run(args.data, args.models_dir, args.reports_dir, args.params, args.out_name, not args.include_best)
