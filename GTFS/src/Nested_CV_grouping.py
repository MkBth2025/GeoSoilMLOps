from pathlib import Path
import argparse
import pandas as pd


def merge_nested_cv(reports_dir: Path):
    """
    Merge all *_nested_cv_folds.csv files found recursively under
    reports_dir/nested_cv.

    Parameters
    ----------
    reports_dir : Path
        Path to reports directory (e.g., reports_AC or reports_AC_class).
    """

    nested_cv_dir = reports_dir / "nested_cv"

    if not nested_cv_dir.exists():
        raise FileNotFoundError(
            f"Directory not found:\n{nested_cv_dir}"
        )

    csv_files = sorted(
        nested_cv_dir.rglob("*_nested_cv_folds.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No '*_nested_cv_folds.csv' files found in\n"
            f"{nested_cv_dir}"
        )

    dfs = []

    for csv_file in csv_files:

        # Model name from filename
        # Example:
        # ANN_nested_cv_folds.csv -> ANN
        model = csv_file.stem.replace(
            "_nested_cv_folds", ""
        )

        # Feature set from parent folder
        # Example:
        # fs1, fs2, fs3, fs4
        feature_set = csv_file.parent.name

        df = pd.read_csv(csv_file)

        # Insert identifying columns
        df.insert(0, "Feature_Set", feature_set)
        df.insert(0, "Model", model)

        dfs.append(df)

        print(
            f"Loaded "
            f"{csv_file.relative_to(reports_dir)} "
            f"({len(df)} rows)"
        )

    merged = pd.concat(
        dfs,
        ignore_index=True,
        sort=False,
    )

    # Determine output filename
    reports_name = reports_dir.name.lower()

    if reports_name == "reports_ac":
        suffix = "_R"
    elif reports_name == "reports_ac_class":
        suffix = "_C"
    else:
        suffix = ""

    output_file = (
        nested_cv_dir /
        f"CV_all_nested{suffix}.csv"
    )

    merged.to_csv(
        output_file,
        index=False,
    )

    print("\n-------------------------------------")
    print(f"Reports directory : {reports_dir}")
    print(f"Nested CV folder  : {nested_cv_dir}")
    print(f"Files merged      : {len(csv_files)}")
    print(f"Rows written      : {len(merged)}")
    print(f"Columns           : {len(merged.columns)}")
    print(f"Output            : {output_file}")


def main():

    parser = argparse.ArgumentParser(
        description="Merge nested cross-validation CSV files."
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help=(
            "Path to reports directory "
            "(e.g., reports_AC or reports_AC_class)"
        ),
    )

    args = parser.parse_args()

    merge_nested_cv(args.input)


if __name__ == "__main__":
    main()