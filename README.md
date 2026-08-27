# GeoSoilML

> A reproducible and configurable MLOps framework for geotechnical and engineering-geology machine learning.

GeoSoilML provides configurable machine-learning workflows for tabular, geotechnical, geological, spatial, and related engineering datasets. Experimental targets, feature sets, grouping strategies, classification settings, model families, input datasets, and output directories are controlled primarily through YAML profiles, reducing the need to modify Python source code between studies.

The main graphical interface is `soil_mlops_gui.py`, and the canonical active configuration file is `params.yaml`.

## Key features

- YAML-driven experiment configuration
- Configurable regression and classification workflows
- Multiple targets and arbitrary named feature sets
- Optional location/group-aware validation
- Optional spatial metadata
- Editable model-specific hyperparameter search spaces and reusable presets
- Nested cross-validation and independent holdout evaluation
- Learning-curve and permutation-sensitivity diagnostics
- Automated model evaluation and ranking
- Interactive single-sample regression and classification prediction
- Automated PDF, Markdown, JSON, and CSV research summaries
- GUI and command-line workflows
- Reproducibility metadata and experiment profiles


## Python installation

Python 3.10 or newer is recommended. Create a virtual environment and install the repository dependencies from `requirements.txt`.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows Command Prompt

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`tkinter` is part of the standard Python installation on Windows. On some Linux distributions it is supplied by the operating system rather than `pip` (for example, a package commonly named `python3-tk`).

## Start the GUI

```bash
python soil_mlops_gui.py --params params.yaml
```

If `--params` is omitted, `params.yaml` remains the default active profile.

## Multiple YAML experiment profiles

You can keep any number of experiment/model configurations, for example:

```text
params.yaml                     # active profile used by the programs
configs/params_AC_example.yaml
configs/params_RF.yaml
configs/params_XGB.yaml
configs/params_spatial.yaml
configs/params_feature_set_A.yaml
```

In the GUI, choose a YAML file under **Params profile** and activate it. The current active configuration is backed up and the selected profile is copied to the canonical `params.yaml`. Existing scripts therefore continue to work with:

```text
--params params.yaml
```

A YAML file selected from outside the repository can also be imported into `configs/` so an experiment configuration can be kept with the GitHub project.

For a new study, start from `params_template.yaml`, save it under a descriptive name, and activate it from the GUI.

## Targets and feature sets

`TARGETS` can contain one or many output variables. Each output may have any number of named feature sets; names such as `fs1`, `baseline`, `spatial`, `geology`, or `all_features` are all valid.

```yaml
TARGETS:
  Settlement:
    baseline: [LL, PI, Depth]
    spatial: [LL, PI, Depth, Distance, Elevation]

  BearingCapacity:
    fs1: [SPT_N, Depth]
    fs2: [SPT_N, Depth, Groundwater]
```

The repository therefore does not require the target to be `AC` and does not require a fixed number of feature sets.

## Optional location/group-aware validation

A location/group column is optional. Merely having a column such as `Location_No` in the CSV does not automatically enable grouped validation.

Grouped study:

```yaml
split:
  grouping_enabled: true
  group_column: Location_No
```

Ordinary tabular study:

```yaml
split:
  grouping_enabled: false
  group_column: null
```

The grouping column can be any suitable identifier, such as `Location_No`, `Site_ID`, `BH`, `Station`, `Patient_ID`, or `Batch_ID`.

## Optional spatial metadata

Spatial information is independent of grouping. A study may have coordinates without using group-aware splitting, use grouped validation without coordinates, use both, or use neither.

```yaml
data:
  spatial:
    enabled: true
    latitude_column: Latitude
    longitude_column: Longitude
```

For a dataset without coordinates:

```yaml
data:
  spatial:
    enabled: false
```

## Regression and classification

Regression and classification can be enabled independently through the active YAML profile.

Classification disabled:

```yaml
classification:
  enabled: false
```

Fixed three-class limits for one target:

```yaml
classification:
  enabled: true
  targets: [AC]
  class_boundaries:
    AC:
      lower: 0.75
      upper: 1.25
```

The numerical limits are not globally hard-coded. Training resolves the limits from the active profile (or the configured automatic-boundary workflow) and stores the resolved numeric boundaries with the trained classifier. Classification evaluation and prediction use the model metadata so a model remains tied to the limits with which it was trained.

### Predefined, editable model hyperparameters

The repository includes `configs/hyperparameter_presets.yaml`, created from the reference `params_Ac` search spaces. These presets provide starting hyperparameter grids for the regression and classification models that were defined in that reference configuration.

When **Target and Feature Set Manager** opens, predefined grids are loaded first and values already present in the active `params.yaml` are overlaid on top. Therefore, project-specific settings always take priority. Select a model to inspect or edit its YAML search space directly.

The model pane provides three useful actions:

- **Refresh YAML** reloads that model from the current active `params.yaml` and discards unsaved editor changes for the selected model.
- **Load Preset** restores the selected model to its predefined search space from `configs/hyperparameter_presets.yaml`.
- **Load All Presets** restores all available predefined search spaces in memory. Nothing is written until **Save All to params.yaml** is used.

A minimal experiment profile may therefore omit large `param_grids` sections while it is being designed in the GUI. After the user reviews/modifies the grids and saves, the chosen model lists and complete editable search spaces are written into the active `params.yaml`, keeping each experiment self-contained and reproducible.

The full supplied reference configuration is also retained as `configs/params_AC_reference.yaml`.

## Selectable project directories

The GUI allows the principal output locations to be changed. Default paths are:

```text
data/processed/
regression_report/
classification_report/
regression_models/
classification_models/
data_processing_report/dq_report/
data_processing_report/multicollinearity/
```

They are stored in the active YAML profile:

```yaml
paths:
  processed_data: data/processed
  reports_regression: regression_report
  reports_classification: classification_report
  models_regression: regression_models
  models_classification: classification_models
  dq_report: data_processing_report/dq_report
  multicollinearity_report: data_processing_report/multicollinearity
```

The raw input dataset is configured separately:

```yaml
data:
  input_csv: data/raw/samples.csv
```
Project-local paths are stored relatively when possible, making configurations more portable across computers and GitHub clones.

### Data availability

The original research dataset is not distributed with this repository due to data-sharing and privacy restrictions. Users can provide their own dataset and configure its location, targets, and predictors through the active YAML profile. A synthetic or sanitized example dataset may be included solely to demonstrate the expected input structure and should not be interpreted as the original research dataset.
## Command reference (`install.txt`)

After paths are confirmed in the GUI, `install.txt` is regenerated to provide explicit command-line examples for the individual programs using the currently active directories and `params.yaml`.

Examples include:

```text
python src/features.py --input data/raw/samples.csv --out data/processed --params params.yaml
python src/train.py --data data/processed --models_dir regression_models --reports_dir regression_report --params params.yaml
python src/trainclass.py --data data/processed --models_dir classification_models --reports_dir classification_report --params params.yaml
```

This file serves as both an installation reference and a reproducible command record for the active configuration.

## Run the configurable pipeline

Run the enabled main stages using the active profile:

```bash
python run_pipeline.py --params params.yaml
```

Or run another profile explicitly:

```bash
python run_pipeline.py --params configs/params_spatial.yaml
```

The input CSV may also be overridden from the command line:

```bash
python run_pipeline.py --params params.yaml --data path/to/new_data.csv
```

Pipeline stages can be enabled or disabled under the `pipeline` section of the YAML profile.

## Repository dependency file

`requirements.txt` includes dependencies used across the repository, not only the main regression script. This includes the numerical/scientific stack, XGBoost, MLflow, statistics, Excel support, and optional spatial/geological utilities imported by repository modules.

Major dependency groups are:

- NumPy, pandas, SciPy, scikit-learn, joblib, PyYAML
- matplotlib and statsmodels
- XGBoost and MLflow
- GeoPandas, pyproj, and rasterio
- openpyxl, chardet, and tqdm

## Recommended workflow for a new research project

1. Copy `params_template.yaml` to a new descriptive profile in `configs/`.
2. Set `data.input_csv`.
3. Define one or more targets under `TARGETS`.
4. Define the feature sets for each target.
5. Select regression/classification models and search grids.
6. Decide whether group-aware validation is required.
7. Configure spatial metadata only when the dataset has it.
8. Configure classification limits only when classification is required.
9. Activate the profile in the GUI.
10. Select/confirm output directories.
11. Check the regenerated `install.txt`.
12. Run individual stages from the GUI or execute `run_pipeline.py`.

## Repository and archival guidance

Generated reports, processed data, caches, virtual environments, MLflow artifacts, private/raw research datasets, and non-selected model artifacts should remain excluded through `.gitignore`. Commit the source code, dependency specification, sanitized YAML profiles, and—when permitted—a small non-sensitive example dataset. A deliberately selected public model artifact may be retained in `models/` when appropriate. For a Zenodo release, archive a tagged GitHub release and include the exact configuration files needed to reproduce the reported experiments.



## Pretrained/public model artifacts

GeoSoilML may include a deliberately selected trained `.pkl` model in `models/` for demonstration or reproducibility purposes. Other generated model artifacts can remain excluded from version control by default.

Before publishing a serialized model, verify that the artifact does not contain raw training rows, sample identifiers, coordinates, private metadata, or other information that should not be distributed. The presence of a public model artifact does not imply that the original training dataset is distributed with this repository.

## Automated research summary report

After the enabled MLOps stages finish, the pipeline can automatically build a compact research report. The default output directory is `summary_report/` and can be changed from **Paths...** in the GUI or with `paths.summary_report` in `params.yaml`.

The report stage collects the outputs that already exist in the regression, classification, data-quality, and multicollinearity directories. It does not rerun or reinterpret missing analyses as successful analyses. If a stage was disabled or no corresponding output files are found, the PDF records **Not performed / not available**.

By default the summary includes:

- active project, dataset, targets, feature sets, split and grouping configuration;
- representative regression and classification model results;
- nested cross-validation outer-fold summaries and stability;
- group-aware validation status and grouping column when configured;
- learning-curve summaries for regression and classification;
- permutation-sensitivity / permutation-importance top predictors;
- availability of data-quality and multicollinearity diagnostics;
- independent-test results when the training workflow produced them;
- Python/package versions, random seed, active YAML path, dataset SHA-256 hash, and output locations for reproducibility.

Default files are:

```text
summary_report/
├── MLOps_Research_Summary.pdf
├── MLOps_Research_Summary.json
├── analysis_inventory.csv
├── model_summary.csv
├── nested_cv_summary.csv
├── learning_curve_summary.csv
└── permutation_summary.csv
```

The PDF is intentionally brief. The CSV and JSON files should be used when exact values or complete per-model results are required.

The corresponding configuration is:

```yaml
paths:
  summary_report: summary_report

pipeline:
  summary_report: true

evaluation:
  nested_cv_outer_repeats: 1
  generate_learning_curves: true
  generate_permutation_sensitivity: true
  permutation_sensitivity_repeats: 30
  permutation_sensitivity_n_jobs: -1

summary_report:
  enabled: true
  pdf: true
  pdf_filename: MLOps_Research_Summary.pdf
  include_nested_cv: true
  include_learning_curves: true
  include_permutation_sensitivity: true
  include_reproducibility: true
```

The final report can also be generated independently after any partial or complete run:

```bash
python src/generate_summary_report.py --params params.yaml --output_dir summary_report
```

### Interpretation of the validation diagnostics

The summary deliberately keeps the roles of the diagnostics separate. Nested CV estimates out-of-sample performance while keeping model/hyperparameter selection inside the inner loop. The independent test partition is reserved for final holdout evaluation and is not used for tuning. Learning curves are diagnostic evidence about sample-size behavior, convergence, bias, and variance. Permutation sensitivity measures the decrease in predictive score after a predictor is perturbed and should be interpreted as model dependence, not causal importance. When grouping is enabled, the training modules use the configured group-aware procedures so the report can describe the validation as group-aware rather than as an ordinary random sample split.


## Classification target modes (binary, multiclass, or threshold-derived)

Classification is no longer restricted to a continuous target split into exactly three classes. Each target can use one of three modes:

```yaml
classification:
  enabled: true
  targets: [Ucs_class, AC]
  target_modes:
    Ucs_class: categorical
    AC: threshold
  class_boundaries:
    AC:
      lower: 0.75
      upper: 1.25
```

- `categorical`: the target already contains class labels. Binary and multiclass targets are encoded internally and the original labels are stored with each model. No numeric class boundaries are calculated.
- `threshold`: the target is continuous and is converted to three classes using configured or approved automatically inferred lower/upper boundaries.
- `auto`: when no mode is supplied, configured boundaries imply `threshold`; non-numeric or low-cardinality targets are treated as categorical; other numeric targets use threshold mode.

For an existing class target such as `Ucs_class`, use `categorical`. This also prevents inappropriate normality testing of the class labels. The regression trainer skips categorical targets by default because regression on arbitrary class codes is normally not meaningful. An intentional override is available with `regression.force_categorical_targets: true`.

The Target and Feature Set Manager now exposes **Classification target mode** (`auto`, `threshold`, or `categorical`) next to the class-boundary controls. In categorical mode, lower/upper boundaries are ignored.


## PDF summary dependency and existing environments

The automated PDF summary uses `reportlab`, which is included in `requirements.txt`. If you update an existing clone/virtual environment after this feature was added, refresh the environment once:

```bash
python -m pip install -r requirements.txt
```

or install only the missing PDF dependency:

```bash
python -m pip install reportlab
```

The summary stage is deliberately non-fatal by default (`summary_report.fail_pipeline_on_error: false`). If ReportLab is missing, the run still completes and creates JSON, CSV, and Markdown summaries; the PDF is skipped with a clear warning. Set `fail_pipeline_on_error: true` only if PDF/report creation must be mandatory for a run to be considered successful.

### Models and reports are intentionally separate

The active output paths have distinct responsibilities:

- `regression_models/` contains fitted regression `.pkl` artifacts and selected-model copies.
- `classification_models/` contains fitted classification `.pkl` artifacts.
- `regression_report/` contains regression metrics, rankings, nested-CV tables, learning curves, permutation analysis, predictions, figures, and other report outputs.
- `classification_report/` contains the corresponding classification reports.

A nested folder such as `regression_report/model/` or `classification_report/model/` is **not** part of the current layout. If an old version left one behind and it is empty, the pipeline removes it automatically. Non-empty legacy folders are never deleted automatically.

## Regression / Classification task selection

The main GUI provides two **Model groups** checkboxes under **New Training**:

- **Regression**
- **Classification**

Their initial states are read from the active `params.yaml`. For example:

```yaml
pipeline:
  regression: true
  classification: true

classification:
  enabled: true
```

starts the GUI with both boxes selected. The researcher can then run regression only, classification only, or both without editing the profile. The selection applies to that run and is passed to `run_pipeline.py` with `--tasks`. If `--tasks` is omitted, the pipeline uses the YAML values directly.

For a continuous target such as `AC`, the same target can be used by both groups: regression predicts the numeric target directly, while classification can derive classes using configured thresholds such as `0.75` and `1.25`.

Regression artifacts are written only to `models_regression` / `reports_regression`, and classification artifacts only to `models_classification` / `reports_classification`. Unselected task folders are not created merely for symmetry.


## Post-training evaluation and final summary order

The automatic pipeline runs the complete non-interactive evaluation chain **before** generating the consolidated research summary. For each enabled task, the order is:

1. train models;
2. evaluate all regression models and create the regression best-per-feature-set ranking (when regression is selected);
3. evaluate the selected classifier, evaluate all classifiers, and create the classification best-per-feature-set ranking (when classification is selected);
4. generate `summary_report/MLOps_Research_Summary.*` as the final aggregation stage.

The prediction programs (`predict_Reg.py --gui` and `predict_class.py --gui`) are deliberately **not** launched automatically because they open interactive windows and would block the pipeline. They remain available as user-driven tools.

The main GUI also provides a **Summary Report** button next to **Regression Report** and **Classification Report**. This regenerates the PDF/Markdown/JSON/CSV summary from the current report directories without retraining models. This is useful after rerunning an evaluator or ranking script.

The active YAML can control post-training evaluation with:

```yaml
pipeline:
  regression: true
  classification: true
  regression_evaluation: true
  classification_evaluation: true
  summary_report: true
```

The summary stage preferentially reads `regression_report/all_evaluations.csv` and `classification_report/all_evaluations_class.csv`, together with nested-CV, learning-curve, permutation-sensitivity, ranking, and final-test outputs. Therefore it should be run last.


## Reproducibility and archival use

For a reproducible publication or Zenodo software release, retain the following together with the source code:

- the exact `params.yaml` or experiment-specific YAML profile used for the reported run;
- `requirements.txt` and the supported Python version;
- the repository version or Git tag corresponding to the archived release;
- sanitized example data or a clear statement describing restrictions on the original dataset;
- generated summary tables needed to trace the reported results, when redistribution is permitted.

The generated research summary records software/package versions, random seed, the active YAML path, dataset SHA-256 hash, and output locations when those values are available. These metadata support traceability but do not replace preservation of the original experiment configuration.


## Safe cleanup before a new experiment

When `soil_mlops_gui.py` starts, it checks for existing generated experiment outputs and opens a **Clean Previous MLOps Outputs** dialog when removable output folders are present. This is intended for rerunning the MLOps workflow without manually deleting old reports and models.

Nothing is selected for deletion by default. Typical selectable folders include `regression_report`, `classification_report`, `regression_models`, `classification_models`, `summary_report`, and custom top-level output folders whose names indicate reports/models/results/evaluations/predictions.

For safety, the cleanup dialog excludes and protects `data/`, any folder whose name contains `data` (for example `data_processing_report`), `src/`, `configs/`, `.venv/`, `.git/`, parameter backups, tests, and other source/environment folders. A second safety check is performed immediately before deletion. The user must select folders explicitly and confirm permanent deletion.

## Citation

If you use GeoSoilML in academic research, please cite the associated software release and publication when citation information becomes available. For an archived release, the repository can be connected to Zenodo and the resulting DOI and citation metadata added here.

## Contributing

Issues and pull requests that improve reproducibility, documentation, model support, validation workflows, or compatibility with additional geotechnical and engineering-geology datasets are welcome. Changes should preserve the YAML-driven configuration approach and avoid introducing study-specific assumptions into the general workflow where possible.

## License

GeoSoilML is intended for distribution under the MIT License. See the repository `LICENSE` file for the exact license terms.

