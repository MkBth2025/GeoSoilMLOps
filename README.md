# GeoSoilMLOps

[![DOI](https://zenodo.org/badge/1348126996.svg)](https://doi.org/10.5281/zenodo.22258519)


> A reproducible and configurable MLOps framework for geotechnical and engineering-geology machine learning.

**Version:** 1.0.0

GeoSoilMLOps provides configurable machine-learning workflows for tabular, geotechnical, geological, spatial, and related engineering datasets. Experimental targets, feature sets, grouping strategies, classification settings, model families, input datasets, and output directories are controlled primarily through YAML profiles, reducing the need to modify Python source code between studies.

The repository contains two self-contained MLOps workflows:

- **GTFS** — the geotechnical-only workflow.
- **GLFS** — the geology-enhanced workflow.

GTFS and GLFS are intentionally maintained in separate top-level folders because they are independent MLOps workflows with their own source files, active YAML configuration, models, reports, processed outputs, and identically named result files. Keeping the directory trees separate prevents output collisions and allows either workflow to be run, archived, or reproduced independently.

Within each workflow, the main graphical interface is `soil_mlops_gui.py`, and the canonical active configuration file is `params.yaml`.

## Manuscript-associated release

Version 1.0.0 is the first archived release of GeoSoilMLOps associated with the manuscript:

> *Machine Learning Prediction of Soil Activity Using Geotechnical and Geological Features: Location-Aware Assessment of Model Generalization.*

The release preserves the software, configurations, selected trained models, and selected computational outputs used to document the GTFS and GLFS experimental workflows. The archived software release is intended to support methodological transparency, inspection of the computational workflow, and reuse of the framework with appropriately structured datasets.

GeoSoilMLOps itself is not restricted to soil-activity prediction. Targets, feature sets, grouping variables, classification modes, model families, and input data are configurable so that the framework can be adapted to other suitable tabular geotechnical, geological, spatial, and related engineering applications.

## Repository structure

```text
GeoSoilMLOps/
├── GTFS/
│   ├── src/
│   ├── configs/
│   ├── data/
│   ├── regression_models/
│   ├── classification_models/
│   ├── data_processing_report/
│   ├── regression_report/
│   ├── classification_report/
│   ├── summary_report/
│   ├── soil_mlops_gui.py
│   ├── run_pipeline.py
│   ├── params.yaml
│   └── install.txt
│
├── GLFS/
│   ├── src/
│   ├── configs/
│   ├── data/
│   ├── regression_models/
│   ├── classification_models/
│   ├── data_processing_report/
│   ├── regression_report/
│   ├── classification_report/
│   ├── summary_report/
│   ├── soil_mlops_gui.py
│   ├── run_pipeline.py
│   ├── params.yaml
│   └── install.txt
│
├── .zenodo.json
├── CITATION.cff
├── LICENSE
├── README.md
├── RELEASE_NOTES_v1.0.0.md
├── requirements.txt
└── .gitignore
```

The root `requirements.txt`, `.gitignore`, `LICENSE`, citation metadata, and archival metadata apply to the repository as a whole. Each workflow may additionally maintain workflow-specific documentation, configurations, dependencies, and outputs.

## Manuscript-associated reference artifacts

This repository intentionally includes selected models, evaluation outputs, diagnostic results, ranking outputs, and research summaries from the experiments associated with the manuscript.

These files are retained as **reference artifacts from the manuscript-associated experiments** to support inspection and traceability of the reported computational workflow. Depending on the workflow, the retained artifacts may include:

- selected trained regression and classification models;
- regression and classification evaluation tables;
- final model-ranking outputs;
- Monte Carlo ranking and weight-sensitivity results;
- prediction outputs where redistribution is permitted;
- data-quality, correlation, normality, multicollinearity, ANOVA, and related diagnostic outputs;
- learning-curve and permutation-sensitivity summaries; and
- consolidated PDF, Markdown, JSON, and CSV research summaries.

The reference artifacts are provided to document and inspect the manuscript-associated analyses. They are not intended to replace the original research dataset or to imply that the complete original study can be independently re-executed without access to that dataset.

The original research dataset is not distributed with this repository due to data-sharing restrictions. Any synthetic or sanitized example `samples.csv` distributed with the software is provided solely to demonstrate the expected input structure and workflow and must not be interpreted as the original research dataset.

## Key features

- Two independent GTFS and GLFS MLOps workflows in isolated directory trees
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

## Requirements and installation

Python 3.10 or newer is recommended. From the repository root, create a virtual environment and install the shared repository dependencies from `requirements.txt`.

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

The automated PDF research summary uses `reportlab`, which is included in `requirements.txt`. If an existing environment predates this feature, refresh the dependencies with:

```bash
python -m pip install -r requirements.txt
```

or install only the PDF dependency:

```bash
python -m pip install reportlab
```

The summary stage is non-fatal by default (`summary_report.fail_pipeline_on_error: false`). If ReportLab is unavailable, the run can still produce JSON, CSV, and Markdown summaries while skipping the PDF with a warning.

## Quick start

Run commands from the selected workflow directory so its local `params.yaml`, models, reports, and outputs remain isolated.

### GTFS

```bash
cd GTFS
python soil_mlops_gui.py --params params.yaml
```

### GLFS

```bash
cd GLFS
python soil_mlops_gui.py --params params.yaml
```

If `--params` is omitted, the workflow's local `params.yaml` remains the default active profile.

The enabled main stages can also be executed from the command line:

```bash
python run_pipeline.py --params params.yaml
```

Another experiment profile can be supplied explicitly:

```bash
python run_pipeline.py --params configs/params_spatial.yaml
```

The input CSV may be overridden from the command line:

```bash
python run_pipeline.py --params params.yaml --data path/to/new_data.csv
```

Pipeline stages can be enabled or disabled under the `pipeline` section of the active YAML profile.

## YAML experiment profiles

GeoSoilMLOps supports multiple experiment/model configurations, for example:

```text
params.yaml                     # canonical active profile
configs/params_AC_example.yaml
configs/params_RF.yaml
configs/params_XGB.yaml
configs/params_spatial.yaml
configs/params_feature_set_A.yaml
```

For a new study, start from `params_template.yaml`, save it under a descriptive name, and activate it from the GUI.

In the GUI, a YAML file under **Params profile** can be selected and activated. The current active configuration is backed up and the selected profile is copied to the canonical `params.yaml`, allowing existing scripts to continue to use:

```text
--params params.yaml
```

A YAML profile selected from outside the active workflow can also be imported into `configs/` so that an experiment configuration can be retained with the project.

## Targets and feature sets

`TARGETS` can contain one or many output variables. Each output may have any number of named feature sets; names such as `fs1`, `baseline`, `spatial`, `geology`, or `all_features` are valid.

```yaml
TARGETS:
  Settlement:
    baseline: [LL, PI, Depth]
    spatial: [LL, PI, Depth, Distance, Elevation]

  BearingCapacity:
    fs1: [SPT_N, Depth]
    fs2: [SPT_N, Depth, Groundwater]
```

The framework therefore does not require the target to be `AC` and does not require a fixed number of feature sets.

## Optional location/group-aware validation

A location/group column is optional. The presence of a column such as `Location_No` in a CSV does not automatically enable grouped validation.

For a grouped study:

```yaml
split:
  grouping_enabled: true
  group_column: Location_No
```

For an ordinary tabular study:

```yaml
split:
  grouping_enabled: false
  group_column: null
```

The grouping column can be any suitable identifier, such as `Location_No`, `Site_ID`, `BH`, `Station`, `Patient_ID`, or `Batch_ID`.

When grouping is enabled, the training modules use the configured group-aware procedures so that validation is performed across groups rather than as an ordinary random sample split.

## Optional spatial metadata

Spatial information is independent of grouping. A study may have coordinates without group-aware splitting, use grouped validation without coordinates, use both, or use neither.

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

Regression and classification can be enabled independently.

```yaml
pipeline:
  regression: true
  classification: true

classification:
  enabled: true
```

The main GUI provides **Regression** and **Classification** model-group checkboxes under **New Training**. Their initial states are read from the active YAML profile. A researcher can therefore run regression only, classification only, or both without rewriting the profile for that individual run.

For a continuous target such as `AC`, regression can predict the numeric target while classification can derive categories from configured thresholds.

### Classification target modes

Classification supports categorical, threshold-derived, and automatic target handling.

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

- `categorical`: the target already contains class labels. Binary and multiclass targets are encoded internally and original labels are stored with the model.
- `threshold`: a continuous target is converted to three classes using configured or approved automatically inferred lower/upper boundaries.
- `auto`: configured boundaries imply threshold mode; non-numeric or low-cardinality targets are treated as categorical; other numeric targets use threshold mode.

For an existing class target such as `Ucs_class`, use `categorical`. The regression trainer skips categorical targets by default unless the intentional override `regression.force_categorical_targets: true` is used.

For threshold-derived classification, the numerical limits are not globally hard-coded. Training resolves them from the active profile or configured automatic-boundary workflow and stores the resolved numeric boundaries with the trained classifier. Evaluation and prediction therefore remain tied to the limits used during training.

## Editable model hyperparameters and presets

The repository includes `configs/hyperparameter_presets.yaml`, created from the reference `params_Ac` search spaces. These presets provide starting hyperparameter grids for regression and classification models defined in that reference configuration.

When **Target and Feature Set Manager** opens, predefined grids are loaded first and values already present in the active `params.yaml` are overlaid. Project-specific settings therefore take priority.

The model pane provides:

- **Refresh YAML** — reload the selected model from the active `params.yaml`;
- **Load Preset** — restore the selected model's predefined search space;
- **Load All Presets** — restore all available predefined search spaces in memory.

Nothing is written until **Save All to params.yaml** is used.

A minimal experiment profile may omit large `param_grids` sections while being designed in the GUI. After grids are reviewed and saved, the selected model lists and complete editable search spaces are written into the active `params.yaml`, helping keep the experiment self-contained and reproducible.

The supplied reference configuration is retained as `configs/params_AC_reference.yaml`.

## Configurable project directories

Within each workflow, the GUI allows the principal output locations to be changed. Typical defaults are:

```text
data/processed/
regression_report/
classification_report/
regression_models/
classification_models/
data_processing_report/dq_report/
data_processing_report/multicollinearity/
summary_report/
```

The paths are stored in the active YAML profile, for example:

```yaml
paths:
  processed_data: data/processed
  reports_regression: regression_report
  reports_classification: classification_report
  models_regression: regression_models
  models_classification: classification_models
  dq_report: data_processing_report/dq_report
  multicollinearity_report: data_processing_report/multicollinearity
  summary_report: summary_report
```

The raw input dataset is configured separately:

```yaml
data:
  input_csv: data/raw/samples.csv
```

Project-local paths are stored relatively when possible, making configurations more portable across computers and repository clones.

## Command reference (`install.txt`)

After paths are confirmed in the GUI, `install.txt` is regenerated to provide explicit command-line examples for individual programs using the currently active directories and `params.yaml`.

Examples include:

```text
python src/features.py --input data/raw/samples.csv --out data/processed --params params.yaml
python src/train.py --data data/processed --models_dir regression_models --reports_dir regression_report --params params.yaml
python src/trainclass.py --data data/processed --models_dir classification_models --reports_dir classification_report --params params.yaml
```

This file serves as an installation reference and a reproducible command record for the active configuration.

## Post-training evaluation

For each enabled task, the automatic non-interactive pipeline runs evaluation before generating the consolidated research summary. The intended order is:

1. train models;
2. evaluate regression models and create the regression best-per-feature-set ranking when regression is selected;
3. evaluate classifiers and create the classification best-per-feature-set ranking when classification is selected;
4. generate `summary_report/MLOps_Research_Summary.*` as the final aggregation stage.

The prediction programs:

```text
predict_Reg.py --gui
predict_class.py --gui
```

are deliberately not launched automatically because they open interactive windows and would block the pipeline. They remain available as user-driven tools.

The active YAML can control post-training evaluation with:

```yaml
pipeline:
  regression: true
  classification: true
  regression_evaluation: true
  classification_evaluation: true
  summary_report: true
```

Regression artifacts are written to the configured regression-model and regression-report directories, and classification artifacts to the corresponding classification directories. The workflows do not require nested `report/model/` directories.

## Automated research summary

After the enabled MLOps stages finish, the pipeline can build a compact research report. The default output directory is `summary_report/`.

The report stage collects outputs that already exist in the regression, classification, data-quality, and multicollinearity directories. It does not rerun or reinterpret missing analyses as successful analyses. If a stage was disabled or no corresponding output is found, the report records it as not performed or unavailable.

By default, the summary can include:

- active project, dataset, targets, feature sets, split, and grouping configuration;
- representative regression and classification results;
- nested cross-validation outer-fold summaries and stability;
- group-aware validation status and grouping column;
- learning-curve summaries;
- permutation-sensitivity/permutation-importance summaries;
- availability of data-quality and multicollinearity diagnostics;
- independent-test results when produced by the training workflow; and
- Python/package versions, random seed, active YAML path, dataset SHA-256 hash, and output locations.

Typical files are:

```text
summary_report/
├── MLOps_Research_Summary.pdf
├── MLOps_Research_Summary.md
├── MLOps_Research_Summary.json
├── model_summary.csv
├── nested_cv_summary.csv
├── learning_curve_summary.csv
└── permutation_summary.csv
```

The PDF is intentionally compact. The CSV and JSON outputs should be consulted when exact values or complete per-model results are required.

Example configuration:

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

The report can also be regenerated independently:

```bash
python src/generate_summary_report.py --params params.yaml --output_dir summary_report
```

The GUI provides a **Summary Report** function that can regenerate the PDF/Markdown/JSON/CSV summary from current report directories without retraining models.

## Interpretation of validation diagnostics

The diagnostic components have distinct purposes:

- **Nested cross-validation** estimates out-of-sample performance while keeping model/hyperparameter selection within the inner loop.
- **Independent holdout evaluation** provides final evaluation on a reserved partition and is not intended for tuning.
- **Learning curves** provide diagnostic evidence about sample-size behavior, convergence, bias, and variance.
- **Permutation sensitivity / permutation importance** measures the change in predictive score after a predictor is perturbed and should be interpreted as model dependence rather than causal importance.
- **Group-aware validation**, when enabled, separates configured groups according to the selected validation procedures rather than treating all rows as independent random samples.

## Data availability

The **original research dataset associated with the manuscript is not distributed** with GeoSoilMLOps because of data-sharing restrictions.

Users can supply their own dataset and configure its path, targets, predictors, grouping strategy, spatial metadata, and classification settings through the active YAML profile.

Any synthetic or sanitized example dataset included with the release exists only to:

- demonstrate the expected input schema;
- allow users to inspect the workflow;
- support software testing or demonstration; and
- illustrate configuration of targets and feature sets.

Such example data **must not be interpreted as the original research dataset or as observations used to obtain the manuscript's reported scientific results**.

## Pretrained/public model artifacts

Each workflow may include deliberately selected trained `.pkl` model artifacts for demonstration, inspection, prediction, or reproducibility support. Additional selected model copies may be retained when required by the GUI or prediction workflow.

Before public distribution, serialized models should be checked to ensure that they do not contain raw training rows, sample identifiers, coordinates, private metadata, or other information that should not be redistributed.

The presence of a public trained model does not imply that its original training dataset is distributed with the repository.

## Reports and generated outputs

The model and report directories have distinct responsibilities:

- `regression_models/` — fitted or deliberately retained regression model artifacts;
- `classification_models/` — fitted or deliberately retained classification model artifacts;
- `regression_report/` — regression metrics, rankings, nested-CV tables, learning curves, permutation analyses, predictions, figures, and related outputs;
- `classification_report/` — corresponding classification outputs;
- `data_processing_report/` — data-quality, correlation, normality, multicollinearity, ANOVA, and related diagnostics where generated;
- `summary_report/` — consolidated research-summary outputs.

For the manuscript-associated release, selected generated outputs are intentionally retained as reference artifacts. For new experiments, generated outputs may be excluded from version control unless they are deliberately preserved for reproducibility or publication.

## Reproducibility and archival use

For a reproducible experiment or publication, preserve the exact GTFS or GLFS workflow directory and configuration used for that experiment. Do not mix reports or model artifacts between the two workflow trees even when filenames are identical.

A reproducible archival package should retain, where redistribution is permitted:

- the exact `params.yaml` or experiment-specific YAML profile used for the reported run;
- `requirements.txt` and the supported Python version;
- the repository version or Git tag corresponding to the archived release;
- sanitized example data or a clear statement describing restrictions on the original data;
- selected trained models when useful and safe to redistribute;
- generated summary tables and diagnostics needed to trace reported results; and
- citation and archival metadata.

The generated research summary records software/package versions, random seed, active YAML path, dataset SHA-256 hash, and output locations when available. These metadata support traceability but do not replace preservation of the original experiment configuration.

## Safe cleanup before a new experiment

When `soil_mlops_gui.py` starts, it checks for existing generated experiment outputs and can open a **Clean Previous MLOps Outputs** dialog when removable output folders are present.

Nothing is selected for deletion by default. Typical selectable folders include `regression_report`, `classification_report`, `regression_models`, `classification_models`, `summary_report`, and custom top-level output folders whose names indicate reports, models, results, evaluations, or predictions.

For safety, the cleanup procedure protects `data/`, folders whose names contain `data` (for example `data_processing_report`), `src/`, `configs/`, `.venv/`, `.git/`, parameter backups, tests, and other source/environment folders. A second safety check is performed immediately before deletion. The user must explicitly select folders and confirm permanent deletion.

## Repository dependencies

The root `requirements.txt` includes dependencies used across the repository, not only the primary regression workflow.

Major dependency groups include:

- NumPy, pandas, SciPy, scikit-learn, joblib, and PyYAML;
- matplotlib and statsmodels;
- XGBoost and MLflow;
- GeoPandas, pyproj, and rasterio; and
- openpyxl, chardet, tqdm, and ReportLab.

## Recommended workflow for a new research project

1. Copy `params_template.yaml` to a new descriptively named profile in `configs/`.
2. Set `data.input_csv`.
3. Define one or more targets under `TARGETS`.
4. Define the feature sets for each target.
5. Select regression/classification models and search grids.
6. Decide whether group-aware validation is required.
7. Configure spatial metadata only when the dataset contains it.
8. Configure classification mode and limits when classification is required.
9. Activate the profile in the GUI.
10. Select or confirm output directories.
11. Check the regenerated `install.txt`.
12. Run individual stages from the GUI or execute `run_pipeline.py`.
13. Preserve the configuration and selected outputs when the experiment is intended for publication or archival use.

## Citation

If you use GeoSoilMLOps in academic research, please cite the archived software release.

**GeoSoilMLOps v1.0.0**

Kiani, M., & Kiani Sheikhabadi, M. (2026). *GeoSoilMLOps: A Reproducible and Configurable MLOps Framework for Geotechnical and Engineering-Geology Machine Learning* (Version 1.0.0) [Computer software]. Zenodo.

**Zenodo DOI:** to be added after the v1.0.0 record is published.

The repository includes `CITATION.cff` for machine-readable citation metadata. After Zenodo assigns the release DOI, the DOI should be added to both this section and `CITATION.cff`.

This software release supports the computational workflow associated with the manuscript:

> *Machine Learning Prediction of Soil Activity Using Geotechnical and Geological Features: Location-Aware Assessment of Model Generalization.*

When the associated article receives its final bibliographic information and DOI, the software and publication records can be cross-linked through their related identifiers.

## Versioning and Zenodo archive

The manuscript-associated archival release is identified as:

```text
GeoSoilMLOps v1.0.0
```

The corresponding Git tag should be:

```text
v1.0.0
```

The Zenodo archive should represent this fixed release rather than the continuously changing development state of the repository. Subsequent substantive changes should be released under a new version rather than altering the scientific meaning of the archived v1.0.0 release.

## License

GeoSoilMLOps is distributed under the **MIT License**. See [`LICENSE`](LICENSE) for the complete license terms.

## Contributing

Issues and pull requests that improve reproducibility, documentation, model support, validation workflows, or compatibility with additional geotechnical and engineering-geology datasets are welcome.

Changes should preserve the YAML-driven configuration approach and avoid introducing study-specific assumptions into the general framework where possible.
