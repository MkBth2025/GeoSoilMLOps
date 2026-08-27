SoilMLops classification refactor
=================================

Files
-----
1. src/trainclass4.py
2. src/train_model_code_class/__init__.py
3. src/train_model_code_class/registry.py

Installation
------------
Copy trainclass4.py into src/ and copy the train_model_code_class folder into src/.
The file reuses build_preprocessor() from the refactored src/train.py.

Run directly
------------
python src/trainclass4.py --data data_Ac/processed --models_dir models_Ac --reports_dir reports_Ac --params params.yaml

Automatic execution from train.py
---------------------------------
Either rename trainclass4.py to trainclass.py, or change this line in train.py:

    trainclass_path = Path(__file__).with_name("trainclass.py")

to:

    trainclass_path = Path(__file__).with_name("trainclass4.py")

Important behavior
------------------
- Hyperparameter tuning uses training-set CV only.
- The authoritative Selection Rank uses CV metrics only.
- Validation, when present, is diagnostic and does not select a model.
- The independent test set is loaded only after model selection.
- Only selected overall/per-feature-set models receive final test evaluation.

Primary reports
---------------
reports_Ac_class/classification_summary.csv
reports_Ac_class/classification_ranking.csv
reports_Ac_class/selected_models_before_test.csv
reports_Ac_class/selected_models_final_test_evaluation.csv
reports_Ac_class/best_classifier_meta.json
reports_Ac_class/metrics.json

Primary saved models
--------------------
models_Ac_class/<MODEL>_<FEATURE_SET>_<TARGET>_class.pkl
models_Ac_class/best_classifier.pkl
