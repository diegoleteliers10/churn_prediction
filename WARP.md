# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Repository: churn_prediction (Windows, pwsh)

1) Setup and common commands
- Create and activate venv (PowerShell):
  - python -m venv .venv
  - .venv\Scripts\Activate.ps1
- Install dependencies:
  - pip install -r requirements.txt
- Set PYTHONPATH so src/ is importable in the shell session:
  - $env:PYTHONPATH = "$PWD\src"
- Launch notebooks (optional):
  - python -m jupyter lab
  - python -m jupyter notebook
- Tests (pytest is listed but no tests directory exists yet):
  - Run all tests (when tests/ exists): pytest -q
  - Run a single test: pytest -q tests\test_module.py::TestClass::test_case
- Code formatting/linting:
  - No formatter/linter is configured in this repo (no black/ruff/flake8 configs found). Skip unless added.

2) High-level architecture and working model
- Intent (from README): Predict telecom customer churn with a focus on high Recall (≥85%). Pipeline phases: data exploration → preprocessing → feature engineering → modeling → evaluation → deployment.
- Directory conventions (per README; some may need to be created locally):
  - data/ (expected subfolders: raw/, processed/, external/)
  - notebooks/ (exploratory/, modeling/)
  - src/ (code package)
  - models/ (trained/, pipelines/)
  - reports/ (figures/, tables/)
  - config/ (YAML configuration)
- Present code (src/):
  - src/utils.py
    - setup_logging(level): basic logging setup used across scripts.
    - load_config(config_path='config/config.yaml'): YAML loader; expects a config/config.yaml file.
    - save_dataframe(df, filename, path='data/processed/'): CSV writer with logging; creates file under data/processed/.
    - load_dataframe(filename, path='data/processed/'): CSV reader with logging.
    - ChurnConfig(config_path='config/config.yaml'): thin wrapper around loaded YAML with properties:
      - target_column, test_size, random_state, target_metrics (expects keys under model: target_column, test_size, random_state, target_recall, target_precision, target_auc).
  - src/data/, src/features/, src/models/, src/visualization/: packages are present but currently empty (only __init__.py), intended for scripts/functions per phase of the pipeline.
- Configuration expectations
  - Create config/config.yaml with at least:
    model:
      target_column: <string>
      test_size: <float>
      random_state: <int>
      target_recall: <float>
      target_precision: <float>
      target_auc: <float>
  - Example minimal file to unblock code that reads config:
    model:
      target_column: Churn
      test_size: 0.2
      random_state: 42
      target_recall: 0.85
      target_precision: 0.4
      target_auc: 0.8
- Data layout expectations
  - Ensure the following exist before running data utilities:
    - data/processed/ (for CSV outputs used by save_dataframe/load_dataframe)
    - data/raw/ and data/external/ as needed by your ingestion/EDA flow.
- README notes to reconcile with code
  - README shows usage from src.models import ChurnPredictor with .load/.predict/.predict_proba. That class is not implemented in the current codebase. Implement it under src/models/ or adjust usage accordingly before attempting that snippet.

3) Practical workflows
- End-to-end local flow (scripts oriented):
  - Activate venv, install deps, set PYTHONPATH.
  - Create config/config.yaml as above.
  - Place source data under data/raw/.
  - Implement ingestion/cleaning functions under src/data/, feature builders under src/features/, model training/eval under src/models/ using ChurnConfig for parameters.
  - Use utils.save_dataframe/load_dataframe to persist intermediate datasets to data/processed/.
  - Persist trained models to models/trained/ (path conventions per your implementation).
- Running ad-hoc code from the shell:
  - python -c "from utils import setup_logging, ChurnConfig; setup_logging(); cfg = ChurnConfig(); print(cfg.target_metrics)"  (requires $env:PYTHONPATH set to src)

4) Repository metadata collected
- requirements.txt specifies: pandas, numpy, matplotlib, seaborn, plotly, scikit-learn, xgboost, lightgbm, catboost, shap, imbalanced-learn, jupyter, ipywidgets, tqdm, joblib, pickle5, flask, fastapi, uvicorn, pytest, pytest-cov, python-dotenv, pyyaml, loguru.
- .env exists (content not read here). If you add dotenv loading, ensure your entry points call python-dotenv to load environment variables early.
- No Makefile, Dockerfile, pyproject.toml, or test suite present at this time.

5) Guardrails for future agents in this repo
- Prefer Windows-friendly paths and PowerShell examples. Use $env:PYTHONPATH to enable imports from src/ when running one-off commands.
- Don’t assume notebooks/ or config/ exist; create them if necessary based on README’s structure.
- When adding tests, place them under tests/ and use pytest; keep test data paths relative to project root.

