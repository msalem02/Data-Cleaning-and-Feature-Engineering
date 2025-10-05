# Data Cleaning and Feature Engineering
### End‑to‑End, Reproducible Data Preparation Pipelines for Modeling and Analytics

---

## 1) Overview

This repository provides a complete, production‑style blueprint for **cleaning raw datasets, auditing data quality, and engineering robust features** that are ready for machine‑learning and analytics. It includes a modular pipeline (designed around scikit‑learn concepts), reproducible configuration, clear folder conventions, and guidance for scaling from exploratory notebooks to reliable scripts.

While the project can work with any tabular dataset, the examples and defaults map to common teaching/portfolio datasets (e.g., **Titanic**, **Mall Customers**, **California Housing**). The structure is intentionally generic so you can drop in your data and run the same steps with minimal edits.

---

## 2) Goals and Non‑Goals

**Goals**
- Deliver **repeatable** data preparation steps (cleaning → transformation → feature engineering → export).
- Provide **transparent documentation** of assumptions and decisions (what changed, why it changed).
- Support **both** notebook‑driven EDA and script‑based automation.
- Use industry‑standard components: **pandas**, **NumPy**, **scikit‑learn** (`Pipeline`, `ColumnTransformer`), and optional **feature_engine**.
- Produce **model‑ready artifacts** (clean CSV/Parquet, transformers, reports, logs).

**Non‑Goals**
- Training complex ML models (that belongs in a separate modeling repo).
- Handling unbounded big data (the patterns are scalable, but this repo targets single‑machine workflows).

---

## 3) Project Structure

Recommended (and assumed by examples):

Data-Cleaning-and-Feature-Engineering/
│
├── data_raw/                         # As‑received datasets (read‑only; never edit in place)
│   └── sample.csv
├── data_intermediate/                # Temporary working files (after partial cleaning)
├── data_processed/                   # Final, model‑ready datasets (CSV/Parquet)
├── notebooks/                        # EDA and step-by-step walk‑throughs
│   ├── 01_data_audit.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_cleaning.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── 05_pca_dimensionality_reduction.ipynb
├── src/                              # Reusable, testable Python modules
│   ├── io_funcs.py                   # load/save helpers, schema checks
│   ├── audit.py                      # missingness profile, duplicates, type inference
│   ├── cleaning.py                   # imputation, type fixes, outliers, rare levels
│   ├── feat_eng.py                   # encoders, scalers, math transforms, date/text features
│   ├── selectors.py                  # feature selection (filter/wrapper/embedded)
│   ├── reducers.py                   # PCA and other DR methods
│   ├── pipeline.py                   # unify steps with ColumnTransformer + Pipeline
│   └── run_pipeline.py               # CLI entry point to run end‑to‑end pipeline
├── configs/
│   └── base.yaml                     # declarative config: column roles, strategies, paths
├── reports/                          # Data Quality Reports (DQR), profiling, logs, artifacts
│   ├── dqr_summary.md
│   └── logs/
├── tests/                            # Optional: unit tests (pytest)
├── requirements.txt
└── README.md

> If a folder does not exist in your current repo, create it as you adopt this structure. The code snippets below do not assume a specific dataset name, only standard column roles (numeric/categorical/date/text/target).

---

## 4) End‑to‑End Pipeline: From Raw to Model‑Ready

The pipeline implements the following **ordered** phases. Each phase can be executed in notebooks (for exploration) or via `src/run_pipeline.py` (for automation).

### 4.1 Data Ingestion
- Load from CSV/Parquet.
- Enforce **read‑only** for `data_raw/` to preserve provenance.
- Optional schema check: column presence, dtypes, allowed ranges.

### 4.2 Data Audit (DQR)
- **Shape and types**: count rows/columns, numeric vs categorical counts.
- **Missingness profile**: per‑column % missing; MCAR/MAR/MNAR hypothesis.
- **Duplicates**: exact row duplicates; key‑based uniqueness checks (e.g., `PassengerId` in Titanic).
- **Cardinality**: levels per categorical column (identify high‑cardinality risks).
- **Targets and leakage**: flag columns created after the target event (e.g., future info).
- **Basic drift checks** (optional): train/test time splits by date to ensure stability.

Artifacts: `reports/dqr_summary.md`, log files in `reports/logs/`.

### 4.3 Cleaning
- **Type fixes**: enforce integer/float/date categories; parse dates with `pd.to_datetime`.
- **Missing values**:
  - Numeric: mean/median imputation; **KNN** or model‑based if needed.
  - Categorical: most frequent or explicit `"Missing"` category.
  - Dates: domain‑specific (e.g., impute with mode month/day); otherwise leave null if meaningful.
- **Outliers**:
  - IQR rule (`Q3 + 1.5*IQR`, `Q1 − 1.5*IQR`) or Z‑scores.
  - **Winsorize** or **clip** where justified; avoid blind removal if data are scarce.
- **Duplicates**: drop exact duplicates; for key duplicates, pick latest by date or aggregate.
- **String normalization**: trim spaces, unify case, standardize known labels.
- **Rare category grouping**: consolidate categories with frequency below a threshold into `"Other"`.

Outputs to: `data_intermediate/clean_<dataset>.csv`

### 4.4 Feature Engineering
- **Categorical encoding**:
  - **One‑Hot** for low‑cardinality.
  - **Ordinal** when order is intrinsic.
  - **Target Encoding** for very high cardinality (with CV and noise, to avoid leakage).
- **Numerical scaling**:
  - **StandardScaler** (mean 0, std 1) for linear models/NNs.
  - **MinMaxScaler** for distance‑based models; **RobustScaler** if heavy tails.
- **Mathematical transforms**: log/Box‑Cox/Yeo‑Johnson for skewed positive features.
- **Date‑time features**: year/month/weekday/hour, **elapsed time** intervals, season flags.
- **Text features** (optional): bag‑of‑words/TF‑IDF for short text columns.
- **Interactions**: polynomial terms or domain interactions (e.g., `price*quantity`).
- **Leakage guardrails**: ensure transforms see only training folds during CV.

### 4.5 Feature Selection
- **Filter methods**: variance threshold, mutual information, univariate tests.
- **Wrapper methods**: RFE with a simple estimator.
- **Embedded**: L1 (Lasso) for linear models; tree‑based importances for nonlinear signals.

### 4.6 Dimensionality Reduction (PCA, optional)
- Standardize inputs; choose **n_components** via explained‑variance targets (e.g., 90–95%).
- Export plots: scree plot and cumulative variance curve.
- Save fitted PCA to reuse on test/production.

### 4.7 Export Artifacts
- **Processed dataset** (CSV/Parquet) in `data_processed/`.
- **Fitted transformers** (encoders/scalers/PCA) via `joblib` for reuse.
- **Data dictionary** and DQR in `reports/`.
- **Logs** of each run (timestamped).

---

## 5) Configuration‑Driven Workflow (configs/base.yaml)

A single YAML file controls column roles and strategies, enabling reproducibility without editing code:

configs/base.yaml (illustrative)
role:
  target: Survived
  id: PassengerId
  numeric: [Age, Fare, SibSp, Parch]
  categorical: [Sex, Embarked, Pclass]
  datetime: []
  text: []
impute:
  numeric: median
  categorical: most_frequent
outliers:
  method: iqr
  clip_quantiles: [0.01, 0.99]
encode:
  categorical: onehot
scale:
  numeric: standard
pca:
  enabled: true
  n_components: 0.95
export:
  processed_path: data_processed/processed.csv
  artifacts_dir: reports/

> Adjust the roles and strategies per dataset. If you do not use YAML, pass equivalent arguments to `run_pipeline.py` CLI flags.

---

## 6) Implementation Details (src/)

### 6.1 ColumnTransformer + Pipeline (core pattern)

Example (minimal sketch):

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

numeric_features = ["Age", "Fare"]
categorical_features = ["Sex", "Embarked"]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocess = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features)
])

# preprocess.fit(X_train); X_train_t = preprocess.transform(X_train)

> The repository generalizes this pattern, adding optional PCA and selection steps, and exporting fitted transformers with `joblib`.

### 6.2 Outlier Handling Helpers
- IQR‑based clipping with configurable quantiles (e.g., 1st–99th percentiles).
- Pluggable Z‑score filters; logging of counts removed/clipped.

### 6.3 Feature Selection
- Implemented as standalone scikit‑learn transformers so they fit inside the primary pipeline (no leakage).

### 6.4 Logging
- Python `logging` with **INFO** level for summary and **DEBUG** for detailed row/column counts.
- Each run writes to `reports/logs/run_<timestamp>.log`.

### 6.5 Reproducibility
- Single **random_state** passed to all stochastic components.
- Deterministic train/validation splits (time‑aware splits if a date column is configured).

---

## 7) How to Run

### 7.1 Prerequisites
- Python 3.10+ recommended

### 7.2 Install
pip install -r requirements.txt

### 7.3 Notebook Workflow (interactive)
- Open `notebooks/01_data_audit.ipynb` → audit.
- `02_eda.ipynb` → visualize distributions and relationships.
- `03_cleaning.ipynb` → apply missing/outlier handling.
- `04_feature_engineering.ipynb` → encodings/scaling/transforms.
- `05_pca_dimensionality_reduction.ipynb` → PCA and plots.
- Export to `data_processed/`.

### 7.4 Script Workflow (automated)
python -m src.run_pipeline --config configs/base.yaml
# or
python src/run_pipeline.py --config configs/base.yaml

CLI flags (if implemented):
--input data_raw/sample.csv
--output data_processed/processed.csv
--loglevel INFO

Outputs:
- `data_processed/processed.csv`
- `reports/dqr_summary.md`
- `reports/logs/run_<timestamp>.log`
- `reports/artifacts/…` (transformers, PCA, selections)

---

## 8) Data Quality Report (DQR) — What We Document

- **Completeness**: % missing by column; total missing cells.
- **Consistency**: type mismatches, category label inconsistencies.
- **Uniqueness**: duplicate rows; duplicate keys.
- **Validity**: domain checks (e.g., `Age ≥ 0`, `Fare ≥ 0`).
- **Timeliness**: if dates exist, time ranges and gaps.
- **Outlier summary**: how many values clipped/removed and where.
- **Change log**: per‑step record of transformations applied.

DQR is written to `reports/dqr_summary.md` and referenced in the README for traceability.

---

## 9) Feature Engineering Playbook (Cheat Sheet)

- **Categorical**
  - Low cardinality → One‑Hot.
  - Ordinal categories → OrdinalEncoder with explicit order.
  - High cardinality → Target encoding with CV and regularization.
- **Numerical**
  - Skewed positive → log1p or Yeo‑Johnson.
  - Heavy outliers → RobustScaler.
  - Interaction terms → polynomial features (with caution on dimensionality).
- **Dates**
  - Extract y/m/d/weekday; cyclical encoding for hour/weekday using sin/cos.
  - Time since event: `now − date_col` in days/weeks.
- **Text**
  - Short categorical‑like text → hashing/TF‑IDF with min_df.
  - Consider length, word count, basic sentiment proxies.

---

## 10) Testing and CI (Optional but Recommended)

- **Unit tests** with `pytest` in `tests/` (e.g., imputation behavior, encoder outputs, PCA shapes).
- **Data contracts**: assert expected columns and types before/after pipeline.
- **CI** (GitHub Actions): run tests and linting on push/PR.
- **Pre‑commit hooks**: `black`, `ruff`, `isort` for style and imports.

---

## 11) Performance and Scaling Tips

- Prefer **Parquet** for intermediate/processed storage (faster IO, types preserved).
- Chunked loading for very large CSVs; incremental fit for encoders if necessary.
- Cache intermediate artifacts to avoid re‑computing heavy steps.
- Use **feature_engine** library for specialized transformers (e.g., rare label grouping, discretization).

---

## 12) Deliverables and Artifacts

- Clean dataset(s) in `data_processed/` (CSV/Parquet).
- Fitted preprocessing pipeline (joblib).
- DQR report and EDA visuals.
- Configuration file(s) used for the run.
- Change log in run logs.

---

## 13) Example Results Snapshot (Illustrative)

After cleaning and feature engineering on a Titanic‑like dataset:

- Missingness reduced from 19.6% → 0.0% (post‑imputation).
- Numeric skew addressed (log transform on `Fare`).
- Outliers clipped at 1st/99th percentiles (2.3% values affected).
- 3 PCA components retained explaining **92.1%** variance.
- Final feature matrix: 24 columns (from 11 original columns after encoding and selection).

> These numbers will vary for your dataset; they serve as documentation targets for your own runs.

---

## 14) Contributing Guidelines (for teamwork)

- Branch naming: `feat/…`, `fix/…`, `refactor/…`.
- Commit messages: imperative mood (“Add imputer for Age”).
- PR checklist: tests pass, docs updated, configs validated.
- Reviews: request at least 1 approval before merge to `main`.

---

## 15) License

This project is released under the **MIT License**. See `LICENSE` for details.

---

## 16) Author

Mohammed Salem
Email: salemmohamad926@gmail.com
LinkedIn: https://www.linkedin.com/in/msalem02
GitHub: https://github.com/msalem02

---

## 17) Version History

| Version | Date        | Notes                                   |
|--------:|-------------|------------------------------------------|
| 1.0     | 2024-09-12  | Initial notebooks and simple pipeline    |
| 1.1     | 2024-10-10  | Added YAML config + PCA + DQR            |
| 1.2     | 2025-02-21  | Introduced feature selection transformers|
| 1.3     | 2025-06-03  | Logging revamp + artifacts export        |

---
