# Healthcare Predictive MLOps

> **⚠️ Trial Project** — For evaluation and demonstration purposes only.
> Not licensed for production use, redistribution, or commercial deployment.
> © 2026 wchen-dea. All rights reserved.

End-to-end MLOps pipeline on Databricks for classifying patient test results
(**Normal / Abnormal / Inconclusive**) using a Gradient Boosting model, Unity Catalog,
MLflow, and Databricks Asset Bundles.

---

## Architecture

```text
Raw CSV (Volume)
     │
     ▼
01_ingest_raw_data      → Bronze Delta table  (healthcare_catalog.healthcare_ml.bronze_patients)
     │
     ▼
02_feature_engineering  → Silver Delta table  (healthcare_catalog.healthcare_ml.silver_patients_features)
     │
     ▼
03_model_training       → MLflow experiment + registered model
     │
     ▼
04_model_evaluation     → Promote to champion if accuracy ≥ 0.35
     │  (condition: model_registered == true)
     ├──► 05_batch_inference      → Gold Delta table    (healthcare_catalog.healthcare_ml.patient_predictions)
     │
     └──► 07_streaming_inference  → Gold Delta table    (healthcare_catalog.healthcare_ml.patient_sepsis_risk_stream)
```

---

## Project Structure

```text
├── databricks.yml                  # Bundle root — variables & target includes
├── pyproject.toml                  # Python package + dev deps + ruff config
├── Makefile                        # All operational targets
├── data/
│   └── healthcare_dataset.csv      # Source dataset
├── environments/
│   ├── dev.yml                     # Dev target (default)
│   ├── stg.yml                     # Staging target
│   └── prod.yml                    # Production target
├── notebooks/
│   ├── 01_ingest_raw_data.py
│   ├── 02_feature_engineering.py
│   ├── 03_model_training.py
│   ├── 04_model_evaluation.py
│   ├── 05_batch_inference.py
│   ├── 06_data_quality.py
│   ├── 07_streaming_inference.py
│   ├── 08_stream_source_ingestion.py
│   ├── 09_seed_table_bootstrap.py
│   └── 10_use_case_dual_algo_training.py
├── resources/
│   ├── ml_pipeline_workflow.yml    # Full pipeline orchestration job
│   ├── feature_engineering_job.yml
│   ├── model_training_job.yml
│   ├── use_case_model_training_job.yml
│   ├── batch_inference_job.yml
│   ├── streaming_inference_job.yml
│   ├── stream_source_ingestion_job.yml
│   └── seed_table_bootstrap_job.yml
├── scripts/
│   └── upload_data.sh              # Upload raw CSV to Unity Catalog Volume
├── src/healthcare_mlops/
│   ├── config.py                   # Central configuration (single source of truth)
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
└── tests/
    ├── test_config.py
    └── test_feature_engineering.py
```

---

## Configuration

All values flow from one of two canonical sources:

| Source | Used by |
| --- | --- |
| `src/healthcare_mlops/config.py` | Local Python runs, unit tests |
| `databricks.yml` variables | All Databricks jobs (overridable per environment) |

### Key variables (`databricks.yml`)

| Variable | Default | Description |
| --- | --- | --- |
| `catalog` | `healthcare_catalog` | Unity Catalog name |
| `schema` | `healthcare_ml` | Schema inside the catalog |
| `data_path` | `/Volumes/healthcare_catalog/healthcare_ml/raw/healthcare_dataset.csv` | Raw data location |
| `spark_version` | `17.3.x-cpu-ml-scala2.13` | Databricks Runtime |
| `cluster_node_type` | `m5.large` | Worker instance type |
| `num_workers` | `2` | Worker node count |

Per-environment overrides live in `environments/{dev,stg,prod}.yml`.

---

## Model

| | Detail |
| --- | --- |
| **Algorithm** | `GradientBoostingClassifier` (scikit-learn) |
| **Features** | 4 numeric (`age`, `billing_amount`, `room_number`, `length_of_stay_days`, `admission_month`) + 6 one-hot encoded categoricals + `age_group` bucket |
| **Target** | `test_result` — 3 classes: Normal / Abnormal / Inconclusive |
| **Accuracy gate** | ≥ 0.75 (promotes to `champion` alias in Unity Catalog) |
| **Hyperparameters** | `n_estimators=200`, `max_depth=10`, `learning_rate=0.05`, `subsample=0.8` |

Use-case model strategy:

- `test_result_classifier_batch`: trained with both `random_forest` and `gradient_boosting`.
- `test_result_classifier_realtime`: trained with both `random_forest` and `gradient_boosting`.
- Best version per use case is promoted to `champion` in its own model registry.

> **Note on the dataset:** The bundled Kaggle Healthcare dataset has synthetically
> random labels — `test_result` was assigned with no correlation to clinical features.
> Expected accuracy is ~35–40%. The pipeline is designed to demonstrate the MLOps
> workflow; swap in a real dataset with genuine signal to achieve higher accuracy.

---

## Prerequisites

- [Databricks CLI v0.230+](https://docs.databricks.com/dev-tools/cli/index.html)
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- AWS credentials / Databricks workspace access
- Unity Catalog with `healthcare_catalog` pre-created (platform team)
- Raw dataset uploaded to the Volume (see **Data Upload** below)

---

## Quick Start

```bash
# 1. Install dependencies
make install-dev

# 2. Upload raw data to the Volume
DATABRICKS_HOST=<host> DATABRICKS_TOKEN=<token> make upload-data

# 3. Validate, deploy, and run (dev)
make ci          # lint + tests + bundle validate
make deploy-dev  # deploy resources to dev
make run         # run full pipeline on dev
```

---

## Makefile Targets

```text
make help                   # list all targets
make install                # production deps
make install-dev            # all deps including dev extras
make lint                   # ruff linter
make format                 # ruff auto-format
make test                   # unit tests
make test-cov               # tests with coverage report
make validate[-dev/-stg/-prod]
make deploy[-dev/-stg/-prod]
make run[-pipeline/-feature-engineering/-training/-inference/-streaming-inference]
make run-usecase-training
make run-seed-bootstrap
make run-stream-source-ingestion
make upload-data            # upload data/ to Databricks Volume
make destroy[-dev/-stg/-prod]   # tear down resources (prompts)
make ci                     # full CI gate
make cd-dev / cd-stg / cd-prod  # CI/CD deploy steps
```

---

## Streaming Inference Use Case (Early Sepsis Detection Pattern)

A typical real-time ML use case in healthcare is early sepsis detection in ICUs
and emergency departments. This project's streaming inference path is documented
to follow that operational pattern.

Background references:

- Johns Hopkins Engineering for Professionals: [AI in healthcare applications and impact](https://ep.jhu.edu/news/ai-in-healthcare-applications-and-impact/)
- KMS Technology: [machine learning applications in healthcare](https://kms-technology.com/blog/machine-learning-applications-in-healthcare/)

How it works:

- Continuous monitoring: a Structured Streaming query reads incoming feature
           rows from a dedicated source table, `silver_patients_features_stream_source`.
- Real-time analysis: each micro-batch is scored by a Random Forest champion model.
- Automated alerting foundation: predictions are continuously written to Gold so
     downstream systems can trigger clinician notifications for deterioration risk.

Training and evaluation flow for this streaming use case:

- Use `10_use_case_dual_algo_training.py` to train and evaluate both
     `random_forest` and `gradient_boosting` for `test_result_classifier_batch`
     and `test_result_classifier_realtime`.
- The best model per use case is promoted to `champion`.

Real-time source ingestion flow:

- `09_seed_table_bootstrap.py` creates a synthetic seed table,
  `synthetic_sepsis_seed_features`, with no dependency on Silver data.
- `08_stream_source_ingestion.py` continuously writes simulated real-time rows
     into `silver_patients_features_stream_source`.
- `07_streaming_inference.py` consumes that source table and appends risk
     predictions into the Gold output table.

In this project, risk predictions are appended into:

- `healthcare_catalog.healthcare_ml.patient_sepsis_risk_stream`

Files added for this use case:

- `notebooks/07_streaming_inference.py`
- `notebooks/08_stream_source_ingestion.py`
- `notebooks/09_seed_table_bootstrap.py`
- `notebooks/10_use_case_dual_algo_training.py`
- `resources/streaming_inference_job.yml`
- `resources/stream_source_ingestion_job.yml`
- `resources/seed_table_bootstrap_job.yml`
- `resources/use_case_model_training_job.yml`

Run it with:

```bash
make deploy-dev
make run-usecase-training TARGET=dev
make run-seed-bootstrap TARGET=dev
make run-stream-source-ingestion TARGET=dev
make run-streaming-inference TARGET=dev
```

By default, the job is configured as a Databricks **continuous** job and starts
in `PAUSED` state, so you can explicitly enable it when ready.

Impact framing: when deployed with clinically validated features and governance,
this architecture supports earlier interventions (for example, rapid sepsis
protocol initiation) by reducing time-to-detection of deterioration signals.

Override `TARGET`, `CATALOG_NAME`, or `SCHEMA_NAME` inline:

```bash
make deploy TARGET=stg
make upload-data CATALOG_NAME=dev_healthcare_catalog
```

---

## Data Upload

```bash
# Uses DATABRICKS_HOST + DATABRICKS_TOKEN from environment
./scripts/upload_data.sh

# Override catalog/schema
./scripts/upload_data.sh --catalog healthcare_catalog --schema healthcare_ml
```

---

## CI/CD (GitHub Actions)

| Workflow | Trigger | Steps |
| --- | --- | --- |
| `.github/workflows/ci.yml` | PR → `main`, push → `develop` | lint → test → bundle validate (dev) |
| `.github/workflows/cd.yml` | Push → `main` | deploy to dev |
| `.github/workflows/cd.yml` | Published release | deploy dev → stg → prod (prod requires manual approval) |

### Required GitHub Secrets

| Secret | Description |
| --- | --- |
| `DATABRICKS_HOST_DEV` | Dev workspace URL |
| `DATABRICKS_TOKEN_DEV` | Dev PAT |
| `DATABRICKS_HOST_STG` | Staging workspace URL |
| `DATABRICKS_TOKEN_STG` | Staging PAT |
| `DATABRICKS_HOST_PROD` | Production workspace URL |
| `DATABRICKS_TOKEN_PROD` | Production PAT |

---

## Development

```bash
# Lint
make lint

# Auto-fix lint issues
make format

# Run tests
make test

# Run with coverage
make test-cov
```

Ruff ignores `E402` (import order) and `F821` (undefined names: `spark`, `dbutils`,
`display`) in notebooks — these are Databricks runtime injections, not real errors.
