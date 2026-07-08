# Healthcare Predictive MLOps

> **⚠️ Trial Project** - For evaluation and demonstration purposes only.  
> Not licensed for production use, redistribution, or commercial deployment.  
> © 2026 wchen-dea. All rights reserved.

End-to-end MLOps implementation on Databricks for classifying patient test results
(Normal / Abnormal / Inconclusive), with both batch and real-time inference paths.

The real-time inference path is designed as an operational streaming system for
early deterioration and sepsis-risk monitoring patterns, where new feature rows
are continuously generated, ingested, scored, and persisted for downstream
clinical alerting workflows.

At a high level, the real-time path includes:

- Dedicated realtime model registry and champion alias management
  (`test_result_classifier_realtime@champion`)
- Synthetic seed bootstrap to remove dependency on pre-existing Silver refresh cycles
- Continuous source ingestion into a streaming-ready Delta source table
- Structured Streaming model scoring into a Gold risk table for operational consumers

This separation from batch inference allows independent model promotion,
independent operational scaling, and independent runtime controls (for example,
continuous jobs, checkpointing, and processing-time triggers).

The current implementation supports:

- A legacy single-model pipeline (`test_result_classifier`) via `ml_pipeline_workflow`
- A use-case model strategy with separate registries for batch and realtime:
  - `test_result_classifier_batch`
  - `test_result_classifier_realtime`
- Dual-algorithm training (`random_forest` and `gradient_boosting`) per use case

---

## Architecture

```text
Raw CSV (Volume)
    |
    v
01_ingest_raw_data -> Bronze Delta table
    |
    v
02_feature_engineering -> Silver Delta table
    |
    +--> Legacy path
    |    09_model_training + 10_model_evaluation -> test_result_classifier
    |    07_batch_inference -> patient_predictions
    |
    +--> Use-case path (recommended)
       03_use_case_dual_algo_training
             - train random_forest + gradient_boosting for batch model registry
             - train random_forest + gradient_boosting for realtime model registry
             - promote best per use case to champion

       04_seed_table_bootstrap -> synthetic_sepsis_seed_features
       05_stream_source_ingestion -> silver_patients_features_stream_source
       06_streaming_inference -> patient_sepsis_risk_stream
```

---

## Project Structure

```text
├── databricks.yml
├── pyproject.toml
├── Makefile
├── data/
│   └── healthcare_dataset.csv
├── environments/
│   ├── dev.yml
│   ├── stg.yml
│   └── prod.yml
├── notebooks/
│   ├── 01_ingest_raw_data.py
│   ├── 02_feature_engineering.py
│   ├── 03_use_case_dual_algo_training.py
│   ├── 04_seed_table_bootstrap.py
│   ├── 05_stream_source_ingestion.py
│   ├── 06_streaming_inference.py
│   ├── 07_batch_inference.py
│   ├── 08_data_quality.py
│   ├── 09_model_training.py
│   └── 10_model_evaluation.py
├── resources/
│   ├── ml_pipeline_workflow.yml
│   ├── feature_engineering_job.yml
│   ├── model_training_job.yml
│   ├── use_case_model_training_job.yml
│   ├── batch_inference_job.yml
│   ├── streaming_inference_job.yml
│   ├── stream_source_ingestion_job.yml
│   └── seed_table_bootstrap_job.yml
├── scripts/
│   └── upload_data.sh
├── src/healthcare_mlops/
│   ├── config.py
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

Configuration is defined in:

- `databricks.yml` (global defaults)
- `environments/*.yml` (target overrides)
- `src/healthcare_mlops/config.py` (Python-side defaults)

Important notes:

- Global defaults use `catalog=healthcare_catalog`
- Dev target currently overrides to `catalog=quickstart_catalog` in `environments/dev.yml`
- Use `${var.catalog}` and `${var.schema}` in job configs, so behavior follows target overrides

---

## Models And Registries

Algorithms implemented:

- `random_forest` (`RandomForestClassifier`)
- `gradient_boosting` (`GradientBoostingClassifier`)

Model registries in current implementation:

- Legacy registry: `test_result_classifier`
- Batch use-case registry: `test_result_classifier_batch`
- Realtime use-case registry: `test_result_classifier_realtime`

Promotion behavior:

- Evaluation gates on `min_accuracy_threshold` (default 0.75)
- Per-use-case best candidate can be promoted to `champion`

---

## Jobs

Current Databricks jobs under `resources/`:

- `feature_engineering_job`
- `model_training_job` (legacy single-model training/eval)
- `ml_pipeline_workflow` (legacy full orchestration)
- `batch_inference_job` (uses `test_result_classifier_batch@champion`)
- `use_case_model_training_job` (trains both algorithms for batch + realtime registries)
- `seed_table_bootstrap_job` (creates synthetic seed table)
- `stream_source_ingestion_job` (feeds streaming source table)
- `sepsis_streaming_inference_job` (uses `test_result_classifier_realtime@champion`)

---

## Quick Start

```bash
# 1) Install dependencies
make install-dev

# 2) Upload data to Databricks Volume
DATABRICKS_HOST=<host> DATABRICKS_TOKEN=<token> make upload-data

# 3) Validate and deploy to dev
make validate-dev
make deploy-dev
```

### Recommended Use-Case Run Order (Current Implementation)

```bash
make run-usecase-training TARGET=dev
make run-seed-bootstrap TARGET=dev
make run-stream-source-ingestion TARGET=dev
make run-streaming-inference TARGET=dev
make run-inference TARGET=dev
```

### Legacy Full Pipeline (Still Available)

```bash
make run TARGET=dev
```

---

## Makefile Targets

```text
make help
make install
make install-dev
make lint
make format
make test
make test-cov
make validate[-dev/-stg/-prod]
make deploy[-dev/-stg/-prod]
make run[-pipeline/-feature-engineering/-training/-inference/-streaming-inference]
make run-usecase-training
make run-seed-bootstrap
make run-stream-source-ingestion
make upload-data
make destroy[-dev/-stg/-prod]
make ci
make cd-dev / cd-stg / cd-prod
```

---

## Realtime Sepsis Pattern

This repository models a realtime healthcare pattern for early sepsis-risk monitoring:

- Seed bootstrap (`04_seed_table_bootstrap.py`)
- Stream source generation (`05_stream_source_ingestion.py`)
- Streaming scoring (`06_streaming_inference.py`)

Scoring uses the realtime use-case champion model:

- `${catalog}.${schema}.test_result_classifier_realtime@champion`

Output table:

- `${catalog}.${schema}.patient_sepsis_risk_stream`

Background references:

- Johns Hopkins Engineering for Professionals: [AI in healthcare applications and impact](https://ep.jhu.edu/news/ai-in-healthcare-applications-and-impact/)
- KMS Technology: [machine learning applications in healthcare](https://kms-technology.com/blog/machine-learning-applications-in-healthcare/)

---

## Data Upload

```bash
./scripts/upload_data.sh

# Override catalog/schema if needed
./scripts/upload_data.sh --catalog quickstart_catalog --schema healthcare_ml
```

---

## CI/CD (GitHub Actions)

Workflow files:

- `.github/workflows/ci.yml`
- `.github/workflows/cd.yml`

Current triggers (as implemented):

- CI:
  - pull_request to `qa` and `develop`
  - push to `develop`
- CD:
  - push to `prod` (deploy dev job in current workflow)
  - published release (deploy dev -> stg -> prod)

Required secrets:

- `DATABRICKS_HOST_DEV`, `DATABRICKS_TOKEN_DEV`
- `DATABRICKS_HOST_STG`, `DATABRICKS_TOKEN_STG`
- `DATABRICKS_HOST_PROD`, `DATABRICKS_TOKEN_PROD`

---

## Development

```bash
make lint
make format
make test
make test-cov
```

Testing details:

- `make test` runs with `uv` and dev extras via:
  - `PYTHONPATH=src uv run --extra dev python -m pytest tests/`

Notebook linting details:

- Ruff ignores `E402` and `F821` for `notebooks/*.py` because Databricks injects runtime symbols such as `spark`, `dbutils`, and `display`.
