# Healthcare Predictive MLOps


End-to-end MLOps implementation on Databricks for classifying patient test results
(Normal / Abnormal / Inconclusive), with both batch and real-time inference paths.

In the healthcare industry, **Real-Time ML** processes live, streaming patient data to deliver
instant predictions with sub-second latency, driving critical care and immediate clinical
decisions. Conversely, **Batch ML** ingests and computes massive volumes of historical data on
scheduled cycles (e.g., daily or weekly), prioritizing large-scale efficiency for strategic
insights.

The integration of both methodologies is fundamentally changing patient outcomes and hospital
operations.

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

## Use Case Paths

### ⏱️ Real-Time ML Path

Real-Time ML focuses on clinical safety, immediate intervention, and continuous monitoring. It
analyzes data as it is generated (within milliseconds to seconds) to deliver point-of-care
predictions.

- **Sepsis and Deterioration Prediction:** Continuously processes real-time vital signs from
  wearable devices or bedside monitors (e.g., via platforms like [Apache Flink](https://flink.apache.org/)
  or [Apache Spark](https://spark.apache.org/)) to alert nurses and doctors before a patient
  experiences life-threatening organ failure.
- **Hospital Capacity Optimization:** Predicts live emergency department inflows, unit-level
  patient occupancy, and bed turnover using streaming data architectures to prevent overcrowding.
- **Critical Care Alerting:** Evaluates temporal Electronic Health Records (EHR) using Fast
  Healthcare Interoperability Resources ([FHIR APIs](https://hl7.org/fhir/overview.html)) to
  surface immediate patient risk scores in emergency settings.

This project implements the real-time path as an operational streaming system for early
deterioration and sepsis-risk monitoring:

- Seed bootstrap (`04_seed_table_bootstrap.py`)
- Stream source generation (`05_stream_source_ingestion.py`)
- Streaming scoring (`06_streaming_inference.py`)

Scoring uses: `${catalog}.${schema}.test_result_classifier_realtime@champion`  
Output table: `${catalog}.${schema}.patient_sepsis_risk_stream`

### 📦 Batch ML Path

Batch ML is relied upon when processing enormous, complex, or unstructured datasets where a
slight delay is acceptable, but absolute accuracy, thoroughness, and cost-efficiency are
critical.

- **Medical Imaging and Diagnostics:** Processes large batches of MRI, CT scan, and mammogram
  datasets during scheduled intervals, enabling radiologists to identify complex patterns such
  as tumors difficult for the human eye to detect.
- **Predictive Risk Modeling & Outreach:** Analyzes massive EHR archives to generate monthly or
  weekly predictive risk scores (e.g., readmission probability, appointment no-show risk) to
  help staff plan proactive outreach.
- **Genomics and Precision Medicine:** Conducts large-scale genomic sequencing and
  bioinformatics analysis to identify genetic markers — computationally intensive workloads
  that are cost-prohibitive to run in real time.

This project implements the batch path for scheduled patient test result classification:

- Feature engineering (`02_feature_engineering.py`)
- Dual-algorithm model training (`03_use_case_dual_algo_training.py`)
- Batch scoring (`07_batch_inference.py`)

Scoring uses: `${catalog}.${schema}.test_result_classifier_batch@champion`  
Output table: `${catalog}.${schema}.patient_predictions`

### ⚖️ Technical Trade-Offs

| Feature | Real-Time ML | Batch ML |
|---|---|---|
| **Latency** | Milliseconds to seconds | Hours, days, or weeks |
| **Data Flow** | Continuous streams from unbounded data sources | Stored or historical finite data at rest |
| **Infrastructure** | Complex, always-on architecture (e.g., Kafka) | Simpler, scheduled workflows (e.g., Airflow) |
| **Cost** | Higher operational compute costs | Highly cost-efficient |

### 🔄 Hybrid Approach

Many modern health systems employ a **hybrid architecture** (often combining
[Lambda or Kappa architectures](https://hazelcast.com/blog/from-batch-machine-learning-to-real-time-machine-learning/)).
Batch ML is utilized for deep-dive historical analysis and training complex foundational models
offline. These pre-trained models are then deployed to serve real-time predictions to clinical
staff.

This project mirrors that hybrid approach: models are trained offline in batch mode and
promoted to a `champion` alias, then served by the streaming inference path for real-time
clinical scoring.

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
