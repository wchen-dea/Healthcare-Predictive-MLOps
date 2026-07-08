# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — Real-Time Inference (Early Sepsis Detection Pattern)
# MAGIC
# MAGIC This notebook demonstrates a real-time healthcare AI pattern used for
# MAGIC early sepsis detection in ICUs and emergency departments.
# MAGIC
# MAGIC How it works in this project:
# MAGIC - Continuous monitoring: stream feature rows from the Silver table.
# MAGIC - Real-time analysis: score each micro-batch with the champion model.
# MAGIC - Automated alerting foundation: write fresh risk-oriented predictions
# MAGIC   to a Gold Delta table for downstream alerting workflows.
# MAGIC
# MAGIC Note: this is a demo pattern for MLOps architecture and operations,
# MAGIC not a clinical-grade sepsis model.

# COMMAND ----------
dbutils.widgets.text("catalog", "healthcare_catalog")
dbutils.widgets.text("schema", "healthcare_ml")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

dbutils.widgets.text(
    "model_name", "healthcare_catalog.healthcare_ml.test_result_classifier_realtime"
)
dbutils.widgets.text("model_alias", "champion")
dbutils.widgets.text("source_table", "silver_patients_features_stream_source")
dbutils.widgets.text("output_table", "patient_sepsis_risk_stream")
dbutils.widgets.text(
    "checkpoint_location",
    "/tmp/healthcare_catalog/healthcare_ml/patient_sepsis_risk_stream_checkpoint",
)
dbutils.widgets.text("trigger_processing_time", "1 minute")
dbutils.widgets.dropdown("available_now", "false", ["true", "false"])
dbutils.widgets.text("await_termination_seconds", "300")

model_name = dbutils.widgets.get("model_name")
model_alias = dbutils.widgets.get("model_alias")
source_table = dbutils.widgets.get("source_table")
output_table = dbutils.widgets.get("output_table")
checkpoint_location = dbutils.widgets.get("checkpoint_location")
trigger_processing_time = dbutils.widgets.get("trigger_processing_time")
available_now = dbutils.widgets.get("available_now").lower() == "true"
await_termination_seconds = int(dbutils.widgets.get("await_termination_seconds"))

# COMMAND ----------

import sys

sys.path.insert(0, "../src")

from healthcare_mlops.config import HealthcareConfig
from healthcare_mlops.inference import StreamingPredictor

config = HealthcareConfig(catalog=catalog, schema=schema)
predictor = StreamingPredictor(spark, config)

# COMMAND ----------

query = predictor.start(
    source_table=source_table,
    output_table=output_table,
    model_name=model_name,
    alias=model_alias,
    checkpoint_location=checkpoint_location,
    trigger_processing_time=trigger_processing_time,
    available_now=available_now,
)

print(f"Started streaming query id={query.id} name={query.name}")
print(f"Writing predictions to {catalog}.{schema}.{output_table}")

# COMMAND ----------

if await_termination_seconds > 0:
    print(f"Waiting up to {await_termination_seconds} seconds...")
    query.awaitTermination(await_termination_seconds)
else:
    print("Waiting without timeout. Interrupt the notebook to stop the stream.")
    query.awaitTermination()

if query.exception() is not None:
    raise query.exception()

print(f"Streaming query active: {query.isActive}")
