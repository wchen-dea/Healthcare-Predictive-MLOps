# Databricks notebook source
# MAGIC %md
# MAGIC # 08 — Real-Time Stream Source Ingestion
# MAGIC Continuously generates real-time rows from the engineered Silver feature
# MAGIC table and appends them into a dedicated streaming source table.

# COMMAND ----------
dbutils.widgets.text("catalog", "healthcare_catalog")
dbutils.widgets.text("schema", "healthcare_ml")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

dbutils.widgets.text("seed_table", "synthetic_sepsis_seed_features")
dbutils.widgets.text("source_table", "silver_patients_features_stream_source")
dbutils.widgets.text("rows_per_second", "5")
dbutils.widgets.text(
    "checkpoint_location",
    "/tmp/healthcare_catalog/healthcare_ml/silver_patients_features_stream_source_checkpoint",
)
dbutils.widgets.text("trigger_processing_time", "30 seconds")
dbutils.widgets.dropdown("available_now", "false", ["true", "false"])
dbutils.widgets.text("await_termination_seconds", "0")

seed_table = dbutils.widgets.get("seed_table")
source_table = dbutils.widgets.get("source_table")
rows_per_second = int(dbutils.widgets.get("rows_per_second"))
checkpoint_location = dbutils.widgets.get("checkpoint_location")
trigger_processing_time = dbutils.widgets.get("trigger_processing_time")
available_now = dbutils.widgets.get("available_now").lower() == "true"
await_termination_seconds = int(dbutils.widgets.get("await_termination_seconds"))

# COMMAND ----------

import sys

sys.path.insert(0, "../src")

from healthcare_mlops.config import HealthcareConfig
from healthcare_mlops.inference import RealTimeSourceFeeder

config = HealthcareConfig(catalog=catalog, schema=schema)
feeder = RealTimeSourceFeeder(spark, config)

# COMMAND ----------

query = feeder.start(
    seed_table=seed_table,
    source_table=source_table,
    rows_per_second=rows_per_second,
    checkpoint_location=checkpoint_location,
    trigger_processing_time=trigger_processing_time,
    available_now=available_now,
)

print(f"Started source-ingestion query id={query.id} name={query.name}")
print(f"Writing source stream rows to {catalog}.{schema}.{source_table}")

# COMMAND ----------

if await_termination_seconds > 0:
    print(f"Waiting up to {await_termination_seconds} seconds...")
    query.awaitTermination(await_termination_seconds)
else:
    print("Waiting without timeout. Interrupt the notebook to stop the stream.")
    query.awaitTermination()

if query.exception() is not None:
    raise query.exception()

print(f"Source-ingestion query active: {query.isActive}")
