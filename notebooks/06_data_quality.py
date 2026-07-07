# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Data Quality & Prediction Drift Check
# MAGIC Validates the prediction output distribution and alerts if significant drift is detected
# MAGIC compared to the baseline (training-time label distribution).

# COMMAND ----------
dbutils.widgets.text("catalog", "healthcare_catalog")
dbutils.widgets.text("schema", "healthcare_ml")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")


dbutils.widgets.text("predictions_table", "patient_predictions")


predictions_table = dbutils.widgets.get("predictions_table")

# COMMAND ----------

import sys

sys.path.insert(0, "../src")

from pyspark.sql import functions as F

from healthcare_mlops.config import HealthcareConfig

config = HealthcareConfig(catalog=catalog, schema=schema)

# COMMAND ----------

gold_df = spark.read.table(f"{catalog}.{schema}.{predictions_table}")

# Latest batch only
latest_ts = gold_df.agg(F.max("prediction_timestamp")).collect()[0][0]
batch_df = gold_df.filter(F.col("prediction_timestamp") == latest_ts)

print(f"Latest prediction batch timestamp: {latest_ts}")
print(f"Batch row count: {batch_df.count()}")

# COMMAND ----------

# Prediction distribution
pred_dist = (
    batch_df.groupBy("predicted_test_result")
    .count()
    .withColumn("pct", F.round(F.col("count") / batch_df.count() * 100, 2))
    .orderBy("count", ascending=False)
)
display(pred_dist)

# COMMAND ----------

# Drift check: flag if any class exceeds 70 % or drops below 10 %
drift_flags = pred_dist.filter((F.col("pct") > 70) | (F.col("pct") < 10))

if drift_flags.count() > 0:
    print("WARNING: Potential prediction drift detected!")
    display(drift_flags)
    dbutils.jobs.taskValues.set(key="drift_detected", value="true")
else:
    print("No significant drift detected.")
    dbutils.jobs.taskValues.set(key="drift_detected", value="false")
