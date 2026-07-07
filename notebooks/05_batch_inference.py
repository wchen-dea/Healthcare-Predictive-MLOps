# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Batch Inference (Gold Layer)
# MAGIC Loads the **champion** model from Unity Catalog and scores the Silver feature table,
# MAGIC writing predictions to the Gold Delta table.

# COMMAND ----------
dbutils.widgets.text("catalog", "healthcare_catalog")
dbutils.widgets.text("schema", "healthcare_ml")


catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")


dbutils.widgets.text(
    "model_name", "healthcare_catalog.healthcare_ml.test_result_classifier"
)
dbutils.widgets.text("model_alias", "champion")
dbutils.widgets.text("output_table", "patient_predictions")


model_name = dbutils.widgets.get("model_name")
model_alias = dbutils.widgets.get("model_alias")
output_table = dbutils.widgets.get("output_table")

# COMMAND ----------

import sys

sys.path.insert(0, "../src")

from healthcare_mlops.config import HealthcareConfig
from healthcare_mlops.inference import BatchPredictor

config = HealthcareConfig(catalog=catalog, schema=schema)
predictor = BatchPredictor(spark, config)

# COMMAND ----------

silver_df = spark.read.table(config.silver_full_name)
print(f"Scoring {silver_df.count()} records...")

# COMMAND ----------

predictions_df = predictor.predict(silver_df, model_name, alias=model_alias)

display(
    predictions_df.select(
        "age",
        "gender",
        "medical_condition",
        "admission_type",
        "test_result",
        "predicted_test_result",
        "prediction_timestamp",
    ).limit(20)
)

# COMMAND ----------

predictor.write_gold(predictions_df, output_table)
print(f"Predictions written to {catalog}.{schema}.{output_table}")
