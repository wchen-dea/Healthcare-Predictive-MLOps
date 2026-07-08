# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Model Evaluation & Registration
# MAGIC Evaluates the latest registered model version and promotes it to the **challenger** alias
# MAGIC if accuracy meets the configured threshold.

# COMMAND ----------

dbutils.widgets.text("catalog", "healthcare_catalog")
dbutils.widgets.text("schema", "healthcare_ml")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")



dbutils.widgets.text(
    "model_name", "healthcare_catalog.healthcare_ml.test_result_classifier"
)
dbutils.widgets.text(
    "experiment_name", "/Shared/healthcare-predictive-mlops/dev/test-result-classifier"
)
dbutils.widgets.text("min_accuracy_threshold", "0.75")
dbutils.widgets.text("target_alias", "challenger")
dbutils.widgets.text("compare_champion", "false")


model_name = dbutils.widgets.get("model_name")
experiment_name = dbutils.widgets.get("experiment_name")
min_accuracy = float(dbutils.widgets.get("min_accuracy_threshold"))
target_alias = dbutils.widgets.get("target_alias")
compare_champion = dbutils.widgets.get("compare_champion").lower() == "true"

# COMMAND ----------

import sys

sys.path.insert(0, "../src")

from healthcare_mlops.config import HealthcareConfig
from healthcare_mlops.evaluate import ModelEvaluator

config = HealthcareConfig(catalog=catalog, schema=schema)
evaluator = ModelEvaluator(spark, config)

# COMMAND ----------

silver_df = spark.read.table(config.silver_full_name)
metrics = evaluator.evaluate(silver_df, model_name, experiment_name=experiment_name)

print(f"Accuracy       : {metrics['accuracy']:.4f}")
print(f"Macro F1       : {metrics['macro_f1']:.4f}")
print(f"Model version  : {metrics['model_version']}")
print(f"Eval run ID    : {metrics['eval_run_id']}")

# COMMAND ----------

promoted = evaluator.promote(
    model_name=model_name,
    model_version=metrics["model_version"],
    alias=target_alias,
    min_accuracy=min_accuracy,
    metrics=metrics,
    compare_champion=compare_champion,
)

# Surface result to downstream tasks via task values
dbutils.jobs.taskValues.set(key="model_registered", value=str(promoted).lower())
dbutils.jobs.taskValues.set(key="model_version", value=metrics["model_version"])
dbutils.jobs.taskValues.set(key="accuracy", value=str(round(metrics["accuracy"], 4)))
