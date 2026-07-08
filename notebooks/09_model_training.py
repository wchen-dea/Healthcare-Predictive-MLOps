# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Model Training
# MAGIC Trains a Random Forest classifier to predict patient test results.
# MAGIC Uses MLflow for experiment tracking and registers the model in Unity Catalog.

# COMMAND ----------
dbutils.widgets.text("catalog", "healthcare_catalog")
dbutils.widgets.text("schema", "healthcare_ml")


catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")



dbutils.widgets.text(
    "experiment_name", "/Volumes/quickstart_catalog/healthcare_ml/raw/test_result_classifier/"
)
dbutils.widgets.text(
    "model_name", "healthcare_catalog.healthcare_ml.test_result_classifier"
)
dbutils.widgets.dropdown("model_algorithm", "random_forest", ["random_forest", "gradient_boosting"])
dbutils.widgets.text("n_estimators", "200")
dbutils.widgets.text("max_depth", "10")


experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")
model_algorithm = dbutils.widgets.get("model_algorithm")
n_estimators = int(dbutils.widgets.get("n_estimators"))
max_depth = int(dbutils.widgets.get("max_depth"))

# COMMAND ----------

import sys

sys.path.insert(0, "../src")

from healthcare_mlops.config import HealthcareConfig
from healthcare_mlops.train import ModelTrainer

config = HealthcareConfig(
    catalog=catalog,
    schema=schema,
    model_algorithm=model_algorithm,
    n_estimators=n_estimators,
    max_depth=max_depth,
)
trainer = ModelTrainer(spark, config)

# COMMAND ----------

silver_df = spark.read.table(config.silver_full_name)
print(f"Training on {silver_df.count()} rows")

# COMMAND ----------

run_id = trainer.train(
    silver_df,
    experiment_name,
    registered_model_name=model_name,
)

print(f"Training complete. MLflow run_id: {run_id}")

# Pass run_id downstream via task values
dbutils.jobs.taskValues.set(key="run_id", value=run_id)
