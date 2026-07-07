# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Ingest Raw Data (Bronze Layer)
# MAGIC Reads the raw healthcare CSV from a Unity Catalog Volume and writes it to a Bronze Delta table.

# COMMAND ----------

dbutils.widgets.text("catalog", "healthcare_catalog")
dbutils.widgets.text("schema", "healthcare_ml")
dbutils.widgets.text(
    "data_path", "/Volumes/healthcare_catalog/healthcare_ml/raw/healthcare_dataset.csv"
)

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
data_path = dbutils.widgets.get("data_path")

# COMMAND ----------

import sys

sys.path.insert(0, "../src")

from healthcare_mlops.config import HealthcareConfig
from healthcare_mlops.data_loader import DataLoader

config = HealthcareConfig(catalog=catalog, schema=schema)
loader = DataLoader(spark, config)

# COMMAND ----------

# Ensure schema exists (catalog is pre-created by platform team)
# spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# COMMAND ----------

raw_df = loader.read_csv(data_path)
raw_df = loader.normalize_column_names(raw_df)

display(raw_df.limit(10))
print(f"Row count: {raw_df.count()}")

# COMMAND ----------

loader.write_bronze(raw_df)
print(f"Bronze table written: {config.bronze_full_name}")
