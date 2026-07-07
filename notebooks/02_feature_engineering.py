# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Feature Engineering (Silver Layer)
# MAGIC Cleans Bronze patient data and engineers ML-ready features into a Silver Delta table.

# COMMAND ----------

dbutils.widgets.text("catalog", "healthcare_catalog")
dbutils.widgets.text("schema", "healthcare_ml")


catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")


# COMMAND ----------

import sys

sys.path.insert(0, "../src")

from healthcare_mlops.config import HealthcareConfig
from healthcare_mlops.data_loader import DataLoader
from healthcare_mlops.feature_engineering import FeatureEngineer

config = HealthcareConfig(catalog=catalog, schema=schema)
loader = DataLoader(spark, config)
engineer = FeatureEngineer(spark, config)

# COMMAND ----------

bronze_df = loader.read_bronze()
print(f"Bronze row count : {bronze_df.count()}")

# COMMAND ----------

silver_df = engineer.run(bronze_df)

display(silver_df.limit(10))
print(f"Silver row count : {silver_df.count()}")

# COMMAND ----------

# Label distribution
display(
    silver_df.groupBy(config.label_column).count().orderBy("count", ascending=False)
)
