# Databricks notebook source
# MAGIC %md
# MAGIC # 09 — Synthetic Seed Table Bootstrap
# MAGIC Generates a pure synthetic seed table (no dependency on Silver) for
# MAGIC real-time streaming demos and writes it as a Delta table.

# COMMAND ----------
dbutils.widgets.text("catalog", "healthcare_catalog")
dbutils.widgets.text("schema", "healthcare_ml")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

dbutils.widgets.text("output_table", "synthetic_sepsis_seed_features")
dbutils.widgets.text("num_rows", "10000")
dbutils.widgets.dropdown("write_mode", "overwrite", ["overwrite", "append"])

output_table = dbutils.widgets.get("output_table")
num_rows = int(dbutils.widgets.get("num_rows"))
write_mode = dbutils.widgets.get("write_mode")

# COMMAND ----------

import sys

sys.path.insert(0, "../src")

from healthcare_mlops.config import HealthcareConfig
from healthcare_mlops.inference import SyntheticSeedBootstrapper

config = HealthcareConfig(catalog=catalog, schema=schema)
bootstrapper = SyntheticSeedBootstrapper(spark, config)

# COMMAND ----------

seed_df = bootstrapper.generate(num_rows=num_rows)
bootstrapper.write_seed_table(seed_df, output_table=output_table, mode=write_mode)

full_table = f"{catalog}.{schema}.{output_table}"
count_df = spark.read.table(full_table)
print(f"Synthetic seed table ready: {full_table}")
print(f"Rows in table: {count_df.count()}")

display(count_df.limit(20))
