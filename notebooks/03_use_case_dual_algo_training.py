# Databricks notebook source
# MAGIC %md
# MAGIC # 10 — Use-Case Model Training (Dual Algorithm)
# MAGIC Trains and evaluates **two algorithms** (`random_forest`, `gradient_boosting`)
# MAGIC for each use case (`batch`, `realtime`) and promotes the best model per use case
# MAGIC to the `champion` alias in Unity Catalog.

# COMMAND ----------
dbutils.widgets.text("catalog", "healthcare_catalog")
dbutils.widgets.text("schema", "healthcare_ml")

dbutils.widgets.text(
    "batch_model_name", "healthcare_catalog.healthcare_ml.test_result_classifier_batch"
)
dbutils.widgets.text(
    "realtime_model_name", "healthcare_catalog.healthcare_ml.test_result_classifier_realtime"
)
dbutils.widgets.text(
    "experiment_root", "/Shared/healthcare-predictive-mlops/dev/use-case-models"
)
dbutils.widgets.text("algorithms", "random_forest,gradient_boosting")
dbutils.widgets.text("min_accuracy_threshold", "0.75")
dbutils.widgets.dropdown("compare_champion", "true", ["true", "false"])
dbutils.widgets.text("n_estimators", "200")
dbutils.widgets.text("max_depth", "10")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
batch_model_name = dbutils.widgets.get("batch_model_name")
realtime_model_name = dbutils.widgets.get("realtime_model_name")
experiment_root = dbutils.widgets.get("experiment_root")
algorithms = [a.strip() for a in dbutils.widgets.get("algorithms").split(",") if a.strip()]
min_accuracy = float(dbutils.widgets.get("min_accuracy_threshold"))
compare_champion = dbutils.widgets.get("compare_champion").lower() == "true"
n_estimators = int(dbutils.widgets.get("n_estimators"))
max_depth = int(dbutils.widgets.get("max_depth"))

# COMMAND ----------

import sys

sys.path.insert(0, "../src")

from healthcare_mlops.config import HealthcareConfig
from healthcare_mlops.evaluate import ModelEvaluator
from healthcare_mlops.train import ModelTrainer

config = HealthcareConfig(
    catalog=catalog,
    schema=schema,
    n_estimators=n_estimators,
    max_depth=max_depth,
)

silver_df = spark.read.table(config.silver_full_name)
print(f"Loaded training dataset rows={silver_df.count()}")

evaluator = ModelEvaluator(spark, config)
use_case_models = {
    "batch": batch_model_name,
    "realtime": realtime_model_name,
}

summary_rows = []

for use_case, model_name in use_case_models.items():
    use_case_experiment = f"{experiment_root}/{use_case}"
    print(f"\n=== Use case: {use_case} | model: {model_name} ===")

    for algorithm in algorithms:
        print(f"\nTraining algorithm={algorithm}")
        trainer_cfg = HealthcareConfig(
            catalog=catalog,
            schema=schema,
            model_algorithm=algorithm,
            n_estimators=n_estimators,
            max_depth=max_depth,
        )
        trainer = ModelTrainer(spark, trainer_cfg)

        run_id = trainer.train(
            silver_df,
            experiment_name=use_case_experiment,
            registered_model_name=model_name,
        )

        metrics = evaluator.evaluate(
            silver_df,
            model_name=model_name,
            experiment_name=use_case_experiment,
        )

        promoted = evaluator.promote(
            model_name=model_name,
            model_version=metrics["model_version"],
            alias="champion",
            min_accuracy=min_accuracy,
            metrics=metrics,
            compare_champion=compare_champion,
        )

        result = {
            "use_case": use_case,
            "algorithm": algorithm,
            "model_name": model_name,
            "model_version": metrics["model_version"],
            "accuracy": round(metrics["accuracy"], 4),
            "macro_f1": round(metrics["macro_f1"], 4),
            "run_id": run_id,
            "promoted": promoted,
        }
        summary_rows.append(result)
        print(result)

# COMMAND ----------

summary_df = spark.createDataFrame(summary_rows)
display(summary_df)

dbutils.jobs.taskValues.set(key="use_case_model_training_completed", value="true")
