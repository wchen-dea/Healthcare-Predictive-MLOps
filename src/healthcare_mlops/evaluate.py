import mlflow
import mlflow.data
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from pyspark.sql import DataFrame, SparkSession

from healthcare_mlops.config import HealthcareConfig


class ModelEvaluator:
    """Manages the full MLflow ML lifecycle: evaluation, comparison, promotion, and archival."""

    def __init__(self, spark: SparkSession, config: HealthcareConfig):
        self.spark = spark
        self.config = config
        self.client = MlflowClient()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_latest_version(self, model_name: str) -> str:
        versions = self.client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise ValueError(f"No versions found for model '{model_name}'")
        return sorted(versions, key=lambda v: int(v.version), reverse=True)[0].version

    def _get_alias_version(self, model_name: str, alias: str) -> str | None:
        """Return the model version currently assigned to an alias, or None."""
        try:
            return self.client.get_model_version_by_alias(model_name, alias).version
        except Exception:
            return None

    def _get_run_metrics(self, run_id: str) -> dict:
        """Fetch logged metrics from an existing MLflow run."""
        run = self.client.get_run(run_id)
        return run.data.metrics

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        silver_df: DataFrame,
        model_name: str,
        model_version: str | None = None,
        experiment_name: str | None = None,
    ) -> dict:
        """Evaluate a registered model version using mlflow.evaluate() and log all results.

        Uses the built-in MLflow classifier evaluator which automatically computes
        accuracy, F1, precision, recall, confusion matrix, and ROC curves, and logs
        them — along with the evaluation dataset — into a dedicated MLflow run.
        """
        if model_version is None:
            model_version = self._get_latest_version(model_name)

        model_uri = f"models:/{model_name}/{model_version}"

        pdf: pd.DataFrame = silver_df.toPandas()
        # mlflow.evaluate expects predictions and ground-truth in the same DataFrame
        eval_df = pdf[
            self.config.numeric_features
            + self.config.categorical_features
            + ["age_group", "admission_month", self.config.label_column]
        ].copy()

        # Encode target to integers so predictions (int) and targets match types
        eval_df[self.config.label_column] = (
            eval_df[self.config.label_column].map(self.config.label_map)
        )

        if experiment_name:
            mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"evaluate-v{model_version}") as run:
            mlflow.set_tags(
                {
                    "model_name": model_name,
                    "model_version": model_version,
                    "stage": "evaluation",
                    "project": "healthcare-predictive-mlops",
                }
            )

            # Log the evaluation dataset for lineage tracking
            eval_dataset = mlflow.data.from_pandas(
                eval_df,
                name=f"{self.config.schema}.{self.config.silver_table}",
                targets=self.config.label_column,
            )
            mlflow.log_input(eval_dataset, context="evaluation")

            # Run the built-in MLflow classifier evaluator — logs metrics,
            # confusion matrix, per-class report, and ROC/PR curve artifacts
            eval_result = mlflow.evaluate(
                model=model_uri,
                data=eval_df,
                targets=self.config.label_column,
                model_type="classifier",
                evaluators="default",
            )

            run_id = run.info.run_id

        # Surface key scalars for downstream use
        metrics = {
            "accuracy": eval_result.metrics.get("accuracy_score", 0.0),
            "macro_f1": eval_result.metrics.get("f1_score", 0.0),
            "model_version": model_version,
            "eval_run_id": run_id,
            "all_metrics": eval_result.metrics,
        }

        # Attach eval run id to the model version for traceability
        self.client.set_model_version_tag(model_name, model_version, "eval_run_id", run_id)

        return metrics

    # ── Champion comparison ───────────────────────────────────────────────────

    def compare_with_champion(
        self,
        challenger_metrics: dict,
        model_name: str,
        champion_alias: str = "champion",
    ) -> bool:
        """Return True if the challenger outperforms the current champion.

        Falls back to True when no champion exists (first deployment).
        """
        champion_version = self._get_alias_version(model_name, champion_alias)
        if champion_version is None:
            print("No current champion found — challenger will be promoted by default.")
            return True

        champion_eval_run_id = self.client.get_model_version(
            model_name, champion_version
        ).tags.get("eval_run_id")

        if champion_eval_run_id is None:
            print("Champion has no eval_run_id tag — skipping comparison.")
            return True

        champion_metrics = self._get_run_metrics(champion_eval_run_id)
        champion_acc = champion_metrics.get("accuracy_score", 0.0)
        challenger_acc = challenger_metrics.get("accuracy", 0.0)

        print(
            f"Champion v{champion_version} accuracy : {champion_acc:.4f}\n"
            f"Challenger v{challenger_metrics['model_version']} accuracy : {challenger_acc:.4f}"
        )
        return challenger_acc > champion_acc

    # ── Promotion & archival ──────────────────────────────────────────────────

    def promote(
        self,
        model_name: str,
        model_version: str,
        alias: str,
        min_accuracy: float | None = None,
        metrics: dict | None = None,
        compare_champion: bool = False,
    ) -> bool:
        """Assign *alias* to *model_version*, with optional accuracy gate and champion comparison.

        When *alias* is 'champion', the previous champion is automatically moved to
        the 'archived' alias so the registry retains a clean promotion history.
        """
        # ── Gate 1: minimum accuracy threshold ───────────────────────────────
        if min_accuracy is not None and metrics is not None:
            acc = metrics.get("accuracy", 0.0)
            if acc < min_accuracy:
                print(f"Accuracy {acc:.4f} < threshold {min_accuracy:.4f}. Rejecting.")
                self.client.set_model_version_tag(
                    model_name, model_version, "promotion_status", "rejected"
                )
                self.client.set_model_version_tag(
                    model_name, model_version, "rejection_reason",
                    f"accuracy {acc:.4f} < threshold {min_accuracy:.4f}",
                )
                return False

        # ── Gate 2: must beat current champion ───────────────────────────────
        if compare_champion and metrics is not None:
            if not self.compare_with_champion(metrics, model_name, champion_alias=alias):
                print("Challenger did not outperform champion. Skipping promotion.")
                self.client.set_model_version_tag(
                    model_name, model_version, "promotion_status", "rejected"
                )
                self.client.set_model_version_tag(
                    model_name, model_version, "rejection_reason",
                    "challenger accuracy did not exceed champion",
                )
                return False

        # ── Archive the previous holder of this alias ─────────────────────────
        if alias == self.config.champion_alias:
            prev_version = self._get_alias_version(model_name, alias)
            if prev_version and prev_version != model_version:
                self.client.set_registered_model_alias(
                    model_name, "archived", prev_version
                )
                self.client.update_model_version(
                    name=model_name,
                    version=prev_version,
                    description=f"Archived — superseded by v{model_version}.",
                )
                print(f"Archived previous champion v{prev_version}.")

        # ── Assign alias and update registry metadata ─────────────────────────
        self.client.set_registered_model_alias(model_name, alias, model_version)

        description_parts = [f"Promoted to '{alias}'."]
        if metrics:
            description_parts.append(
                f"accuracy={metrics.get('accuracy', 0.0):.4f}, "
                f"macro_f1={metrics.get('macro_f1', 0.0):.4f}"
            )
        self.client.update_model_version(
            name=model_name,
            version=model_version,
            description=" ".join(description_parts),
        )

        # Tag the version with promotion metadata
        self.client.set_model_version_tag(model_name, model_version, "promotion_status", "promoted")
        self.client.set_model_version_tag(model_name, model_version, "alias", alias)
        if metrics:
            self.client.set_model_version_tag(
                model_name, model_version, "eval_accuracy",
                str(round(metrics.get("accuracy", 0.0), 4)),
            )
            if "eval_run_id" in metrics:
                self.client.set_model_version_tag(
                    model_name, model_version, "eval_run_id", metrics["eval_run_id"]
                )

        print(f"Assigned alias '{alias}' to {model_name} v{model_version}.")
        return True

