import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from healthcare_mlops.config import HealthcareConfig


class BatchPredictor:
    """Loads the champion model and scores new patient records at scale."""

    def __init__(self, spark: SparkSession, config: HealthcareConfig):
        self.spark = spark
        self.config = config

    def _load_model_udf(self, model_name: str, alias: str):
        """Return a Pandas UDF that wraps the registered model."""
        import mlflow
        from pyspark.sql.functions import pandas_udf

        model_uri = f"models:/{model_name}@{alias}"
        model = mlflow.sklearn.load_model(model_uri)
        inv_label_map = {v: k for k, v in self.config.label_map.items()}

        broadcast_model = self.spark.sparkContext.broadcast(model)
        broadcast_map = self.spark.sparkContext.broadcast(inv_label_map)

        feature_cols = (
            self.config.numeric_features
            + self.config.categorical_features
            + ["age_group", "admission_month"]
        )

        @pandas_udf(StringType())
        def predict_udf(*cols: pd.Series) -> pd.Series:
            pdf = pd.concat(list(cols), axis=1)
            pdf.columns = feature_cols
            preds_encoded = broadcast_model.value.predict(pdf)
            preds = [broadcast_map.value.get(p, str(p)) for p in preds_encoded]
            return pd.Series(preds)

        return predict_udf, feature_cols

    def predict(
        self,
        input_df: DataFrame,
        model_name: str,
        alias: str = "champion",
    ) -> DataFrame:
        """Apply the model to input_df and return a DataFrame with predictions."""
        predict_udf, feature_cols = self._load_model_udf(model_name, alias)

        result_df = input_df.withColumn(
            "predicted_test_result",
            predict_udf(*[F.col(c) for c in feature_cols]),
        ).withColumn("prediction_timestamp", F.current_timestamp())

        return result_df

    def write_gold(self, predictions_df: DataFrame, output_table: str) -> None:
        """Persist predictions to a Gold Delta table."""
        full_table = f"{self.config.catalog}.{self.config.schema}.{output_table}"
        (
            predictions_df.write.format("delta")
            .mode("append")
            .option("mergeSchema", "true")
            .saveAsTable(full_table)
        )
