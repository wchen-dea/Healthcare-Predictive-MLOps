import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.streaming import StreamingQuery
from pyspark.sql.types import StringType
from pyspark.sql.window import Window

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


class StreamingPredictor:
    """Applies champion-model scoring to a Structured Streaming input table."""

    def __init__(self, spark: SparkSession, config: HealthcareConfig):
        self.spark = spark
        self.config = config
        self.batch_predictor = BatchPredictor(spark, config)

    def start(
        self,
        source_table: str,
        output_table: str,
        model_name: str,
        alias: str = "champion",
        checkpoint_location: str = "",
        trigger_processing_time: str | None = "1 minute",
        available_now: bool = False,
    ) -> StreamingQuery:
        """Start a streaming query that appends predictions into a Delta table."""
        full_source = f"{self.config.catalog}.{self.config.schema}.{source_table}"
        full_output = f"{self.config.catalog}.{self.config.schema}.{output_table}"

        if not checkpoint_location:
            checkpoint_location = (
                f"/tmp/{self.config.catalog}/{self.config.schema}/{output_table}_checkpoint"
            )

        stream_df = self.spark.readStream.table(full_source)
        predictions_df = self.batch_predictor.predict(
            stream_df,
            model_name,
            alias=alias,
        )

        writer = (
            predictions_df.writeStream.format("delta")
            .outputMode("append")
            .option("checkpointLocation", checkpoint_location)
        )

        if available_now:
            writer = writer.trigger(availableNow=True)
        elif trigger_processing_time:
            writer = writer.trigger(processingTime=trigger_processing_time)

        return writer.toTable(full_output)


class RealTimeSourceFeeder:
    """Generates synthetic real-time rows from a seed table into a streaming source table."""

    def __init__(self, spark: SparkSession, config: HealthcareConfig):
        self.spark = spark
        self.config = config

    def start(
        self,
        seed_table: str,
        source_table: str,
        rows_per_second: int = 5,
        checkpoint_location: str = "",
        trigger_processing_time: str | None = "30 seconds",
        available_now: bool = False,
    ) -> StreamingQuery:
        """Start a rate-driven stream that appends rows into source_table."""
        full_seed = f"{self.config.catalog}.{self.config.schema}.{seed_table}"
        full_source = f"{self.config.catalog}.{self.config.schema}.{source_table}"

        if not checkpoint_location:
            checkpoint_location = (
                f"/tmp/{self.config.catalog}/{self.config.schema}/{source_table}_checkpoint"
            )

        seed_df = self.spark.read.table(full_seed)
        if seed_df.limit(1).count() == 0:
            raise ValueError(f"Seed table '{full_seed}' is empty. Cannot generate stream.")

        seeded = (
            seed_df.withColumn(
                "_stream_idx",
                F.row_number().over(Window.orderBy(F.monotonically_increasing_id())) - 1,
            )
            .withColumn("_stream_idx", F.col("_stream_idx").cast("long"))
        )
        seed_count = seeded.count()

        rate_df = self.spark.readStream.format("rate").option("rowsPerSecond", rows_per_second).load()

        stream_rows = (
            rate_df.withColumn("_stream_idx", (F.col("value") % F.lit(seed_count)).cast("long"))
            .join(F.broadcast(seeded), on="_stream_idx", how="left")
            .drop("_stream_idx", "value", "timestamp")
            .withColumn("source_ingest_ts", F.current_timestamp())
        )

        writer = (
            stream_rows.writeStream.format("delta")
            .outputMode("append")
            .option("checkpointLocation", checkpoint_location)
        )

        if available_now:
            writer = writer.trigger(availableNow=True)
        elif trigger_processing_time:
            writer = writer.trigger(processingTime=trigger_processing_time)

        return writer.toTable(full_source)


class SyntheticSeedBootstrapper:
    """Builds a synthetic feature table for streaming demos without Silver-table dependency."""

    def __init__(self, spark: SparkSession, config: HealthcareConfig):
        self.spark = spark
        self.config = config

    def generate(self, num_rows: int = 10000) -> DataFrame:
        """Generate synthetic feature rows aligned to model input columns."""
        if num_rows <= 0:
            raise ValueError("num_rows must be > 0")

        genders = F.array(F.lit("male"), F.lit("female"))
        blood_types = F.array(
            F.lit("a+"), F.lit("a-"), F.lit("b+"), F.lit("b-"), F.lit("ab+"), F.lit("ab-"), F.lit("o+"), F.lit("o-")
        )
        conditions = F.array(
            F.lit("diabetes"),
            F.lit("hypertension"),
            F.lit("obesity"),
            F.lit("asthma"),
            F.lit("cancer"),
            F.lit("arthritis"),
        )
        providers = F.array(
            F.lit("aetna"),
            F.lit("medicare"),
            F.lit("cigna"),
            F.lit("blue cross"),
            F.lit("unitedhealth"),
        )
        admission_types = F.array(F.lit("emergency"), F.lit("urgent"), F.lit("elective"))
        medications = F.array(
            F.lit("aspirin"),
            F.lit("ibuprofen"),
            F.lit("paracetamol"),
            F.lit("metformin"),
            F.lit("lisinopril"),
        )

        base = self.spark.range(num_rows)

        seeded = (
            base.withColumn("age", (F.rand(7) * 73 + 18).cast("int"))
            .withColumn("billing_amount", F.round(F.rand(11) * 45000 + 500, 2))
            .withColumn("room_number", (F.rand(13) * 499 + 1).cast("int"))
            .withColumn("length_of_stay_days", (F.rand(17) * 20 + 1).cast("int"))
            .withColumn("admission_month", (F.rand(19) * 12 + 1).cast("int"))
            .withColumn("gender", F.element_at(genders, (F.rand(23) * 2 + 1).cast("int")))
            .withColumn("blood_type", F.element_at(blood_types, (F.rand(29) * 8 + 1).cast("int")))
            .withColumn("medical_condition", F.element_at(conditions, (F.rand(31) * 6 + 1).cast("int")))
            .withColumn("insurance_provider", F.element_at(providers, (F.rand(37) * 5 + 1).cast("int")))
            .withColumn("admission_type", F.element_at(admission_types, (F.rand(41) * 3 + 1).cast("int")))
            .withColumn("medication", F.element_at(medications, (F.rand(43) * 5 + 1).cast("int")))
            .withColumn(
                "age_group",
                F.when(F.col("age") < 18, "pediatric")
                .when(F.col("age") < 40, "young_adult")
                .when(F.col("age") < 65, "middle_aged")
                .otherwise("senior"),
            )
            .withColumn(
                "test_result",
                F.when(F.rand(47) < 0.45, "Normal")
                .when(F.rand(53) < 0.75, "Inconclusive")
                .otherwise("Abnormal"),
            )
            .drop("id")
        )

        return seeded

    def write_seed_table(self, df: DataFrame, output_table: str, mode: str = "overwrite") -> None:
        """Write synthetic seed rows to a Delta table for downstream stream-source feeding."""
        full_output = f"{self.config.catalog}.{self.config.schema}.{output_table}"
        (
            df.write.format("delta")
            .mode(mode)
            .option("overwriteSchema", "true")
            .saveAsTable(full_output)
        )
