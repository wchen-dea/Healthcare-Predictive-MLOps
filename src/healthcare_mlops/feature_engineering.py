from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

from healthcare_mlops.config import HealthcareConfig


class FeatureEngineer:
    """Transforms Bronze patient data into Silver ML-ready feature table."""

    def __init__(self, spark: SparkSession, config: HealthcareConfig):
        self.spark = spark
        self.config = config

    def clean(self, df: DataFrame) -> DataFrame:
        """Standardize string casing, drop nulls in critical columns."""
        critical_cols = ["name", "age", "gender", "medical_condition", "test_results"]
        df = df.dropna(subset=critical_cols)

        # Title-case name, lowercase categoricals
        df = df.withColumn("name", F.initcap(F.col("name")))
        for col in ["gender", "blood_type", "medical_condition",
                    "insurance_provider", "admission_type", "medication", "test_results"]:
            df = df.withColumn(col, F.lower(F.trim(F.col(col))))

        return df

    def engineer_features(self, df: DataFrame) -> DataFrame:
        """Create derived features from raw columns."""
        # Length of stay in days
        df = df.withColumn(
            "length_of_stay_days",
            F.datediff(
                F.to_date(F.col("discharge_date")),
                F.to_date(F.col("date_of_admission")),
            ).cast(IntegerType()),
        )

        # Month of admission (seasonality signal)
        df = df.withColumn(
            "admission_month",
            F.month(F.to_date(F.col("date_of_admission"))).cast(IntegerType()),
        )

        # Age group buckets
        df = df.withColumn(
            "age_group",
            F.when(F.col("age") < 18, "pediatric")
            .when(F.col("age") < 40, "young_adult")
            .when(F.col("age") < 65, "middle_aged")
            .otherwise("senior"),
        )

        # Billing amount as float
        df = df.withColumn("billing_amount", F.col("billing_amount").cast(DoubleType()))
        df = df.withColumn("age", F.col("age").cast(IntegerType()))
        df = df.withColumn("room_number", F.col("room_number").cast(IntegerType()))

        # Normalize target label
        df = df.withColumn(
            "test_result",
            F.when(F.col("test_results") == "normal", "Normal")
            .when(F.col("test_results") == "abnormal", "Abnormal")
            .otherwise("Inconclusive"),
        )

        return df

    def select_feature_columns(self, df: DataFrame) -> DataFrame:
        """Keep only columns needed for model training."""
        keep = (
            self.config.numeric_features
            + self.config.categorical_features
            + ["age_group", "admission_month", self.config.label_column]
        )
        return df.select(*keep)

    def write_silver(self, df: DataFrame) -> None:
        """Persist feature table as a Silver Delta table."""
        (
            df.write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(self.config.silver_full_name)
        )

    def read_silver(self) -> DataFrame:
        return self.spark.read.table(self.config.silver_full_name)

    def run(self, bronze_df: DataFrame) -> DataFrame:
        df = self.clean(bronze_df)
        df = self.engineer_features(df)
        df = self.select_feature_columns(df)
        self.write_silver(df)
        return df
