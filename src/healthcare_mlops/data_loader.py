from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from healthcare_mlops.config import HealthcareConfig


class DataLoader:
    """Handles ingestion of raw healthcare CSV data into Bronze Delta tables."""

    def __init__(self, spark: SparkSession, config: HealthcareConfig):
        self.spark = spark
        self.config = config

    def read_csv(self, path: str) -> DataFrame:
        """Read raw CSV from a DBFS / Unity Catalog Volume path."""
        return (
            self.spark.read.option("header", "true")
            .option("inferSchema", "true")
            .option("multiLine", "true")
            .option("escape", '"')
            .csv(path)
        )

    def normalize_column_names(self, df: DataFrame) -> DataFrame:
        """Lowercase and replace spaces with underscores in column names."""
        renamed = {c: c.lower().replace(" ", "_") for c in df.columns}
        for old, new in renamed.items():
            df = df.withColumnRenamed(old, new)
        return df

    def write_bronze(self, df: DataFrame) -> None:
        """Persist raw data as a Bronze Delta table with CDC metadata."""
        df = df.withColumn("ingestion_timestamp", F.current_timestamp())
        (
            df.write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(self.config.bronze_full_name)
        )

    def read_bronze(self) -> DataFrame:
        return self.spark.read.table(self.config.bronze_full_name)
