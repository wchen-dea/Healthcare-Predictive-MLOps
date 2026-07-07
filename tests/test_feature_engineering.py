
import pytest
from pyspark.sql import SparkSession

from healthcare_mlops.config import HealthcareConfig
from healthcare_mlops.feature_engineering import FeatureEngineer


@pytest.fixture(scope="module")
def spark():
    return (
        SparkSession.builder.master("local[1]")
        .appName("healthcare-mlops-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )


@pytest.fixture
def config():
    return HealthcareConfig(catalog="test_catalog", schema="test_schema")


@pytest.fixture
def sample_df(spark):
    data = [
        ("Bobby Jackson", 30, "Male", "B-", "Cancer", "2024-01-31", "2024-02-02",
         "Blue Cross", 18856.28, 328, "Urgent", "Paracetamol", "Normal"),
        ("Leslie Terry", 62, "Male", "A+", "Obesity", "2019-08-20", "2019-08-26",
         "Medicare", 33643.33, 265, "Emergency", "Ibuprofen", "Inconclusive"),
        ("Danny Smith", 76, "Female", "A-", "Obesity", "2022-09-22", "2022-10-07",
         "Aetna", 27955.10, 205, "Emergency", "Aspirin", "Normal"),
    ]
    columns = [
        "name", "age", "gender", "blood_type", "medical_condition",
        "date_of_admission", "discharge_date", "insurance_provider",
        "billing_amount", "room_number", "admission_type", "medication", "test_results",
    ]
    return spark.createDataFrame(data, columns)


class TestFeatureEngineer:
    def test_clean_drops_nulls(self, spark, config, sample_df):
        engineer = FeatureEngineer(spark, config)
        # Inject a row with null medical_condition
        null_row = spark.createDataFrame(
            [(None, 45, "Female", "O+", None, "2023-01-01", "2023-01-05",
              "Aetna", 10000.0, 101, "Elective", "Aspirin", "Normal")],
            sample_df.schema,
        )
        df_with_null = sample_df.union(null_row)
        cleaned = engineer.clean(df_with_null)
        assert cleaned.count() == sample_df.count()

    def test_engineer_features_length_of_stay(self, spark, config, sample_df):
        engineer = FeatureEngineer(spark, config)
        cleaned = engineer.clean(sample_df)
        df = engineer.engineer_features(cleaned)
        assert "length_of_stay_days" in df.columns
        los_values = [r.length_of_stay_days for r in df.select("length_of_stay_days").collect()]
        assert all(v >= 0 for v in los_values)

    def test_engineer_features_age_group(self, spark, config, sample_df):
        engineer = FeatureEngineer(spark, config)
        cleaned = engineer.clean(sample_df)
        df = engineer.engineer_features(cleaned)
        assert "age_group" in df.columns
        age_groups = {r.age_group for r in df.select("age_group").collect()}
        assert age_groups.issubset({"pediatric", "young_adult", "middle_aged", "senior"})

    def test_normalized_test_result_labels(self, spark, config, sample_df):
        engineer = FeatureEngineer(spark, config)
        cleaned = engineer.clean(sample_df)
        df = engineer.engineer_features(cleaned)
        labels = {r.test_result for r in df.select("test_result").collect()}
        assert labels.issubset({"Normal", "Abnormal", "Inconclusive"})
