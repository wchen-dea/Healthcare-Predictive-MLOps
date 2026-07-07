from healthcare_mlops.config import HealthcareConfig


def test_full_model_name():
    config = HealthcareConfig(catalog="my_catalog", schema="my_schema")
    assert config.full_model_name == "my_catalog.my_schema.test_result_classifier"


def test_bronze_full_name():
    config = HealthcareConfig(catalog="cat", schema="sch")
    assert config.bronze_full_name == "cat.sch.bronze_patients"


def test_silver_full_name():
    config = HealthcareConfig(catalog="cat", schema="sch")
    assert config.silver_full_name == "cat.sch.silver_patients_features"


def test_gold_full_name():
    config = HealthcareConfig(catalog="cat", schema="sch")
    assert config.gold_full_name == "cat.sch.patient_predictions"


def test_label_map_keys():
    config = HealthcareConfig()
    assert set(config.label_map.keys()) == {"Normal", "Inconclusive", "Abnormal"}


def test_default_feature_columns_non_empty():
    config = HealthcareConfig()
    assert len(config.numeric_features) > 0
    assert len(config.categorical_features) > 0
