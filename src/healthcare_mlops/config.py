from dataclasses import dataclass, field
from typing import List


@dataclass
class HealthcareConfig:
    """Central configuration for the Healthcare Predictive MLOps pipeline."""

    # Unity Catalog
    catalog: str = "healthcare_catalog"
    schema: str = "healthcare_ml"

    # Table names
    bronze_table: str = "bronze_patients"
    silver_table: str = "silver_patients_features"
    gold_predictions_table: str = "patient_predictions"

    # MLflow
    experiment_name: str = "/Shared/healthcare-predictive-mlops/test-result-classifier"
    model_name: str = "test_result_classifier"
    champion_alias: str = "champion"
    challenger_alias: str = "challenger"

    # Target column
    label_column: str = "test_result"

    # Categorical features to one-hot encode
    categorical_features: List[str] = field(default_factory=lambda: [
        "gender",
        "blood_type",
        "medical_condition",
        "insurance_provider",
        "admission_type",
        "medication",
    ])

    # Numeric features
    numeric_features: List[str] = field(default_factory=lambda: [
        "age",
        "billing_amount",
        "room_number",
        "length_of_stay_days",
    ])

    # Label encoding map for target
    label_map: dict = field(default_factory=lambda: {
        "Normal": 0,
        "Inconclusive": 1,
        "Abnormal": 2,
    })

    # Training
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 200
    max_depth: int = 10
    min_accuracy_threshold: float = 0.75

    @property
    def full_model_name(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.model_name}"

    @property
    def bronze_full_name(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.bronze_table}"

    @property
    def silver_full_name(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.silver_table}"

    @property
    def gold_full_name(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.gold_predictions_table}"
