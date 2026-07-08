__all__ = [
    "HealthcareConfig",
    "DataLoader",
    "FeatureEngineer",
    "ModelTrainer",
    "ModelEvaluator",
    "BatchPredictor",
]


def __getattr__(name: str):
    """Lazily import public symbols to avoid loading heavy deps at package import time."""
    if name == "HealthcareConfig":
        from healthcare_mlops.config import HealthcareConfig

        return HealthcareConfig
    if name == "DataLoader":
        from healthcare_mlops.data_loader import DataLoader

        return DataLoader
    if name == "FeatureEngineer":
        from healthcare_mlops.feature_engineering import FeatureEngineer

        return FeatureEngineer
    if name == "ModelTrainer":
        from healthcare_mlops.train import ModelTrainer

        return ModelTrainer
    if name == "ModelEvaluator":
        from healthcare_mlops.evaluate import ModelEvaluator

        return ModelEvaluator
    if name == "BatchPredictor":
        from healthcare_mlops.inference import BatchPredictor

        return BatchPredictor
    raise AttributeError(f"module 'healthcare_mlops' has no attribute '{name}'")
