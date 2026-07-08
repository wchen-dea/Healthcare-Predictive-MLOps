import mlflow
import mlflow.sklearn
import pandas as pd
from databricks.sdk import WorkspaceClient
from mlflow.models.signature import infer_signature
from pyspark.sql import DataFrame, SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from healthcare_mlops.config import HealthcareConfig


class ModelTrainer:
    """Trains a Random Forest classifier to predict patient test results."""

    def __init__(self, spark: SparkSession, config: HealthcareConfig):
        self.spark = spark
        self.config = config

    def _build_pipeline(self) -> Pipeline:
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        # admission_month is an engineered numeric feature — must be listed explicitly
        # to avoid being silently dropped by remainder="drop"
        all_numeric = self.config.numeric_features + ["admission_month"]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, all_numeric),
                (
                    "cat",
                    categorical_transformer,
                    self.config.categorical_features + ["age_group"],
                ),
            ],
            remainder="drop",
        )

        if self.config.model_algorithm == "random_forest":
            clf = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
                n_jobs=-1,
            )
        elif self.config.model_algorithm == "gradient_boosting":
            clf = GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.config.random_state,
            )
        else:
            raise ValueError(
                "Unsupported model_algorithm. Use 'random_forest' or 'gradient_boosting'."
            )

        return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])

    def train(
        self,
        silver_df: DataFrame,
        experiment_name: str,
        registered_model_name: str | None = None,
    ) -> str:
        """Train model, log to MLflow, return run_id."""
        pdf: pd.DataFrame = silver_df.toPandas()

        X = pdf.drop(columns=[self.config.label_column])
        y = pdf[self.config.label_column]

        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_enc,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y_enc,
        )

        WorkspaceClient().workspace.mkdirs(experiment_name.rsplit("/", 1)[0])
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run() as run:
            mlflow.set_tags(
                {
                    "project": "healthcare-predictive-mlops",
                    "model_type": self.config.model_algorithm,
                    "target": self.config.label_column,
                }
            )

            mlflow.log_params(
                {
                    "model_algorithm": self.config.model_algorithm,
                    "n_estimators": self.config.n_estimators,
                    "max_depth": self.config.max_depth,
                    "test_size": self.config.test_size,
                    "random_state": self.config.random_state,
                }
            )

            pipeline = self._build_pipeline()
            pipeline.fit(X_train, y_train)

            train_acc = pipeline.score(X_train, y_train)
            test_acc = pipeline.score(X_test, y_test)

            mlflow.log_metrics({"train_accuracy": train_acc, "test_accuracy": test_acc})

            signature = infer_signature(X_train, pipeline.predict(X_train))

            model_registry_name = registered_model_name or self.config.full_model_name

            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                signature=signature,
                input_example=X_train.iloc[:5],
                registered_model_name=model_registry_name,
            )

            run_id = run.info.run_id

        return run_id
