"""
- Download training dataset from AWS S3.
- Prepare data and train model.
- Persist model to AWS S3.
"""
import sys
from typing import Any, Dict, NamedTuple, Tuple

from bodywork_pipeline_utils import aws, logging
from bodywork_pipeline_utils.aws.datasets import Dataset
from numpy import ndarray
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor

AWS_S3_PROJECT_BUCKET = "bodywork-time-to-dispatch"
HYPERPARAM_GRID = {
    "random_state": [42],
    "criterion": ["mse", "mae"],
    "max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10, None],
    "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "min_samples_leaf": [2, 3, 4, 5, 6, 7, 8, 9, 10],
}
R2_METRIC_THRESHOLD = 0.8

log = logging.configure_logger()


class FeatureAndLabels(NamedTuple):
    """Container for features and labels split by test and train sets."""

    X_train: DataFrame
    X_test: DataFrame
    y_train: DataFrame
    y_test: DataFrame


class TaskMetrics(NamedTuple):
    """Container for the task's performance metrics."""

    r_squared: float
    mean_absolute_error: float


def main() -> None:
    """Main training job."""
    log.info("Starting train-model stage.")
    try:
        dataset = aws.get_latest_csv_dataset_from_s3(AWS_S3_PROJECT_BUCKET, "datasets")
        log.info(f"Retrieved dataset from s3://{AWS_S3_PROJECT_BUCKET}/{dataset.key}")

        feature_and_labels = prepare_data(dataset.data)
        model, metrics = train_model(feature_and_labels, HYPERPARAM_GRID)
        log.info(
            f"Trained model: r-squared={metrics.r_squared:.3f}, "
            f"MAE={metrics.mean_absolute_error:.3f}"
        )

        if metrics.r_squared >= R2_METRIC_THRESHOLD:
            s3_location = persist_model(model, dataset, metrics)
            log.info(f"Model serialise dand persisted to s3://{s3_location}")
        else:
            msg = (
                f"r-squared metric ({{metrics.r_squared:.3f}}) is below deployment "
                f"threshold {R2_METRIC_THRESHOLD}"
            )
            raise RuntimeError(msg)
    except Exception as e:
        log.error(f"Error encountered when training model - {e}")
        sys.exit(1)


def prepare_data(data: DataFrame) -> FeatureAndLabels:
    """Split the data into features and labels for training and testing."""
    X = data.drop("hours_to_dispatch", axis=1)
    y = data["hours_to_dispatch"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=data["product_code"].values, random_state=42
    )
    return FeatureAndLabels(X_train, X_test, y_train, y_test)


def compute_metrics(y_true: ndarray, y_pred: ndarray) -> TaskMetrics:
    """Compute performance metrics for the task and log them."""
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return TaskMetrics(r2, mae)


def train_model(
    data: FeatureAndLabels, hyperparam_grid: Dict[str, Any]
) -> Tuple[BaseEstimator, TaskMetrics]:
    """Train a model and compute performance metrics."""
    grid_search = GridSearchCV(
        estimator=DecisionTreeRegressor(),
        param_grid=hyperparam_grid,
        scoring="r2",
        cv=5,
        refit=True,
    )
    grid_search.fit(preprocess(data.X_train), data.y_train)
    best_model = grid_search.best_estimator_
    y_test_pred = best_model.predict(preprocess(data.X_test))
    performance_metrics = compute_metrics(data.y_test, y_test_pred)
    return (best_model, performance_metrics)


def preprocess(df: DataFrame) -> DataFrame:
    """Create features for traiing model."""
    df_processed = df.copy()
    category_map = {"SKU001": 0, "SKU002": 1, "SKU003": 2, "SKU004": 3, "SKU005": 4}
    df_processed["product_code"] = df["product_code"].apply(lambda e: category_map[e])
    return df_processed.values


def persist_model(model: BaseEstimator, dataset: Dataset, metrics: TaskMetrics) -> str:
    """Persist the model and metadata to S3."""
    metadata = {
        "r_squared": metrics.r_squared,
        "mean_absolute_error": metrics.mean_absolute_error,
    }
    wrapped_model = aws.Model("time-to-dispatch", model, dataset, metadata)
    s3_location = wrapped_model.put_model_to_s3(AWS_S3_PROJECT_BUCKET, "models")
    return s3_location


if __name__ == "__main__":
    main()
