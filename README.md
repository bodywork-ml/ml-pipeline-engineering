# Engineering ML Pipelines - Part Two

This is the second part in a series of articles demonstrating best practices for engineering ML pipelines and deploying them to production. In the [first part](https://www.bodyworkml.com/posts/engineering-ml-pipelines-part-1) we focused on project setup - everything from codebase structure to configuring a CI/CD pipeline and making an initial deployment of a skeleton pipeline.

In this part we are going to focus on developing a fully-operational pipeline and will cover:

- A simple approach to data and model versioning, using cloud object storage.
- How to factor-out common code and make it reusable between projects.
- Defending against errors and handling failure.
- How to enable configurable pipelines that can run in multiple environments without code changes.
- Developing the automated model-training stage and how to write tests for it.
- Developing and testing the serve-model stage that exposes the trained model via a web API.
- Updating the deployment configuration and releasing the changes to production.
- Scheduling the pipeline to run on a schedule.

All of the code referred to in this series of posts is available on  [GitHub](https://github.com/bodywork-ml/ml-pipeline-engineering), with a dedicated branch for each part, so you can explore the code in its various stages of development. Have a quick look before reading on.

## A Simple Strategy for Dataset and Model Versioning

To recap, the data engineering team will deliver the latest tranche of training data to an AWS S3 bucket, in CSV format. They will take responsibility for verifying that these files have the correct schema and contain no unexpected errors. Each filename will contain the timestamp of its creation, in ISO format, so that the datasets in the bucket will look as follows:

```text
s3://time-to-dispatch/
|-- datasets/
    |-- time_to_dispatch_2021-07-03T23:05:32.csv
    |-- time_to_dispatch_2021-07-02T23:05:13.csv
    |-- time_to_dispatch_2021-07-01T23:04:52.csv
    |-- ...
```

The train-model stage of the pipeline will only need to download the latest file for training a new model. We could stop here and rely solely on the filenames as a lightweight versioning strategy, but it is safer to enable [versioning](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Versioning.html) for the S3 bucket and to track of the hash of the dataset used for training, which is computed automatically for every object stored on S3 (the MD5 hash of an object is stored as its [Entity Tag or ETag](https://docs.aws.amazon.com/AmazonS3/latest/API/API_Object.html)). This allows us to defend against accidental deletes and/or overwrites and enables us to locate the precise dataset associated with a trained model.

Because this concept of a dataset is bigger than just an arbitrarily named file on S3, we will need to develop a custom `Dataset` class for representing files on S3 and retrieving their hashes, together with functions/methods for getting and putting `Datasets` to S3.  All of this can be developed on top of  the [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html) AWS client library for Python.

Trained models will be serialised to file using Python’s [pickle](https://docs.python.org/3.8/library/pickle.html) module (this works well for SciKit-Learn models), and uploaded to the same AWS bucket, using the same timestamped file-naming convention:

```text
s3://time-to-dispatch/
|-- models/
    |-- time_to_dispatch_2021-07-03T23:45:23.csv
    |-- time_to_dispatch_2021-07-02T23:45:31.csv
    |-- time_to_dispatch_2021-07-01T23:44:25.csv
    |-- ...
```

When triggered, the serve-model stage of the pipeline will only need to download the most recently persisted model, to ensure that it will generate predictions using the model from the output of the train-model stage. As with the datasets, we could stop here and rely solely on the filenames as a lightweight versioning strategy, but auditing and debugging predictions will be made much easier if we can access model metadata, such as the details of the exact dataset used for training.

The concept of a model becomes bigger than just the trained model in isolation, so we will also need to develop a custom `Model` class. This needs to ‘wrap’ the trained model object, so that it can be associated with all of the metadata that we need to operate our basic model versioning system. As with the custom `Dataset` class, we will need to develop functions/methods for getting and putting the `Model` object to S3.

There is a significant development effort required for implementing the functionality described above and it is likely that this will be repeated in many projects. We are going to cover how to handle reusable code in the section below, but you can see our implementations for the `Dataset` and `Model` classes using the links below, which we have also reproduced at the end of this article.

- [Dataset class](https://github.com/bodywork-ml/bodywork-pipeline-utils/blob/main/src/bodywork_pipeline_utils/aws/datasets.py)
- [Model class](https://github.com/bodywork-ml/bodywork-pipeline-utils/blob/main/src/bodywork_pipeline_utils/aws/models.py)

## Reusing Common Code

The canonical way for distributing reusable Python modules, is by implementing them within a Python package that can be installed into any project that benefits from the functionality. This is what we have done for the dataset and model versioning functionality described in the previous section, and for configuring the logger used in both stages (so we can can enforce a common log format across projects). You can explore the codebase for this package, named `bodywork-pipeline-utils`,  on [GitHub](https://github.com/bodywork-ml/bodywork-pipeline-utils). The functions and classes within it are shown below,

```text
|-- aws
    |-- Dataset
    |-- get_latest_csv_dataset_from_s3
    |-- get_latest_parquet_dataset_from_s3
    |-- put_csv_dataset_to_s3
    |-- put_parquet_dataset_to_s3
    |-- Model
    |-- get_latest_pkl_model_from_s3
|-- logging
    |-- configure_logger
```

A discussion of best practices for developing a Python package is beyond the scope of these articles, but you can use `bodywork-pipeline-utils` as a template and/or refer to the [Python Packaging Authority](https://www.pypa.io/en/latest/). The Scikit-Learn team has also published their insights into [API design for machine learning software](https://arxiv.org/abs/1309.0238), which we recommend reading.

### Distributing Python Packages within your Company

The easiest way to distribute Python packages within an organisation is directly from your Version Control System (VCS) - e.g. a remote Git repository hosted on GitHub. You do not **need** to host an internal PyPI server, unless you have a specific reason to do so. To install a Python package from a remote Git repo you can use,

```plaintext
$ pip install git+https://github.com/bodywork-ml/bodywork-pipeline-utils@v0.1.5
```

Where `v0.1.5` is the release tag, but could also be a Git commit hash. This will need to be specified in `requrements_pipe.txt` as,

```text
git+https://github.com/bodywork-ml/bodywork-pipeline-utils@v0.1.5
```

Pip supports many VCSs and protocols - e.g. private Git repositories can be accessed via SSH by using `git+ssh` and ensuring that the machine making the request has the appropriate SSH keys available. Refer to the [documentation for pip](https://pip.pypa.io/en/stable/cli/pip_install/#vcs-support) for more information.

## Defending Against Errors and Handling Failures

Pipelines can experience many types of error - here are some examples:

- Invalid configuration, such as specifying the wrong storage location for datasets and models.
- Access to datasets and models becomes temporarily unavailable.
- Errors in an unverified dataset causes model-training to fail.
- An unexpected jump in [concept drift](https://en.wikipedia.org/wiki/Concept_drift) causes model metrics to breach performance thresholds.

When developing pipeline stages, it is critical that error events such as these are identified and logged to aid with debugging, and that the pipeline is not allowed to proceed. Our chosen pattern for handling errors is demonstrated in this snippet from `train_model.py`,

```python
import sys

# ...

if __name__ == "__main__":

# ...

    try:
        main(
            s3_bucket,
            r2_metric_error_threshold,
            r2_metric_warning_threshold,
            HYPERPARAM_GRID
        )
		  sys.exit(0)
    except Exception as e:
        log.error(f"Error encountered when training model - {e}")
        sys.exit(1)
```

The pipeline is defined in the `main` function, which is executed within a `try... except` block. If it executes without error, then we signal this to Kubernetes with an exit-code of `0` . If any error is encountered, then the exception is caught, we log the details and signal this to Kubernetes with an exit-code of `1` (so it can attempt a retry, if this has been configured).

Exceptions within `main` are likely to be raised from within 3rd party packages that we’ve installed - e.g. if `bodywork-pipeline-utils` can’t access AWS or if Scikit-Learn fails to train a model. We recommend reading the documentation (or source code) for external functions and classes to understand what exceptions they raise and if the pipeline would benefit from custom handling and logging.

Sometimes, however, we need to look for the error ourselves and raise the exception manually, as shown below when the key test metric falls below a pre-configured threshold level,

```python
def main(
    s3_bucket: str,
    metric_error_threshold: float,
    metric_warning_threshold: float,
    hyperparam_grid: Dict[str, Any]
) -> None:
    """Main training job."""
    log.info("Starting train-model stage.")

    # ...

    if metrics.r_squared >= metric_error_threshold:
        if metrics.r_squared >= metric_warning_threshold:
            log.warning("Metrics breached warning threshold - check for drift.")
        s3_location = persist_model(s3_bucket, model, dataset, metrics)
        log.info(f"Model serialised and persisted to s3://{s3_location}")
    else:
        msg = (
            f"r-squared metric ({{metrics.r_squared:.3f}}) is below deployment "
            f"threshold {metric_error_threshold}"
        )
        raise RuntimeError(msg)
```

This works as follows:

- If the r-squared metric is above the error threshold and the warning threshold, then persist the trained model.
- If the r-squared metric is above the error threshold, but below the warning threshold, then log a warning message and then persist the trained model.
- If the r-squared metric is below the error threshold, then raise an exception, which will cause the stage to log an error and exit with a non-zero exit code (halting the pipeline), using the logic in the `try... except` block discussed earlier in this section.

Using logs to communicate pipeline state will take on additional importance later on in Part Three of this series, when we add monitoring, observability and alerting to our pipeline.

## Configurable Pipelines

Pipelines can benefit from parametrisation to make them re-usable across deployment environments (and potentially tenants, if this makes sense for your project). For example, passing the S3 bucket as an external argument to each stage, enables the pipeline to operate both in a staging environment, as well as in production. Similarly, external arguments can be used to set thresholds for defining when warnings and alerts are triggered, based on model training metrics, which can make testing the pipeline much easier.

Each stage of our pipeline is defined by an executable Python module.  The easiest way to pass arguments to a module is via the command line. For example,

```text
$ python -m pipeline.train_model time-to-dispatch 0.9 0.8
```

Passes an array of strings, `["time-to-dispatch", "0.9", "0.8"]` to `train_model.py`, that can be retrieved from `sys.argv` as demonstrated in the excerpt from `train_model.py` below.

```python
import sys

# ...

if __name__ == "__main__":
    try:
        args = sys.argv
        s3_bucket = args[1]
        r2_metric_error_threshold = float(args[2])
        if r2_metric_error_threshold <= 0 or r2_metric_error_threshold > 1:
            raise ValueError()
        r2_metric_warning_threshold = float(args[3])
        if r2_metric_warning_threshold <= 0 or r2_metric_warning_threshold > 1:
            raise ValueError()
    except (ValueError, IndexError):
        log.error(
            "Invalid arguments passed to train_model.py. "
            "Expected S3_BUCKET R_SQUARED_ERROR_THRESHOLD R_SQUARED_WARNING_THRESHOLD, "
            "where all thresholds must be in the range [0, 1]."
        )
        sys.exit(1)

    try:
        main(
            s3_bucket,
            r2_metric_error_threshold,
            r2_metric_warning_threshold,
            HYPERPARAM_GRID
        )
    except Exception as e:
        log.error(f"Error encountered when training model - {e}")
        sys.exit(1)
```

Note how we cast the numeric arguments to `float` types before performing basic input validation to ensure that users can’t accidentally specify invalided arguments that could lead to unintended consequences.

When deployed by Bodywork,  `train_model.py`will be executed in a dedicated container on Kubernetes. The required arguments can be passed via the `args` parameter in the `bodywork.yaml` file that describes the deployment, as shown below.

```yaml
# bodywork.yaml
...
stages:
  train_model:
    executable_module_path: pipeline/train_model.py
      args: ["time-to-dispatch", "0.9", "0.8"]
      ...
```

## Engineering the Model Training Job

The core task here is to engineer the ML solution in the [time_to_dispatch_model.ipynb notebook](https://github.com/bodywork-ml/ml-pipeline-engineering/blob/master/notebooks/time_to_dispatch_model.ipynb),  provided to us by the data scientist who worked on this task, into the pipeline stage defined in [pipeline/train_model.py](https://github.com/bodywork-ml/ml-pipeline-engineering/blob/part-two/pipeline/train_model.py) (reproduced in the Appendix below). The central workflow is defined in the `main` function,

```python
from typing import Any, Dict, List, NamedTuple, Tuple

from bodywork_pipeline_utils import aws, logging
from bodywork_pipeline_utils.aws import Dataset

# ...

log = logging.configure_logger()

# ...

def main(
    s3_bucket: str,
    metric_error_threshold: float,
    metric_warning_threshold: float,
    hyperparam_grid: Dict[str, Any]
) -> None:
    """Main training job."""
    log.info("Starting train-model stage.")
    dataset = aws.get_latest_csv_dataset_from_s3(s3_bucket, "datasets")
    log.info(f"Retrieved dataset from s3://{s3_bucket}/{dataset.key}")

    feature_and_labels = prepare_data(dataset.data)
    model, metrics = train_model(feature_and_labels, hyperparam_grid)
    validate_trained_model_logic(model, feature_and_labels)
    log.info(
        f"Trained model: r-squared={metrics.r_squared:.3f}, "
        f"MAE={metrics.mean_absolute_error:.3f}"
    )

    if metrics.r_squared >= metric_error_threshold:
        if metrics.r_squared >= metric_warning_threshold:
            log.warning("Metrics breached warning threshold - check for drift.")
        s3_location = persist_model(s3_bucket, model, dataset, metrics)
        log.info(f"Model serialised and persisted to s3://{s3_location}")
    else:
        msg = (
            f"r-squared metric ({{metrics.r_squared:.3f}}) is below deployment "
            f"threshold {metric_error_threshold}"
        )
        raise RuntimeError(msg)
```

This splits the job into smaller sub-tasks, such as preparing the data, that can be delegated to specialised functions that are easier to write (unit) tests for. All interaction with cloud object storage (AWS S3), for retrieving datasets and persisting trained models, is handled by functions imported from the [bodywork-pipeline-utils](https://github.com/bodywork-ml/bodywork-pipeline-utils) package, leaving three key functions that we will discuss in turn:

- `prepare_data`
- `train_model`
- `validate_trained_model_logic`

The `persist_model` function creates the `Model` object and calls its `put_model_to_S3` method. It will be tested implicitly in the functional tests for `main`, which we will look at later on.

### Prepare Data

This purpose of this function is to start with the dataset as a `DataFrame`, split the features from the labels and then partition each of these into ‘test’ and ‘train ‘subsets. We return the results as a `NamedTuple`  called `FeaturesAndLabels`, which facilitates easier access within functions that consume these data structures.

```python
from typing import Any, Dict, List, NamedTuple, Tuple

from sklearn.model_selection import GridSearchCV, train_test_split

# ...

class FeatureAndLabels(NamedTuple):
    """Container for features and labels split by test and train sets."""

    X_train: DataFrame
    X_test: DataFrame
    y_train: DataFrame
    y_test: DataFrame

# ...

def prepare_data(data: DataFrame) -> FeatureAndLabels:
    """Split the data into features and labels for training and testing."""
    X = data.drop("hours_to_dispatch", axis=1)
    y = data["hours_to_dispatch"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=data["product_code"].values, random_state=42
    )
    return FeatureAndLabels(X_train, X_test, y_train, y_test)
```

This is tested in [tests/test_train_model.py](https://github.com/bodywork-ml/ml-pipeline-engineering/blob/part-two/tests/test_train_model.py) as follows,

```python
from pandas import read_csv, DataFrame
from pytest import fixture, raises

from bodywork_pipeline_utils.aws import Dataset

# ...

@fixture(scope="session")
def dataset() -> Dataset:
    data = read_csv("tests/resources/dataset.csv")
    dataset = Dataset(data, datetime(2021, 7, 15), "tests", "resources", "foobar")
    return dataset


def test_prepare_data_splits_labels_and_features_into_test_and_train(dataset: Dataset):
    label_column = "hours_to_dispatch"
    n_rows_in_dataset = dataset.data.shape[0]
    n_cols_in_dataset = dataset.data.shape[1]
    prepared_data = prepare_data(dataset.data)

    assert prepared_data.X_train.shape[1] == n_cols_in_dataset - 1
    assert label_column not in prepared_data.X_train.columns

    assert prepared_data.X_test.shape[1] == n_cols_in_dataset - 1
    assert label_column not in prepared_data.X_test.columns

    assert prepared_data.y_train.ndim == 1
    assert prepared_data.y_train.name == label_column

    assert prepared_data.y_test.ndim == 1
    assert prepared_data.y_test.name == label_column

    assert (prepared_data.X_train.shape[0] + prepared_data.X_test.shape[0]
            == n_rows_in_dataset)

    assert (prepared_data.y_train.shape[0] + prepared_data.y_test.shape[0]
            == n_rows_in_dataset)
```

To help with testing, we have saved a snapshot of CSV data to `tests/resources/dataset.csv` within the project repository, and made it available as a `DataFrame` to all tests in this model, via a [Pytest fixture](https://docs.pytest.org/en/6.2.x/fixture.html) called `dataset`. There is only one unit test for this function and it tests that `prepare_data` splits labels from features, for both  ‘test’ and ‘train’ sets, and that it doesn’t lose any rows of data in the process. If we refactor `prepare_data` in the future, then this test will help prevent us from accidentally leaking the label into the features.

### Train Model

Given a `FeaturesAndLabels` object together with a grid of hyper-parameters, this function will yield a trained model, together with the model’s performance metrics for the ‘test’ set . The hyper-parameter grid is an input  to this function, so that when testing we can use a single point, but can specify many more points for the actual job, when training time is less of a constraint. The metrics are contained within a `NamedTuple` called `TaskMetrics`, to make passing them between functions easier and less prone to error.

```python
from sklearn.model_selection import GridSearchCV, train_test_split

# ...

PRODUCT_CODE_MAP = {"SKU001": 0, "SKU002": 1, "SKU003": 2, "SKU004": 3, "SKU005": 4}

# ...

class TaskMetrics(NamedTuple):
    """Container for the task's performance metrics."""

    r_squared: float
    mean_absolute_error: float

# ...

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
    performance_metrics = TaskMetrics(
        r2_score(data.y_test, y_test_pred),
        mean_absolute_error(data.y_test, y_test_pred)
    )
    return (best_model, performance_metrics)


def preprocess(df: DataFrame) -> DataFrame:
    """Create features for training model."""
    processed = df.copy()
    processed["product_code"] = df["product_code"].apply(lambda e: PRODUCT_CODE_MAP[e])
    return processed.values
```

We have further delegated the task of pre-processing the features for the model (in this case just mapping categories to integers), to a dedicated function called `preprocess`. The `train_model` function is tested in [tests/test_train_model.py](https://github.com/bodywork-ml/ml-pipeline-engineering/blob/part-two/tests/test_train_model.py) as follows,

```python
from sklearn.utils.validation import check_is_fitted

# ...

@fixture(scope="session")
def prepared_data(dataset: Dataset) -> FeatureAndLabels:
    return FeatureAndLabels(
        dataset.data[["orders_placed", "product_code"]][:800],
        dataset.data[["orders_placed", "product_code"]][800:999],
        dataset.data["hours_to_dispatch"][:800],
        dataset.data["hours_to_dispatch"][800:999]
    )

# ...

def test_train_model_yields_model_and_metrics(prepared_data: FeaturesAndLabels):
    model, metrics = train_model(prepared_data, {"random_state": [42]})
    try:
        check_is_fitted(model)
        assert True
    except NotFittedError:
        assert False

    assert metrics.r_squared >= 0.9
    assert metrics.mean_absolute_error <= 1.25
```

Which tests that `train_model` returns a fitted model and acceptable performance metrics, given a reasonably sized tranche of data.

Note, that we haven’t relied on `prepare_data` to create the `FeatureAndLabels object`- we have created this manually in another fixture that relies on the `dataset` fixture discussed earlier. This is a deliberate choice made with the aim of decoupling the outcome of this test from the behaviour of `prepare_data`. Tests that are dependent on multiple functions can be ‘brittle’ and lead to cascades of failing tests when only a single function or method is raising an error. We cannot stress enough how important it is to structure your code in such a way that it can be easily tested.

For completeness, we also provide a simple test for `preprocess`,

```python
from pandas import read_csv, DataFrame

# ...

def test_preprocess_processes_features():
    data = DataFrame({"orders_placed": [30], "product_code": ["SKU004"]})
    processed_data = preprocess(data)
    assert processed_data[0, 0] == 30
    assert processed_data[0, 1] == 3
```

### Validating Trained Models

The goal of the pipeline is to automate the process of training a new model and deploying it - i.e. to take the data scientist out-of-the-loop. Consequently, we need to exercise caution before deploying the latest model. Although the final go/no-go decision on deploying the model will be based on performance metrics, we should also sense-check the model based on basic behaviours we expect it to have. The `validate_trained_model_logic` function performs three logical tests of the model and will raise an exception if it finds an issue (thereby terminating the pipeline before deployment). The three checks are:

1. Does the `hours_to_dispatch` variable increase with `order_placed`, for each product?
2. Are all predictions for the ‘test’ set positive?
3. Are all predictions for the ‘test’ within 25% of the highest `hours_to_dispatch` observation?

```python
def validate_trained_model_logic(model: BaseEstimator, data: FeatureAndLabels) -> None:
    """Verify that a trained model passes basic logical expectations."""
    issues_detected: List[str] = []

    orders_placed_sensitivity_checks = [
        model.predict(array([[100, product], [150, product]])).tolist()
        for product in range(len(PRODUCT_CODE_MAP))
    ]
    if not all(e[0] < e[1] for e in orders_placed_sensitivity_checks):
        issues_detected.append(
            "hours_to_dispatch predictions do not increase with orders_placed"
        )

    test_set_predictions = model.predict(preprocess(data.X_test)).reshape(-1)
    if len(test_set_predictions[test_set_predictions < 0]) > 0:
        issues_detected.append(
            "negative hours_to_dispatch predictions found for test set"
        )
    if len(test_set_predictions[test_set_predictions > data.y_test.max() * 1.25]) > 0:
        issues_detected.append(
            "outlier hours_to_dispatch predictions found for test set"
        )

    if issues_detected:
        msg = "Trained model failed verification: " + ", ".join(issues_detected) + "."
        raise RuntimeError(msg)
```

Note, that we perform all three checks before raising the exception, so that the error message and the logs that will be generated from it, can be maximally informative when it comes to debugging.

The associated test can also be found in [tests/test_train_model.py](https://github.com/bodywork-ml/ml-pipeline-engineering/blob/part-two/tests/test_train_model.py).  This is the most complex test thus far, because we have to use Scikit-Learn’s `DummyRegressor` to create models that will fail each one of the tests individually, as can be seen below.

```python
from pytest import fixture, raises
from sklearn.dummy import DummyRegressor

# ...

def test_validate_trained_model_logic_raises_exception_for_failing_models(
    prepared_data: FeaturesAndLabels
):
    dummy_model = DummyRegressor(strategy="constant", constant=-1.0)
    dummy_model.fit(prepared_data.X_train, prepared_data.y_train)
    expected_exception_str = (
        "Trained model failed verification: "
        "hours_to_dispatch predictions do not increase with orders_placed."
    )
    with raises(RuntimeError, match=expected_exception_str):
        validate_trained_model_logic(dummy_model, prepared_data)

    dummy_model = DummyRegressor(strategy="constant", constant=-1.0)
    dummy_model.fit(prepared_data.X_train, prepared_data.y_train)
    expected_exception_str = (
        "Trained model failed verification: "
        "hours_to_dispatch predictions do not increase with orders_placed, "
        "negative hours_to_dispatch predictions found for test set."
    )
    with raises(RuntimeError, match=expected_exception_str):
        validate_trained_model_logic(dummy_model, prepared_data)

    dummy_model = DummyRegressor(strategy="constant", constant=1000.0)
    dummy_model.fit(prepared_data.X_train, prepared_data.y_train)
    expected_exception_str = (
        "Trained model failed verification: "
        "hours_to_dispatch predictions do not increase with orders_placed, "
        "outlier hours_to_dispatch predictions found for test set."
    )
    with raises(RuntimeError, match=expected_exception_str):
        validate_trained_model_logic(dummy_model, prepared_data)
```

### End-to-End Functional Tests

We’ve tested the individual sub-tasks within `main` , but how do we know that we’ve assembled them correctly, so that `persist_model` will upload the expected `Model` object to cloud storage? We now need to turn our attention to testing `main` from end-to-end - i.e. functional tests for the train-model stage.

The `main` function will try to access AWS S3 to get a dataset and then save a pickled `Model` to S3. We could setup a S3 bucket for testing this integration, but this constitutes an integration test and is not our current aim. We will disable the calls to AWS by mocking the `bodywork_pipeline_utils.aws` module using the `patch` function from the Python standard library’s [unittest.mock](https://docs.python.org/3/library/unittest.mock.html) module.

Decorating our test with `@patch("pipeline.train_model.aws")`, causes `bodywork_pipeline_utils.aws` (which we import into `train_model.py`) to be replaced by a `MagicMock` object called `mock_aws`. This allows us to perform a number of useful tasks:

- Hard-code the return value from `aws.get_latest_csv_dataset_from_s3`, so that it returns our local test dataset instead of a remote dataset on S3.
- Check if the `put_model_to_s3`method of the `aws.Model` object created in `persist_model`, was called.

You can see this in action below.

```python
from unittest.mock import MagicMock, patch

from pytest import fixture, raises
from _pytest.logging import LogCaptureFixture

# ...

@patch("pipeline.train_model.aws")
def test_train_job_happy_path(
    mock_aws: MagicMock,
    dataset: Dataset,
    caplog: LogCaptureFixture,
):
    mock_aws.get_latest_csv_dataset_from_s3.return_value = dataset
    main("project-bucket", 0.8, 0.9, {"random_state": [42]})
    mock_aws.Model().put_model_to_s3.assert_called_once()
    logs = caplog.text
    assert "Starting train-model stage" in logs
    assert "Retrieved dataset from s3" in logs
    assert "Trained model" in logs
    assert "Model serialised and persisted to s3" in logs
```

This test also makes use of Pytest’s [caplog](https://docs.pytest.org/en/6.2.x/reference.html?highlight=caplog#pytest.logging.caplog) fixture, enabling us to test that `main` yields the expected log records when everything goes according to plan (i.e. the ‘happy path’). This gives us confidence that model artefacts will be persisted as expected, when run in production.

What about the ‘unhappy paths’ - when performance metrics fall below warning and error thresholds? We need to test that `main` will behave as we expect it too, and so we will have to write tests for these scenarios, as well.

```python
@patch("pipeline.train_model.aws")
def test_train_job_raises_exception_when_metrics_below_error_threshold(
    mock_aws: MagicMock,
    dataset: Dataset,
):
    mock_aws.get_latest_csv_dataset_from_s3.return_value = dataset
    with raises(RuntimeError, match="below deployment threshold"):
        main("project-bucket", 1, 0.9, {"random_state": [42]})


@patch("pipeline.train_model.aws")
def test_train_job_logs_warning_when_metrics_below_warning_threshold(
    mock_aws: MagicMock,
    dataset: Dataset,
    caplog: LogCaptureFixture,
):
    mock_aws.get_latest_csv_dataset_from_s3.return_value = dataset
    main("project-bucket", 0.5, 0.9, {"random_state": [42]})
    assert "WARNING" in caplog.text
    assert "breached warning threshold" in caplog.text
```

These tests work by setting the thresholds artificially high (or low) and checking that exceptions are raised or that warning messages are logged. Note, that this testing strategy only works because `main` accepts the thresholds as arguments, which was one of the key motivations for designing it in this way.

### Input Validation for the Stage

The train-model stage works by executing `train_model.py`, which requires three arguments to be passed to it (as discussed earlier on). These inputs are validated and this validation needs to be tested for completeness. This is a long and boring test, so we will not reproduce the whole thing, but instead discuss the testing strategy (which is a bit more interesting).

The approach to testing input validation, is to run `test_model.py` as Bodywork would run it within a container on Kubernetes, by calling `python pipeline/train_model.py` from the command line. We can replicate this using `subprocess.run` from the Python standard library and capturing the output. We can then pass invalid arguments and check the output for the expected error messages. You can see this pattern in-action below, for the case when no arguments are passed.

```python
from subprocess import run

# ...

def test_run_job_handles_error_for_invalid_args():
    process_one = run(
        ["python", "pipeline/train_model.py"], capture_output=True, encoding="utf-8"
    )
    assert process_one.returncode != 0
    assert "ERROR" in process_one.stdout
    assert "Invalid arguments passed to train_model.py" in process_one.stdout

	  # ...
```

## Developing the Model Serving Stage

In Part One of this series we developed a skeleton web service that returned a hard-coded value whenever the API was called. Our task in this part is to extend this to downloading the latest model persisted to cloud object storage (AWS S3), and then use the model for generating predictions. Unlike the train-model stage, the effort required for this task is relatively small and so we will reproduce `serve_model.py` in full and then discuss it in more detail afterwards.

```python
import sys
from enum import Enum
from typing import Dict, Union

import uvicorn
from bodywork_pipeline_utils import aws, logging
from fastapi import FastAPI, status
from numpy import array
from pydantic import BaseModel, Field

from pipeline.train_model import PRODUCT_CODE_MAP

app = FastAPI(debug=False)
log = logging.configure_logger()


class ProductCode(Enum):
    SKU001 = "SKU001"
    SKU002 = "SKU002"
    SKU003 = "SKU003"
    SKU004 = "SKU004"
    SKU005 = "SKU005"


class Data(BaseModel):
    product_code: ProductCode
    orders_placed: float = Field(..., ge=0.0)


class Prediction(BaseModel):
    est_hours_to_dispatch: float
    model_version: str


@app.post(
    "/api/v0.1/time_to_dispatch",
    status_code=status.HTTP_200_OK,
    response_model=Prediction,
)
def time_to_dispatch(data: Data) -> Dict[str, Union[str, float]]:
    features = array([[data.orders_placed, PRODUCT_CODE_MAP[data.product_code.value]]])
    prediction = wrapped_model.model.predict(features).tolist()[0]
    return {"est_hours_to_dispatch": prediction, "model_version": str(wrapped_model)}


if __name__ == "__main__":
    try:
        args = sys.argv
        s3_bucket = args[1]
        wrapped_model = aws.get_latest_pkl_model_from_s3(s3_bucket, "models")
        log.info(f"Successfully loaded model: {wrapped_model}")
    except IndexError:
        log.error("Invalid arguments passed to serve_model.py - expected S3_BUCKET")
        sys.exit(1)
    except Exception as e:
        log.error(f"Could not get latest model and start web server - {e}")
        sys.exit(1)
    uvicorn.run(app, host="0.0.0.0", workers=1)
```

The key changes from the version in Part One are as follows:

- We now pass the name of the AWS S3 bucket as an argument to `serve_model.py`.
- In the `if __name__ == "__main__"` block we now attempt to to retrieve latest `Model` object that was persisted to AWS S3, before starting the FastAPI server.
- We placed a new constraint on the `Data.orders_placed` field to ensure that all values sent to the API must be greater-than-or-equal-to zero, and another new constraint on `Data.product_code` that forces this field to be one of the values specified in the `ProductCode` [enumeration](https://docs.python.org/3/library/enum.html).
- We now use the model to generate predictions, using the `PRODUCT_CODE_MAP` dictionary from `train_model.py` to map product codes to integers, before calling the model.
- We use the string representation of the `Model` object in the response’s `model_version` field, which contains the full information on which S3 object is being used, as well as other metadata such as the dataset used to train the model, the type of model, etc. This verbose information is designed to facilitate easy debugging of problematic responses.

If we start the server locally,

```text
$ python -m pipeline.serve_model "bodywork-time-to-dispatch"

2021-07-24 09:56:42,718 - INFO - serve_model.<module> - Successfully loaded model: name:time-to-dispatch|model_type:<class 'sklearn.tree._classes.DecisionTreeRegressor'>|model_timestamp:2021-07-20 14:44:13.558375|model_hash:b4860f56fa24193934fe1ea51b66818d|train_dataset_key:datasets/time_to_dispatch_2021-07-01T16|45|38.csv|train_dataset_hash:"759eccda4ceb7a07cda66ad4ef7cdfbc"|pipeline_git_commit_hash:NA
2021-07-24 09:56:42,718 - INFO - serve_model.<module> - Successfully loaded model: name:time-to-dispatch|model_type:<class 'sklearn.tree._classes.DecisionTreeRegressor'>|model_timestamp:2021-07-20 14:44:13.558375|model_hash:b4860f56fa24193934fe1ea51b66818d|train_dataset_key:datasets/time_to_dispatch_2021-07-01T16|45|38.csv|train_dataset_hash:"759eccda4ceb7a07cda66ad4ef7cdfbc"|pipeline_git_commit_hash:NA
INFO:     Started server process [88289]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Then we can send a test request,

```text
$ curl http://localhost:8000/api/v0.1/time_to_dispatch \
    --request POST \
    --header "Content-Type: application/json" \
    --data '{"product_code": "SKU001", "orders_placed": 10}'
```

Which should return a response along the lines of,

```json
{
  "est_hours_to_dispatch": 0.6527543057985115,
  "model_version": "name:time-to-dispatch|model_type:<class 'sklearn.tree._classes.DecisionTreeRegressor'>|model_timestamp:2021-07-20 14:44:13.558375|model_hash:b4860f56fa24193934fe1ea51b66818d|train_dataset_key:datasets/time_to_dispatch_2021-07-01T16|45|38.csv|train_dataset_hash:\"759eccda4ceb7a07cda66ad4ef7cdfbc\"|pipeline_git_commit_hash:ed3113197adcbdbe338bf406841b930e895c42d6"
}
```

### Updating the Tests

We only need to add one more (small) test to [tests/test_serve_model.py](https://github.com/bodywork-ml/ml-pipeline-engineering/blob/part-two/tests/test_serve_model.py), but we will have to modify the existing tests to take into account that we are now using a trained model to generate predictions, as opposed to returning fixed values. This introduces a complication, because we need to inject a working model into the module.

To facilitate testing, we have persisted a valid `Model` object to `tests/resources/model.pkl`, which will be loaded in a function called `wrapped_model` and injected into the module at test-time as a new object, using `unittest.mock.patch`. We are unable to use `patch` as we did in `train_model.py`, because the model is only loaded when `serve_model.py` is executed, whereas our tests rely only the FastAPI test client.

The modified test for a valid request is shown

```python
import pickle
from subprocess import run
from unittest.mock import patch

from bodywork_pipeline_utils.aws import Model
from fastapi.testclient import TestClient
from numpy import array

test_client = TestClient(app)

def wrapped_model() -> Model:
    with open("tests/resources/model.pkl", "r+b") as file:
        wrapped_model = pickle.load(file)
    return wrapped_model


@patch("pipeline.serve_model.wrapped_model", new=wrapped_model(), create=True)
def test_web_api_returns_valid_response_given_valid_data():
    prediction_request = {"product_code": "SKU001", "orders_placed": 100}
    prediction_response = test_client.post(
        "/api/v0.1/time_to_dispatch", json=prediction_request
    )
    model_obj = wrapped_model()
    expected_prediction = model_obj.model.predict(array([[100, 0]])).tolist()[0]
    assert prediction_response.status_code == 200
    assert prediction_response.json()["est_hours_to_dispatch"] == expected_prediction
    assert prediction_response.json()["model_version"] == str(model_obj)
```

This works by checking the output from the API against the output from the model loaded from the test resources, to make sure that they are identical. Next, we modify the test that covers the API data validation, to reflect the extra constraints we have placed on requests.

```python
@patch("pipeline.serve_model.wrapped_model", new=wrapped_model(), create=True)
def test_web_api_returns_error_code_given_invalid_data():
    prediction_request = {"product_code": "SKU001", "foo": 100}
    prediction_response = test_client.post(
        "/api/v0.1/time_to_dispatch", json=prediction_request
    )
    assert prediction_response.status_code == 422
    assert "value_error.missing" in prediction_response.text

    prediction_request = {"product_code": "SKU000", "orders_placed": 100}
    prediction_response = test_client.post(
        "/api/v0.1/time_to_dispatch", json=prediction_request
    )
    assert prediction_response.status_code == 422
    assert "not a valid enumeration member" in prediction_response.text

    prediction_request = {"product_code": "SKU001", "orders_placed": -100}
    prediction_response = test_client.post(
        "/api/v0.1/time_to_dispatch", json=prediction_request
    )
    assert prediction_response.status_code == 422
    assert "ensure this value is greater than or equal to 0" in prediction_response.text
```

Finally, we add one more test to cover the input validation for the `serve_model.py` module, using the same strategy as we did for the equivalent test for `train_model.py`.

```python
from subprocess import run

# ...

def test_web_server_raises_exception_if_passed_invalid_args():
    process = run(
        ["python", "-m", "pipeline.serve_model"], capture_output=True, encoding="utf-8"
    )
    assert process.returncode != 0
    assert "ERROR" in process.stdout
    assert "Invalid arguments passed to serve_model.py" in process.stdout
```

## Updating the Deployment and Releasing to Production

The last task we need to complete before we can commit all changes, push to GitHub and trigger the CI/CD pipeline, is to update the deployment configuration in `bodywork.yaml`. This requires three changes:

- Arguments now need to be passed to each stage.
- The Python package requirements for each stage need to be updated.
- AWS credentials need to be injected into each stage, as required by `bodywork_pipeline_utils.aws`.

```yaml
version: "1.0"
project:
  name: time-to-dispatch
  docker_image: bodyworkml/bodywork-core:2.1.7
  DAG: train_model >> serve_model
stages:
  train_model:
    executable_module_path: pipeline/train_model.py
    args: ["bodywork-time-to-dispatch", "0.9", "0.8"]
    requirements:
      - numpy==1.21.0
      - pandas==1.2.5
      - scikit-learn==0.24.2
      - git+https://github.com/bodywork-ml/bodywork-pipeline-utils@v0.1.5
    cpu_request: 0.5
    memory_request_mb: 100
    batch:
      max_completion_time_seconds: 15
      retries: 2
    secrets:
      AWS_ACCESS_KEY_ID: aws-credentials
      AWS_SECRET_ACCESS_KEY: aws-credentials
      AWS_DEFAULT_REGION: aws-credentials
  serve_model:
    executable_module_path: pipeline/serve_model.py
    args: ["bodywork-time-to-dispatch"]
    requirements:
      - numpy==1.21.0
      - fastapi==0.65.2
      - uvicorn==0.14.0
      - git+https://github.com/bodywork-ml/bodywork-pipeline-utils@v0.1.5
    cpu_request: 0.25
    memory_request_mb: 100
    service:
      max_startup_time_seconds: 15
      replicas: 2
      port: 8000
      ingress: true
    secrets:
      AWS_ACCESS_KEY_ID: aws-credentials
      AWS_SECRET_ACCESS_KEY: aws-credentials
      AWS_DEFAULT_REGION: aws-credentials
logging:
  log_level: INFO
```

This will instruct Bodywork to look for `AWS_ACCESS_KEY_ID`,  `AWS_SECRET_ACCESS_KEY` and `AWS_DEFAULT_REGION` in a secret record called `aws-credentials`, so that it can inject these secrets into the containers running the stages of our pipeline (as environment variables that will be detected silently). So, these will have to be created, which can be done as follows,

```text
$ bodywork secret create \
    --namespace=pipelines \
    --name=aws-credentials \
    --data AWS_ACCESS_KEY_ID=put-your-key-in-here \
           AWS_SECRET_ACCESS_KEY=put-your-other-key-in-here \
           AWS_DEFAULT_REGION=wherever-your-cluster-is
```

Now you’re ready to push this branch to your remote Git repo! If your tests pass and your colleagues approve the merge, the CD part of the CI/CD pipeline we setup in Part One will ensure the new pipeline is deployed to Kubernetes by Bodywork and executed immediately. Bodywork will perform a rolling-deployment that will ensure zero down-time and automatically roll-back failed deployments to the previous version. When Bodywork has finished, test the new web API,

```text
$ curl http://CLUSTER_IP/pipelines/time-to-dispatch--serve-model/api/v0.1/time_to_dispatch \
    --request POST \
    --header "Content-Type: application/json" \
    --data '{"product_code": "SKU001", "orders_placed": 10}'
```

Where you should observe the same response you received when testing locally,

```json
{
  "est_hours_to_dispatch": 0.6527543057985115,
  "model_version": "name:time-to-dispatch|model_type:<class 'sklearn.tree._classes.DecisionTreeRegressor'>|model_timestamp:2021-07-20 14:44:13.558375|model_hash:b4860f56fa24193934fe1ea51b66818d|train_dataset_key:datasets/time_to_dispatch_2021-07-01T16|45|38.csv|train_dataset_hash:\"759eccda4ceb7a07cda66ad4ef7cdfbc\"|pipeline_git_commit_hash:ed3113197adcbdbe338bf406841b930e895c42d6"
}
```

## Scheduling the Pipeline to run on a Schedule

At this point, the pipeline will have deployed a model using the most recent dataset made available for this task. We know, however, that new data will arrive every Friday evening and so we’d like to schedule the pipeline to run just after the data is expected. We can achieve this using Bodywork cronjobs, as follows,

```text
bodywork cronjob create \
    --namespace=pipelines \
    --name=weekly-update \
    --schedule="0 45 * * *" \
    --git-repo-url=https://github.com/bodywork-ml/ml-pipeline-engineering \
    --git-repo-branch=master \
	  --retries=2
```

## Wrap-Up

In this second part we have gone from a skeleton “Hello, Production!” deployment to a fully-functional train-and-deploy pipeline, that automates re-training and re-deployment in a production environment, on a periodic basis. We have factored-out common code so that it can be re-used across projects and discussed various strategies for developing automated tests for both stages of the pipeline, ensuring that subsequent modifications can be reliably integrated and deployed, with relative ease.

In the final part of this series we will cover monitoring and observability and aim to to answer the question, “*How will I know when something has gone wrong?*”.

## Appendix

For reference.

### The `Dataset` Class

Reproduced from the [bodywork-pipeline-utils](https://github.com/bodywork-ml/bodywork-pipeline-utils) package, which is available to download from [PyPI](https://pypi.org/project/bodywork-pipeline-utils/).

```python
from datetime import datetime
from tempfile import NamedTemporaryFile
from typing import Any, NamedTuple

from pandas import DataFrame, read_csv, read_parquet

from bodywork_pipeline_utils.aws.artefacts import (
    find_latest_artefact_on_s3,
    make_timestamped_filename,
    put_file_to_s3,
)


class Dataset(NamedTuple):
    """Container for downloaded datasets and associated metadata."""

    data: DataFrame
    datetime: datetime
    bucket: str
    key: str
    hash: str


def get_latest_csv_dataset_from_s3(bucket: str, folder: str = "") -> Dataset:
    """Get the latest CSV dataset from S3.

    Args:
        bucket: S3 bucket to look in.
        folder: Folder within bucket to limit search, defaults to "".

    Returns:
        Dataset object.
    """
    artefact = find_latest_artefact_on_s3("csv", bucket, folder)
    data = read_csv(artefact.get())
    return Dataset(data, artefact.timestamp, bucket, artefact.obj_key, artefact.etag)


def get_latest_parquet_dataset_from_s3(bucket: str, folder: str = "") -> Dataset:
    """Get the latest Parquet dataset from S3.

    Args:
        bucket: S3 bucket to look in.
        folder: Folder within bucket to limit search, defaults to "".

    Returns:
        Dataset object.
    """
    artefact = find_latest_artefact_on_s3("parquet", bucket, folder)
    data = read_parquet(artefact.get())
    return Dataset(data, artefact.timestamp, bucket, artefact.obj_key, artefact.etag)


def put_csv_dataset_to_s3(
    data: DataFrame,
    filename_prefix: str,
    ref_datetime: datetime,
    bucket: str,
    folder: str = "",
    **kwargs: Any,
) -> None:
    """Upload DataFrame to S3 as a CSV file.

    Args:
        data: The DataFrame to upload.
        filename_prefix: Prefix before datetime filename element.
        ref_datetime: The reference date associated with data.
        bucket: Location on S3 to persist the data.
        folder: Folder within the bucket, defaults to "".
        kwargs: Keywork arguments to pass to pandas.to_csv.
    """
    filename = make_timestamped_filename(filename_prefix, ref_datetime, "csv")
    with NamedTemporaryFile() as temp_file:
        data.to_csv(temp_file, **kwargs)
        put_file_to_s3(temp_file.name, bucket, folder, filename)


def put_parquet_dataset_to_s3(
    data: DataFrame,
    filename_prefix: str,
    ref_datetime: datetime,
    bucket: str,
    folder: str = "",
    **kwargs: Any,
) -> None:
    """Upload DataFrame to S3 as a Parquet file.

    Args:
        data: The DataFrame to upload.
        filename_prefix: Prefix before datetime filename element.
        ref_datetime: The reference date associated with data.
        bucket: Location on S3 to persist the data.
        folder: Folder within the bucket, defaults to "".
        kwargs: Keywork arguments to pass to pandas.to_csv.
    """
    filename = make_timestamped_filename(filename_prefix, ref_datetime, "parquet")
    with NamedTemporaryFile() as temp_file:
        data.to_parquet(temp_file, **kwargs)
        put_file_to_s3(temp_file.name, bucket, folder, filename)
```

### The `Model` Class

Reproduced from the [bodywork-pipeline-utils](https://github.com/bodywork-ml/bodywork-pipeline-utils) package, which is available to download from [PyPI](https://pypi.org/project/bodywork-pipeline-utils/).

```python
from datetime import datetime
from hashlib import md5
from os import environ
from pickle import dump, dumps, loads, PicklingError, UnpicklingError
from tempfile import NamedTemporaryFile
from typing import Any, cast, Dict, Optional

from bodywork_pipeline_utils.aws.datasets import Dataset
from bodywork_pipeline_utils.aws.artefacts import (
    find_latest_artefact_on_s3,
    make_timestamped_filename,
    put_file_to_s3,
)


class Model:
    """Base class for representing ML models and metadata."""

    def __init__(
        self,
        name: str,
        model: Any,
        train_dataset: Dataset,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Constructor.

        Args:
            name: Model name.
            model: Trained model object.
            train_dataset: Dataset object used to train the model.
            metadata: Arbitrary model metadata.
        """
        self._name = name
        self._train_dataset_key = train_dataset.key
        self._train_dataset_hash = train_dataset.hash
        self._model_hash = self._compute_model_hash(model)
        self._model = model
        self._model_type = type(model)
        self._creation_time = datetime.now()
        self._pipeline_git_commit_hash = environ.get("GIT_COMMIT_HASH", "NA")
        self._metadata = metadata

    def __eq__(self, other: object) -> bool:
        """Model quality operator."""
        if isinstance(other, Model):
            conditions = [
                self._train_dataset_hash == other._train_dataset_hash,
                self._train_dataset_key == other._train_dataset_key,
                self._creation_time == other._creation_time,
                self._pipeline_git_commit_hash == other._pipeline_git_commit_hash,
            ]
            if all(conditions):
                return True
            else:
                return False
        else:
            return False

    def __repr__(self) -> str:
        """Stdout representation."""
        info = (
            f"name: {self._name}\n"
            f"model_type: {self._model_type}\n"
            f"model_timestamp: {self._creation_time}\n"
            f"model_hash: {self._model_hash}\n"
            f"train_dataset_key: {self._train_dataset_key}\n"
            f"train_dataset_hash: {self._train_dataset_hash}\n"
            f"pipeline_git_commit_hash: {self._pipeline_git_commit_hash}"
        )
        return info

    def __str__(self) -> str:
        """String representation."""
        info = (
            f"name:{self._name}|"
            f"model_type:{self._model_type}|"
            f"model_timestamp:{self._creation_time}|"
            f"model_hash:{self._model_hash}|"
            f"train_dataset_key:{self._train_dataset_key}|"
            f"train_dataset_hash:{self._train_dataset_hash}|"
            f"pipeline_git_commit_hash:{self._pipeline_git_commit_hash}"
        )
        return info

    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        return self._metadata

    @property
    def model(self) -> Any:
        return self._model

    @staticmethod
    def _compute_model_hash(model: Any) -> str:
        """Compute a hash for a model object."""
        try:
            model_bytestream = dumps(model, protocol=5)
            hash = md5(model_bytestream)
            return hash.hexdigest()
        except PicklingError:
            msg = "Could not pickle model into bytes before hashing."
            raise RuntimeError(msg)
        except Exception as e:
            msg = "Could not hash model."
            raise RuntimeError(msg) from e

    def put_model_to_s3(self, bucket: str, folder: str = "") -> str:
        """Upload model to S3 as a pickle file.

        Args:
            bucket: Location on S3 to persist the data.
            folder: Folder within the bucket, defaults to "".
        """
        filename = make_timestamped_filename(self._name, self._creation_time, "pkl")
        with NamedTemporaryFile() as temp_file:
            dump(self, temp_file, protocol=5)
            put_file_to_s3(temp_file.name, bucket, folder, filename)
        return f"{bucket}/{folder}/{filename}"


def get_latest_pkl_model_from_s3(bucket: str, folder: str = "") -> Model:
    """Get the latest model from S3.

    Args:
        bucket: S3 bucket to look in.
        folder: Folder within bucket to limit search, defaults to "".

    Returns:
        Dataset object.
    """
    artefact = find_latest_artefact_on_s3("pkl", bucket, folder)
    try:
        artefact_bytes = artefact.get().read()
        model = cast(Model, loads(artefact_bytes))
        return model
    except UnpicklingError:
        msg = "artefact at {bucket}/{model.obj_key} could not be unpickled."
        raise RuntimeError(msg)
    except AttributeError:
        msg = "artefact at {bucket}/{model.obj_key} is not type Model."
        raise RuntimeError(msg)
```

### `train_model.py`

Reproduced from the ml-[pipeline-engineering](https://github.com/bodywork-ml/ml-pipeline-engineering/tree/part-two) repository.

```python
"""
- Download training dataset from AWS S3.
- Prepare data and train model.
- Persist model to AWS S3.
"""
import sys
from typing import Any, Dict, List, NamedTuple, Tuple

from bodywork_pipeline_utils import aws, logging
from bodywork_pipeline_utils.aws import Dataset
from numpy import array, ndarray
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor

PRODUCT_CODE_MAP = {"SKU001": 0, "SKU002": 1, "SKU003": 2, "SKU004": 3, "SKU005": 4}
HYPERPARAM_GRID = {
    "random_state": [42],
    "criterion": ["mse", "mae"],
    "max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10, None],
    "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "min_samples_leaf": [2, 3, 4, 5, 6, 7, 8, 9, 10],
}

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


def main(
    s3_bucket: str,
    metric_error_threshold: float,
    metric_warning_threshold: float,
    hyperparam_grid: Dict[str, Any]
) -> None:
    """Main training job."""
    log.info("Starting train-model stage.")
    dataset = aws.get_latest_csv_dataset_from_s3(s3_bucket, "datasets")
    log.info(f"Retrieved dataset from s3://{s3_bucket}/{dataset.key}")

    feature_and_labels = prepare_data(dataset.data)
    model, metrics = train_model(feature_and_labels, hyperparam_grid)
    validate_trained_model_logic(model, feature_and_labels)
    log.info(
        f"Trained model: r-squared={metrics.r_squared:.3f}, "
        f"MAE={metrics.mean_absolute_error:.3f}"
    )

    if metrics.r_squared >= metric_error_threshold:
        if metrics.r_squared >= metric_warning_threshold:
            log.warning("Metrics breached warning threshold - check for drift.")
        s3_location = persist_model(s3_bucket, model, dataset, metrics)
        log.info(f"Model serialised and persisted to s3://{s3_location}")
    else:
        msg = (
            f"r-squared metric ({{metrics.r_squared:.3f}}) is below deployment "
            f"threshold {metric_error_threshold}"
        )
        raise RuntimeError(msg)


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


def validate_trained_model_logic(model: BaseEstimator, data: FeatureAndLabels) -> None:
    """Verify that a trained model passes basic logical expectations."""
    issues_detected: List[str] = []

    orders_placed_sensitivity_checks = [
        model.predict(array([[100, product], [150, product]])).tolist()
        for product in range(len(PRODUCT_CODE_MAP))
    ]
    if not all(e[0] < e[1] for e in orders_placed_sensitivity_checks):
        issues_detected.append(
            "hours_to_dispatch predictions do not increase with orders_placed"
        )

    test_set_predictions = model.predict(preprocess(data.X_test)).reshape(-1)
    if len(test_set_predictions[test_set_predictions < 0]) > 0:
        issues_detected.append(
            "negative hours_to_dispatch predictions found for test set"
        )
    if len(test_set_predictions[test_set_predictions > data.y_test.max() * 1.25]) > 0:
        issues_detected.append(
            "outlier hours_to_dispatch predictions found for test set"
        )

    if issues_detected:
        msg = "Trained model failed verification: " + ", ".join(issues_detected) + "."
        raise RuntimeError(msg)


def preprocess(df: DataFrame) -> DataFrame:
    """Create features for training model."""
    processed = df.copy()
    processed["product_code"] = df["product_code"].apply(lambda e: PRODUCT_CODE_MAP[e])
    return processed.values


def persist_model(
    bucket: str, model: BaseEstimator, dataset: Dataset, metrics: TaskMetrics
) -> str:
    """Persist the model and metadata to S3."""
    metadata = {
        "r_squared": metrics.r_squared,
        "mean_absolute_error": metrics.mean_absolute_error,
    }
    wrapped_model = aws.Model("time-to-dispatch", model, dataset, metadata)
    s3_location = wrapped_model.put_model_to_s3(bucket, "models")
    return s3_location


if __name__ == "__main__":
    try:
        args = sys.argv
        s3_bucket = args[1]
        r2_metric_error_threshold = float(args[2])
        if r2_metric_error_threshold <= 0 or r2_metric_error_threshold > 1:
            raise ValueError()
        r2_metric_warning_threshold = float(args[3])
        if r2_metric_warning_threshold <= 0 or r2_metric_warning_threshold > 1:
            raise ValueError()
    except (ValueError, IndexError):
        log.error(
            "Invalid arguments passed to train_model.py. "
            "Expected S3_BUCKET R_SQUARED_ERROR_THRESHOLD R_SQUARED_WARNING_THRESHOLD, "
            "where all thresholds must be in the range [0, 1]."
        )
        sys.exit(1)

    try:
        main(
            s3_bucket,
            r2_metric_error_threshold,
            r2_metric_warning_threshold,
            HYPERPARAM_GRID
        )
    except Exception as e:
        log.error(f"Error encountered when training model - {e}")
        sys.exit(1)
```
