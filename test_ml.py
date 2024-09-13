import pytest
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, inference, compute_model_metrics
from ml.data import process_data
# TODO: add necessary import


project_path = os.path.dirname(__file__)  # Gets the directory of the current test file
data_path = os.path.join(project_path, "data", "census.csv")

@pytest.fixture
def sample_data():
    # Provide a small sample dataset loaded from the actual data file

    # Load the first 20 rows from the file
    data = pd.read_csv(data_path).head(20)

    # Process the data to get X and y
    cat_features = ["workclass", "education", "marital-status", "occupation",
                    "relationship", "race", "sex", "native-country"]

    X, y, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    return X, y, encoder, lb


def test_return_model_correct_type(sample_data):
    # Test if the train_model function returns a RandomForestClassifier.

    X, y, _, _ = sample_data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier), "Expected RandomForestClassifier"


def test_inference_output_shape(sample_data):
    # Test if the inference function returns a numpy array of the correct shape.

    X, y, _, _ = sample_data
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray), "Expected a numpy array"
    assert preds.shape[0] == X.shape[0], "Expected predictions for each input sample"


def test_compute_metrics_with_sample_data():
    # Test if compute_model_metrics function returns expected values within valid ranges.

    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert 0 <= precision <= 1, "Precision out of expected range"
    assert 0 <= recall <= 1, "Recall out of expected range"
    assert 0 <= fbeta <= 1, "F1 Score out of expected range"