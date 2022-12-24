import pytest
import pandas as pd
from .data import process_data
from .model import train_model, inference, compute_model_metrics

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

@pytest.fixture
def data():
    df = pd.read_csv('starter/data/census_clean.csv')
    return df

def test_process_data(data):
    X, y, encoder, lb = process_data(data, categorical_features=cat_features, label="salary", training=True)
    assert X.shape == (32561, 108)
    assert y.shape == (32561,)

def test_process_data_type(data):
    X, y, encoder, lb = process_data(data, categorical_features=cat_features, label="salary", training=True)
    assert 'numpy.ndarray' in str(type(X))
    assert 'numpy.ndarray' in str(type(y))
    assert 'sklearn.preprocessing._label.LabelBinarizer' in str(type(lb))
    assert 'sklearn.preprocessing._encoders.OneHotEncoder' in str(type(encoder))

def test_model_type(data):
    X, y, encoder, lb = process_data(data, categorical_features=cat_features, label="salary", training=True)
    model = train_model(X, y)
    assert 'sklearn.ensemble._forest.RandomForestClassifier' in str(type(model))

def test_model_metrics_data_type(data):
    X, y, encoder, lb = process_data(data, categorical_features=cat_features, label="salary", training=True)
    model = train_model(X, y)
    preds = inference(model,X)
    assert 'numpy.ndarray' in str(type(preds))

def test_model_metrics_data_type(data):
    X, y, encoder, lb = process_data(data, categorical_features=cat_features, label="salary", training=True)
    model = train_model(X, y)
    preds = inference(model,X)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert 'float64' in str(type(precision))
    assert 'float64' in str(type(recall))
    assert 'float64' in str(type(fbeta))
