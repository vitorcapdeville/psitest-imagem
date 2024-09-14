import keras
import numpy as np
import pytest
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score

ACCURACY_THRESHOLD = 0.95
PRECISION_THRESHOLD = 0.5
RECALL_THRESHOLD = 0.5

@pytest.fixture
def model():
    return keras.models.load_model("model.keras")


@pytest.fixture
def data():
    return tf.data.Dataset.load("tests/ds_test_sample")


def test_model_accurary(model, data):
    accuracy = model.evaluate(data)[1]
    assert accuracy >= ACCURACY_THRESHOLD


def test_category_precision_and_recall(model, data):
    predictions = model.predict(data)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = []
    for _, label in data:
        true_labels.append(label.numpy())

    true_labels = np.concatenate(true_labels)

    precision = precision_score(true_labels, predicted_labels, average=None)
    recall = recall_score(true_labels, predicted_labels, average=None)

    assert (precision >= PRECISION_THRESHOLD).all()
    assert (recall >= RECALL_THRESHOLD).all()
