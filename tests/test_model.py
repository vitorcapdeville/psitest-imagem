import keras
import pytest
import tensorflow as tf

THRESHOLD = 0.95


@pytest.fixture
def model():
    return keras.models.load_model("model.keras")


@pytest.fixture
def data():
    return tf.data.Dataset.load("tests/ds_test_sample")


def test_model(model, data):
    accuracy = model.evaluate(data)[1]
    assert accuracy >= THRESHOLD
