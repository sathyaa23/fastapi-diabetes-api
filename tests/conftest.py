"""Train the model once before the tests run if it does not exist yet."""
import pytest

from src import train


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    """Make sure a model file is available for every test."""
    if not train.MODEL_PATH.exists():
        train.main()
