"""Unit tests for data loading, training and prediction logic."""
from src import data, train
from src.predict import predict_data, risk_level

EXAMPLE_ROW = [59, 2, 32.1, 101, 157, 93.2, 38, 4, 4.86, 87]


def test_load_data_shape():
    """The dataset has 442 patients and 10 features."""
    X, y = data.load_data()
    assert X.shape == (442, 10)
    assert len(y) == 442


def test_split_sizes():
    """The split keeps 20 percent of rows for testing."""
    X, y = data.load_data()
    X_train, X_test, _, _ = data.split_data(X, y)
    assert len(X_train) == 353 and len(X_test) == 89


def test_feature_names_match_columns():
    """There is one API name for each of the 10 model columns."""
    assert len(data.FEATURES) == 10


def test_model_beats_baseline():
    """The model explains a meaningful share of the variance."""
    X, y = data.load_data()
    X_train, X_test, y_train, y_test = data.split_data(X, y)
    metrics = train.evaluate(train.fit_model(X_train, y_train), X_test, y_test)
    assert metrics["r2"] > 0.3


def test_risk_level_bands():
    """Scores map to the right band around the thresholds."""
    t = {"low_below": 100, "high_above": 200}
    assert risk_level(50, t) == "low"
    assert risk_level(150, t) == "moderate"
    assert risk_level(250, t) == "high"


def test_predict_data_output():
    """Predictions come back as a score and a valid risk label."""
    result = predict_data([EXAMPLE_ROW])[0]
    assert result["progression_score"] > 0
    assert result["risk_level"] in {"low", "moderate", "high"}
