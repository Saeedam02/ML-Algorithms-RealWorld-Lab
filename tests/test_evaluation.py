import numpy as np

from ml_realworld.evaluation import classification_metrics, regression_metrics


def test_regression_metrics_perfect_prediction():
    y = np.array([100.0, 200.0, 300.0])
    metrics = regression_metrics(y, y)
    assert metrics["r2"] == 1.0
    assert metrics["mae"] == 0.0


def test_classification_metrics_are_bounded():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    score = np.array([0.1, 0.6, 0.8, 0.9])
    metrics = classification_metrics(y_true, y_pred, score)
    for key, value in metrics.items():
        assert 0.0 <= value <= 1.0, key


class _ProbEstimator:
    def predict_proba(self, X):
        import numpy as np
        return np.column_stack([1 - np.asarray(X), np.asarray(X)])


class _DecisionEstimator:
    def decision_function(self, X):
        import numpy as np
        return np.asarray(X)


class _PredictOnlyEstimator:
    def predict(self, X):
        import numpy as np
        return (np.asarray(X) > 0.5).astype(int)


def test_model_score_supports_common_estimator_interfaces():
    from ml_realworld.evaluation import model_score

    x = np.array([0.2, 0.8])
    assert np.allclose(model_score(_ProbEstimator(), x), x)
    assert np.allclose(model_score(_DecisionEstimator(), x), x)
    assert np.array_equal(model_score(_PredictOnlyEstimator(), x), np.array([0, 1]))
