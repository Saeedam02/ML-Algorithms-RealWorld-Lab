from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml_realworld.data import generate_housing_dataset, generate_phishing_dataset
from ml_realworld.models import make_classifiers, make_linear_regression
from ml_realworld.preprocessing import build_preprocessor


def test_linear_regression_pipeline_smoke():
    df = generate_housing_dataset(n_samples=500, seed=11)
    categorical = ["neighborhood", "home_type", "condition", "heating_type"]
    X = df.drop(columns=["sale_price"])
    y = df["sale_price"]
    numeric = [c for c in X.columns if c not in categorical]
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=1)
    pipe = Pipeline(
        [
            ("preprocess", build_preprocessor(numeric, categorical)),
            ("model", make_linear_regression()),
        ]
    )
    pipe.fit(X_train, y_train)
    assert len(pipe.predict(X_test)) == len(X_test)


def test_all_classifier_pipelines_smoke():
    df = generate_phishing_dataset(n_samples=700, seed=12)
    categorical = ["page_category", "tls_issuer"]
    X = df.drop(columns=["is_phishing"])
    y = df["is_phishing"]
    numeric = [c for c in X.columns if c not in categorical]
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=1, stratify=y
    )

    for estimator in make_classifiers().values():
        pipe = Pipeline(
            [
                ("preprocess", build_preprocessor(numeric, categorical)),
                ("model", estimator),
            ]
        )
        pipe.fit(X_train, y_train)
        assert len(pipe.predict(X_test)) == len(X_test)
