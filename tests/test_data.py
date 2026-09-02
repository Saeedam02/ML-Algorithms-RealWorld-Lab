from ml_realworld.data import generate_housing_dataset, generate_phishing_dataset


def test_housing_generator_is_nontrivial():
    df = generate_housing_dataset(n_samples=500, seed=7)
    assert len(df) == 500
    assert df.shape[1] >= 20
    assert df["sale_price"].nunique() > 400
    assert df.isna().sum().sum() > 0


def test_phishing_generator_is_nontrivial():
    df = generate_phishing_dataset(n_samples=800, seed=9)
    assert len(df) == 800
    assert df.shape[1] >= 30
    assert set(df["is_phishing"].unique()) == {0, 1}
    assert df.isna().sum().sum() > 0
