"""Dataset generation and loading utilities.

Why generate data instead of shipping a toy dataset?
----------------------------------------------------
The repository is designed to run fully offline while still exercising realistic
ML challenges: mixed feature types, missing values, outliers, correlated
variables, class imbalance, nonlinear interactions, and label noise.

The two datasets are *synthetic benchmarks with realistic semantics*. They are
not claimed to be real transactions or security incidents. This avoids licensing
ambiguity and lets every user reproduce exactly the same benchmark from a seed.

Problem 1: Urban housing affordability / sale-price prediction.
Problem 2: Phishing website detection.

The generators intentionally include signal that a linear model can learn plus
some nonlinear structure that exposes the limits of each algorithm.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import HOUSING_CSV, PHISHING_CSV, RANDOM_SEED


def generate_housing_dataset(
    n_samples: int = 12_000,
    *,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Create a challenging mixed-type house-price dataset.

    The target is generated from interpretable economic and property drivers,
    then perturbed with heteroscedastic noise and a small number of outliers.

    Parameters
    ----------
    n_samples:
        Number of synthetic property sales.
    seed:
        Random seed for reproducibility.
    """

    rng = np.random.default_rng(seed)

    neighborhoods = np.array(
        [
            "Downtown",
            "Riverside",
            "University",
            "OldTown",
            "NorthHeights",
            "WestPark",
            "EastMarket",
            "GreenValley",
            "TechDistrict",
            "LakeView",
            "SouthGate",
            "Hillcrest",
        ]
    )
    home_types = np.array(["Detached", "Townhouse", "Condo", "Duplex"])
    conditions = np.array(["Poor", "Fair", "Average", "Good", "Excellent"])
    heating = np.array(["Gas", "Electric", "HeatPump", "District"])

    living_area = np.clip(rng.lognormal(np.log(1650), 0.38, n_samples), 450, 5500)
    lot_area = np.clip(rng.lognormal(np.log(7200), 0.60, n_samples), 700, 45_000)
    overall_quality = np.clip(np.rint(rng.normal(6.2, 1.7, n_samples)), 1, 10).astype(int)
    year_built = rng.integers(1920, 2026, n_samples)
    renovated = rng.random(n_samples) < 0.34
    renovation_year = np.where(
        renovated,
        rng.integers(np.maximum(year_built, 1970), 2027),
        year_built,
    )
    bedrooms = np.clip(rng.poisson(2.6, n_samples) + 1, 1, 7)
    bathrooms = np.clip(
        np.round(rng.normal(1.8 + 0.00035 * (living_area - 1500), 0.55, n_samples) * 2) / 2,
        1,
        5,
    )
    garage_spaces = np.clip(
        np.rint(rng.normal(1.6 + 0.12 * (overall_quality - 6), 0.8, n_samples)),
        0,
        4,
    ).astype(float)
    distance_downtown_km = np.clip(rng.gamma(2.2, 4.0, n_samples), 0.2, 35)
    school_score = np.clip(
        rng.normal(68 + 2.4 * (overall_quality - 6), 12, n_samples), 25, 100
    )
    crime_index = np.clip(
        rng.gamma(2.0, 9.0, n_samples) + 0.55 * distance_downtown_km, 1, 80
    )
    transit_score = np.clip(
        92 - 2.3 * distance_downtown_km + rng.normal(0, 12, n_samples), 5, 100
    )
    energy_efficiency = np.clip(
        38
        + 0.48 * (year_built - 1950)
        + 4.0 * (heating[rng.integers(0, len(heating), n_samples)] == "HeatPump")
        + rng.normal(0, 10, n_samples),
        10,
        100,
    )
    mortgage_rate = np.clip(rng.normal(5.4, 1.1, n_samples), 2.0, 9.5)
    unemployment_rate = np.clip(rng.normal(5.6, 1.5, n_samples), 2.0, 12.0)

    neighborhood = rng.choice(
        neighborhoods,
        n_samples,
        p=[0.08, 0.08, 0.07, 0.10, 0.09, 0.10, 0.09, 0.09, 0.08, 0.07, 0.08, 0.07],
    )
    home_type = rng.choice(home_types, n_samples, p=[0.58, 0.18, 0.18, 0.06])
    condition = rng.choice(conditions, n_samples, p=[0.04, 0.10, 0.42, 0.32, 0.12])
    heating_type = rng.choice(heating, n_samples, p=[0.48, 0.14, 0.27, 0.11])

    neighborhood_effect = {
        "Downtown": 55_000,
        "Riverside": 42_000,
        "University": 38_000,
        "OldTown": -12_000,
        "NorthHeights": 62_000,
        "WestPark": 28_000,
        "EastMarket": -5_000,
        "GreenValley": 46_000,
        "TechDistrict": 72_000,
        "LakeView": 88_000,
        "SouthGate": -18_000,
        "Hillcrest": 52_000,
    }
    home_type_effect = {
        "Detached": 28_000,
        "Townhouse": 7_000,
        "Condo": -18_000,
        "Duplex": -8_000,
    }
    condition_effect = {
        "Poor": -55_000,
        "Fair": -25_000,
        "Average": 0,
        "Good": 22_000,
        "Excellent": 48_000,
    }

    price = (
        28_000
        + 118 * living_area
        + 2.7 * lot_area
        + 24_000 * overall_quality
        + 13_000 * bathrooms
        + 11_500 * garage_spaces
        + 1_250 * school_score
        - 3_900 * distance_downtown_km
        - 1_250 * crime_index
        + 520 * transit_score
        + 620 * energy_efficiency
        - 7_500 * mortgage_rate
        - 3_200 * unemployment_rate
        + 650 * (year_built - 1950)
        + 210 * (renovation_year - year_built)
        + pd.Series(neighborhood).map(neighborhood_effect).to_numpy()
        + pd.Series(home_type).map(home_type_effect).to_numpy()
        + pd.Series(condition).map(condition_effect).to_numpy()
    )

    # Nonlinear interactions create realistic model misspecification for a purely
    # linear learner. The model can still perform well but not perfectly.
    price += 7.0 * overall_quality * np.sqrt(living_area) * 100
    price += np.where(
        (neighborhood == "LakeView") & (overall_quality >= 8),
        55_000,
        0,
    )

    noise_scale = 18_000 + 0.055 * np.maximum(price, 50_000)
    price += rng.normal(0, noise_scale)

    # Rare luxury / distressed transactions create heavy-tailed residuals.
    luxury_mask = rng.random(n_samples) < 0.012
    distress_mask = rng.random(n_samples) < 0.010
    price[luxury_mask] *= rng.uniform(1.35, 1.75, luxury_mask.sum())
    price[distress_mask] *= rng.uniform(0.55, 0.78, distress_mask.sum())
    price = np.clip(price, 55_000, None)

    df = pd.DataFrame(
        {
            "living_area_sqft": np.round(living_area, 1),
            "lot_area_sqft": np.round(lot_area, 1),
            "overall_quality": overall_quality,
            "year_built": year_built,
            "renovation_year": renovation_year,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "garage_spaces": garage_spaces,
            "distance_downtown_km": np.round(distance_downtown_km, 2),
            "school_score": np.round(school_score, 1),
            "crime_index": np.round(crime_index, 1),
            "transit_score": np.round(transit_score, 1),
            "energy_efficiency": np.round(energy_efficiency, 1),
            "mortgage_rate_pct": np.round(mortgage_rate, 2),
            "unemployment_rate_pct": np.round(unemployment_rate, 2),
            "neighborhood": neighborhood,
            "home_type": home_type,
            "condition": condition,
            "heating_type": heating_type,
            "sale_price": np.round(price, 0),
        }
    )

    # Missingness is deliberately non-zero so the pipeline must impute.
    for column, rate in {
        "lot_area_sqft": 0.025,
        "school_score": 0.035,
        "garage_spaces": 0.020,
        "energy_efficiency": 0.030,
        "heating_type": 0.012,
    }.items():
        mask = rng.random(n_samples) < rate
        df.loc[mask, column] = np.nan

    return df


def generate_phishing_dataset(
    n_samples: int = 18_000,
    *,
    seed: int = RANDOM_SEED + 1,
) -> pd.DataFrame:
    """Create a realistic phishing-detection benchmark.

    Features mimic URL, TLS, page-content, redirect and domain-trust signals
    commonly used in phishing research. A nonlinear latent risk function creates
    the binary label, followed by a small amount of label noise.
    """

    rng = np.random.default_rng(seed)

    page_categories = np.array(["banking", "ecommerce", "social", "cloud", "blog", "other"])
    tls_issuers = np.array(["trusted_ca", "unknown_ca", "self_signed", "missing"])

    url_length = np.clip(rng.gamma(4.5, 13.0, n_samples), 8, 320)
    num_dots = np.clip(rng.poisson(2.3, n_samples), 0, 12)
    num_hyphens = np.clip(rng.poisson(1.1, n_samples), 0, 12)
    num_digits = np.clip(rng.poisson(4.2, n_samples), 0, 40)
    num_subdomains = np.clip(rng.poisson(1.25, n_samples), 0, 8)
    has_ip_address = rng.binomial(1, 0.075, n_samples)
    has_at_symbol = rng.binomial(1, 0.045, n_samples)
    uses_https = rng.binomial(1, 0.86, n_samples)
    suspicious_tld = rng.binomial(1, 0.14, n_samples)
    shortened_url = rng.binomial(1, 0.11, n_samples)
    domain_age_days = np.clip(rng.lognormal(np.log(900), 1.0, n_samples), 0, 9000)
    dns_record_age_days = np.clip(
        domain_age_days * rng.uniform(0.2, 1.0, n_samples) + rng.normal(0, 60, n_samples),
        0,
        9000,
    )
    redirect_count = np.clip(rng.poisson(0.65, n_samples), 0, 8)
    external_link_ratio = np.clip(rng.beta(2.2, 3.7, n_samples), 0, 1)
    request_url_ratio = np.clip(rng.beta(2.0, 3.5, n_samples), 0, 1)
    anchor_url_ratio = np.clip(rng.beta(1.9, 3.0, n_samples), 0, 1)
    form_action_external = rng.binomial(1, 0.12, n_samples)
    iframe_count = np.clip(rng.poisson(0.45, n_samples), 0, 7)
    popup_count = np.clip(rng.poisson(0.35, n_samples), 0, 8)
    script_count = np.clip(rng.lognormal(np.log(14), 0.65, n_samples), 0, 150)
    url_entropy = np.clip(rng.normal(3.4, 0.75, n_samples), 1.0, 6.5)
    brand_mismatch = rng.binomial(1, 0.16, n_samples)
    login_form = rng.binomial(1, 0.38, n_samples)
    certificate_days_left = np.clip(rng.normal(120, 95, n_samples), -30, 825)
    page_rank_proxy = np.clip(rng.beta(1.4, 3.0, n_samples) * 100, 0, 100)
    spelling_score = np.clip(rng.normal(91, 8, n_samples), 35, 100)
    favicon_external = rng.binomial(1, 0.09, n_samples)
    nonstandard_port = rng.binomial(1, 0.035, n_samples)
    page_category = rng.choice(page_categories, n_samples, p=[0.13, 0.18, 0.18, 0.14, 0.15, 0.22])
    tls_issuer = rng.choice(tls_issuers, n_samples, p=[0.76, 0.10, 0.045, 0.095])

    # Latent log-odds. Several interactions deliberately favor tree/kernel models
    # over a purely linear boundary while leaving enough linear signal for
    # logistic regression to remain competitive.
    risk = (
        -4.7
        + 0.015 * (url_length - 55)
        + 0.23 * num_hyphens
        + 0.10 * num_digits
        + 0.31 * num_subdomains
        + 1.65 * has_ip_address
        + 1.35 * has_at_symbol
        - 1.05 * uses_https
        + 1.25 * suspicious_tld
        + 0.95 * shortened_url
        - 0.00075 * domain_age_days
        + 0.38 * redirect_count
        + 2.2 * external_link_ratio
        + 1.45 * request_url_ratio
        + 1.35 * anchor_url_ratio
        + 1.55 * form_action_external
        + 0.25 * iframe_count
        + 0.32 * popup_count
        + 0.55 * url_entropy
        + 1.7 * brand_mismatch
        + 0.65 * login_form
        - 0.004 * np.maximum(certificate_days_left, 0)
        - 0.022 * page_rank_proxy
        - 0.025 * (spelling_score - 75)
        + 1.1 * favicon_external
        + 1.35 * nonstandard_port
        + 1.35 * (tls_issuer == "self_signed")
        + 1.65 * (tls_issuer == "missing")
        + 0.65 * (tls_issuer == "unknown_ca")
    )

    # Nonlinear interactions.
    risk += 1.35 * ((url_length > 120) & (num_subdomains >= 3))
    risk += 1.65 * ((brand_mismatch == 1) & (login_form == 1))
    risk += 1.10 * ((uses_https == 0) & (form_action_external == 1))
    risk += 1.25 * ((domain_age_days < 90) & (suspicious_tld == 1))
    risk += 0.75 * ((page_category == "banking") & (brand_mismatch == 1))

    probability = 1.0 / (1.0 + np.exp(-np.clip(risk, -25, 25)))
    phishing = rng.binomial(1, probability)

    # Simulate imperfect labeling / ambiguous pages.
    flip = rng.random(n_samples) < 0.025
    phishing[flip] = 1 - phishing[flip]

    df = pd.DataFrame(
        {
            "url_length": np.round(url_length, 0),
            "num_dots": num_dots,
            "num_hyphens": num_hyphens,
            "num_digits": num_digits,
            "num_subdomains": num_subdomains,
            "has_ip_address": has_ip_address,
            "has_at_symbol": has_at_symbol,
            "uses_https": uses_https,
            "suspicious_tld": suspicious_tld,
            "shortened_url": shortened_url,
            "domain_age_days": np.round(domain_age_days, 0),
            "dns_record_age_days": np.round(dns_record_age_days, 0),
            "redirect_count": redirect_count,
            "external_link_ratio": np.round(external_link_ratio, 4),
            "request_url_ratio": np.round(request_url_ratio, 4),
            "anchor_url_ratio": np.round(anchor_url_ratio, 4),
            "form_action_external": form_action_external,
            "iframe_count": iframe_count,
            "popup_count": popup_count,
            "script_count": np.round(script_count, 0),
            "url_entropy": np.round(url_entropy, 3),
            "brand_mismatch": brand_mismatch,
            "login_form": login_form,
            "certificate_days_left": np.round(certificate_days_left, 0),
            "page_rank_proxy": np.round(page_rank_proxy, 2),
            "spelling_score": np.round(spelling_score, 2),
            "favicon_external": favicon_external,
            "nonstandard_port": nonstandard_port,
            "page_category": page_category,
            "tls_issuer": tls_issuer,
            "is_phishing": phishing.astype(int),
        }
    )

    for column, rate in {
        "domain_age_days": 0.018,
        "dns_record_age_days": 0.025,
        "certificate_days_left": 0.022,
        "page_rank_proxy": 0.012,
        "tls_issuer": 0.010,
    }.items():
        mask = rng.random(n_samples) < rate
        df.loc[mask, column] = np.nan

    return df


def generate_and_save(
    housing_path: Path = HOUSING_CSV,
    phishing_path: Path = PHISHING_CSV,
    *,
    seed: int = RANDOM_SEED,
) -> tuple[Path, Path]:
    """Generate both benchmark datasets and save them as CSV files."""

    housing = generate_housing_dataset(seed=seed)
    phishing = generate_phishing_dataset(seed=seed + 1)

    housing_path.parent.mkdir(parents=True, exist_ok=True)
    phishing_path.parent.mkdir(parents=True, exist_ok=True)
    housing.to_csv(housing_path, index=False)
    phishing.to_csv(phishing_path, index=False)
    return housing_path, phishing_path


def ensure_data() -> None:
    """Create the benchmark datasets only when they are missing."""

    if not HOUSING_CSV.exists() or not PHISHING_CSV.exists():
        generate_and_save()


def load_housing() -> pd.DataFrame:
    """Load the house-price benchmark, generating it if necessary."""

    ensure_data()
    return pd.read_csv(HOUSING_CSV)


def load_phishing() -> pd.DataFrame:
    """Load the phishing benchmark, generating it if necessary."""

    ensure_data()
    return pd.read_csv(PHISHING_CSV)
