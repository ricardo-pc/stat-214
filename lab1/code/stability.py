"""
stability.py - Bootstrap stability analyses and Bayesian inference for PECARN TBI.

Provides four stability analyses:
1. Bayesian posterior probabilities and Bayes factors for binary symptoms.
2. Bootstrap CIs for Bayes factors (1,000 resamples).
3. Bootstrap CIs for Kuppermann rule variants (original + augmented, >= 2 years).
4. Bootstrap CIs for Kuppermann overall metrics (1,000 resamples).
5. Bootstrap CIs for Logistic Regression (200 resamples, retrain each time).
6. Bootstrap CIs for Random Forest (200 resamples, retrain each time).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from models import kuppermann_predict


# ── Binary features used for Bayesian analysis ──
BINARY_FEATURES = [
    "altered_mental_status", "loc", "amnesia", "seizure",
    "skull_fx_palpable", "basilar_skull_fx", "fontanelle_bulging",
    "scalp_hematoma", "neuro_deficit", "vomiting", "headache",
    "acting_normal", "dizziness", "drug_intoxication", "other_injuries",
    "intubated", "paralyzed", "sedated",
]

# Symptoms used in Bayes factor bootstrap (subset of binary features)
BAYES_SYMPTOMS = [
    "skull_fx_palpable", "basilar_skull_fx", "altered_mental_status",
    "seizure", "neuro_deficit", "loc", "amnesia", "other_injuries",
    "vomiting", "scalp_hematoma", "headache", "acting_normal",
]


def bayesian_inference(df_prep):
    """Compute Bayesian posterior P(TBI | symptom) and Bayes factors.

    For each binary symptom, applies Bayes' theorem:
        P(TBI | symptom) = P(symptom | TBI) * P(TBI) / P(symptom)

    Parameters
    ----------
    df_prep : pd.DataFrame
        Cleaned PECARN dataset.

    Returns
    -------
    pd.DataFrame
        Columns: symptom, P(symptom|TBI), P(symptom), P(TBI|symptom),
        bayes_factor. Sorted by posterior descending.
    """
    prior_tbi = (df_prep["clinically_important_tbi"] == 1).sum() / len(df_prep)
    symptoms = [f for f in BINARY_FEATURES if f != "clinically_important_tbi"]
    tbi_patients = df_prep[df_prep["clinically_important_tbi"] == 1]

    results = []
    for symptom in symptoms:
        symptom_given_tbi_count = (tbi_patients[symptom] == 1).sum()
        likelihood = symptom_given_tbi_count / len(tbi_patients)

        symptom_count = (df_prep[symptom] == 1).sum()
        evidence = symptom_count / len(df_prep)

        if evidence > 0:
            posterior = (likelihood * prior_tbi) / evidence
        else:
            posterior = 0

        prior_odds = prior_tbi / (1 - prior_tbi)
        if posterior < 1:
            posterior_odds = posterior / (1 - posterior)
        else:
            posterior_odds = float("inf")
        bayes_factor = posterior_odds / prior_odds if prior_odds > 0 else 0

        results.append({
            "symptom": symptom,
            "P(symptom|TBI)": round(likelihood * 100, 1),
            "P(symptom)": round(evidence * 100, 1),
            "P(TBI|symptom)": round(posterior * 100, 2),
            "bayes_factor": round(bayes_factor, 1),
        })

    df_bayes = pd.DataFrame(results).sort_values(
        "P(TBI|symptom)", ascending=False
    )
    return df_bayes


def bootstrap_bayes_factors(df_prep, n_bootstrap=1000, seed=214):
    """Bootstrap 95% CIs for Bayes factors of each symptom.

    Parameters
    ----------
    df_prep : pd.DataFrame
        Cleaned PECARN dataset.
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys are symptom names. Values are lists of bootstrap Bayes factors.
    dict
        Keys are symptom names. Values are lists of bootstrap posteriors (%).
    """
    np.random.seed(seed)
    n_patients = len(df_prep)

    boot_posteriors = {s: [] for s in BAYES_SYMPTOMS}
    boot_bayes = {s: [] for s in BAYES_SYMPTOMS}

    for _ in range(n_bootstrap):
        idx = np.random.choice(n_patients, size=n_patients, replace=True)
        boot = df_prep.iloc[idx]

        prior = (boot["clinically_important_tbi"] == 1).sum() / len(boot)
        tbi = boot[boot["clinically_important_tbi"] == 1]

        for symptom in BAYES_SYMPTOMS:
            symptom_given_tbi = (
                (tbi[symptom] == 1).sum() / len(tbi) if len(tbi) > 0 else 0
            )
            evidence = (boot[symptom] == 1).sum() / len(boot)

            if evidence > 0 and prior > 0:
                posterior = (symptom_given_tbi * prior) / evidence
                prior_odds = prior / (1 - prior)
                post_odds = (
                    posterior / (1 - posterior)
                    if posterior < 1
                    else float("inf")
                )
                bf = post_odds / prior_odds
            else:
                posterior = 0
                bf = 0

            boot_posteriors[symptom].append(posterior * 100)
            boot_bayes[symptom].append(bf)

    return boot_bayes, boot_posteriors


def bootstrap_kuppermann_variants(df_prep, n_bootstrap=1000, seed=214):
    """Bootstrap CIs for Kuppermann rule variants (>= 2 years only).

    Tests four rule variants:
    - Original Kuppermann (>= 2 years)
    - + Scalp hematoma
    - + Moderate headache
    - + Both

    Parameters
    ----------
    df_prep : pd.DataFrame
        Cleaned PECARN dataset.
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int
        Random seed.

    Returns
    -------
    dict
        {variant_name: {"sens": [...], "spec": [...]}}
    """
    np.random.seed(seed)
    all_over2 = df_prep[df_prep["age_group"] == 2]
    n = len(all_over2)

    def _kuppermann_flags(data):
        base = (
            (data["altered_mental_status"] == 1)
            | (data["basilar_skull_fx"] == 1)
            | (data["loc"] == 1)
            | (data["vomiting"] == 1)
            | (data["injury_severity"] == 1)
            | (data["headache_severity"] == 3)
        )
        plus_scalp = base | (data["scalp_hematoma"] == 1)
        plus_headache = base | (data["headache_severity"] == 2)
        plus_both = (
            base
            | (data["scalp_hematoma"] == 1)
            | (data["headache_severity"] == 2)
        )
        return {
            "Original": base,
            "+ Scalp hematoma": plus_scalp,
            "+ Moderate headache": plus_headache,
            "+ Both": plus_both,
        }

    def _compute_sens_spec(flag, outcome):
        tp = (flag & (outcome == 1)).sum()
        fn = (~flag & (outcome == 1)).sum()
        tn = (~flag & (outcome == 0)).sum()
        fp = (flag & (outcome == 0)).sum()
        sens = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
        return sens, spec

    variant_names = [
        "Original", "+ Scalp hematoma", "+ Moderate headache", "+ Both"
    ]
    results = {
        name: {"sens": [], "spec": []} for name in variant_names
    }

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        boot_data = all_over2.iloc[idx]
        boot_outcome = boot_data["clinically_important_tbi"]

        flags = _kuppermann_flags(boot_data)
        for name, flag in flags.items():
            sens, spec = _compute_sens_spec(flag, boot_outcome)
            results[name]["sens"].append(sens)
            results[name]["spec"].append(spec)

    return results


def bootstrap_kuppermann(df_prep, n_bootstrap=1000, seed=214):
    """Bootstrap CIs for Kuppermann model metrics on full dataset.

    Parameters
    ----------
    df_prep : pd.DataFrame
        Cleaned PECARN dataset.
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Bootstrap sensitivities.
    np.ndarray
        Bootstrap specificities.
    np.ndarray
        Bootstrap precisions.
    """
    np.random.seed(seed)

    boot_sensitivity = []
    boot_specificity = []
    boot_precision = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(len(df_prep), size=len(df_prep), replace=True)
        boot_data = df_prep.iloc[idx]
        boot_y = boot_data["clinically_important_tbi"]
        boot_pred = kuppermann_predict(boot_data)

        tp = ((boot_pred == 1) & (boot_y == 1)).sum()
        fp = ((boot_pred == 1) & (boot_y == 0)).sum()
        fn = ((boot_pred == 0) & (boot_y == 1)).sum()
        tn = ((boot_pred == 0) & (boot_y == 0)).sum()

        sens = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0

        boot_sensitivity.append(sens)
        boot_specificity.append(spec)
        boot_precision.append(prec)

    return (
        np.array(boot_sensitivity),
        np.array(boot_specificity),
        np.array(boot_precision),
    )


def bootstrap_logistic_regression(
    X_train, X_test, y_train, y_test, threshold, n_bootstrap=200, seed=214
):
    """Bootstrap CIs for Logistic Regression by retraining on resampled data.

    Resamples the training set, retrains LR, evaluates on the same test set.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Feature matrices.
    y_train, y_test : pd.Series
        Labels.
    threshold : float
        Decision threshold (e.g., 0.29).
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Bootstrap sensitivities.
    np.ndarray
        Bootstrap specificities.
    np.ndarray
        Bootstrap precisions.
    np.ndarray
        Bootstrap coefficient arrays, shape (n_bootstrap, n_features).
    """
    np.random.seed(seed)

    boot_lr_sensitivity = []
    boot_lr_specificity = []
    boot_lr_precision = []
    boot_lr_coefs = []

    for i in range(n_bootstrap):
        idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_boot = X_train.iloc[idx]
        y_boot = y_train.iloc[idx]

        scaler_boot = StandardScaler()
        X_boot_scaled = scaler_boot.fit_transform(X_boot)
        X_test_scaled_boot = scaler_boot.transform(X_test)

        model_boot = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            penalty="l2",
            random_state=214,
        )
        model_boot.fit(X_boot_scaled, y_boot)

        y_proba_boot = model_boot.predict_proba(X_test_scaled_boot)[:, 1]
        y_pred_boot = (y_proba_boot >= threshold).astype(int)

        tp = ((y_pred_boot == 1) & (y_test == 1)).sum()
        fp = ((y_pred_boot == 1) & (y_test == 0)).sum()
        fn = ((y_pred_boot == 0) & (y_test == 1)).sum()
        tn = ((y_pred_boot == 0) & (y_test == 0)).sum()

        sens = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0

        boot_lr_sensitivity.append(sens)
        boot_lr_specificity.append(spec)
        boot_lr_precision.append(prec)
        boot_lr_coefs.append(model_boot.coef_[0])

        if (i + 1) % 50 == 0:
            print(f"  LR Bootstrap {i + 1}/{n_bootstrap} complete")

    return (
        np.array(boot_lr_sensitivity),
        np.array(boot_lr_specificity),
        np.array(boot_lr_precision),
        np.array(boot_lr_coefs),
    )


def bootstrap_random_forest(
    X_train, X_test, y_train, y_test,
    threshold, best_params, n_bootstrap=200, seed=214
):
    """Bootstrap CIs for Random Forest by retraining on resampled data.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Feature matrices.
    y_train, y_test : pd.Series
        Labels.
    threshold : float
        Decision threshold (e.g., 0.21).
    best_params : dict
        Must contain n_estimators, max_depth, min_samples_leaf.
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Bootstrap sensitivities.
    np.ndarray
        Bootstrap specificities.
    np.ndarray
        Bootstrap precisions.
    np.ndarray
        Bootstrap feature importances, shape (n_bootstrap, n_features).
    """
    np.random.seed(seed)

    boot_rf_sensitivity = []
    boot_rf_specificity = []
    boot_rf_precision = []
    boot_rf_importances = []

    for i in range(n_bootstrap):
        idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_boot = X_train.iloc[idx]
        y_boot = y_train.iloc[idx]

        rf_boot = RandomForestClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_leaf=best_params["min_samples_leaf"],
            max_features="sqrt",
            class_weight="balanced",
            random_state=214,
            n_jobs=-1,
        )
        rf_boot.fit(X_boot, y_boot)

        y_proba_boot = rf_boot.predict_proba(X_test)[:, 1]
        y_pred_boot = (y_proba_boot >= threshold).astype(int)

        tp = ((y_pred_boot == 1) & (y_test == 1)).sum()
        fp = ((y_pred_boot == 1) & (y_test == 0)).sum()
        fn = ((y_pred_boot == 0) & (y_test == 1)).sum()
        tn = ((y_pred_boot == 0) & (y_test == 0)).sum()

        sens = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0

        boot_rf_sensitivity.append(sens)
        boot_rf_specificity.append(spec)
        boot_rf_precision.append(prec)
        boot_rf_importances.append(rf_boot.feature_importances_)

        if (i + 1) % 50 == 0:
            print(f"  RF Bootstrap {i + 1}/{n_bootstrap} complete")

    return (
        np.array(boot_rf_sensitivity),
        np.array(boot_rf_specificity),
        np.array(boot_rf_precision),
        np.array(boot_rf_importances),
    )


if __name__ == "__main__":
    from clean import clean_data
    from models import (
        prepare_features,
        split_data,
        train_logistic_regression,
        train_random_forest,
    )

    # Load and clean
    raw_df = pd.read_csv("../data/TBI PUD 10-08-2013.csv")
    df_prep, _ = clean_data(raw_df)
    print(f"Data: {len(df_prep)} patients")

    # ── 1. Bayesian inference ──
    print("\n" + "=" * 60)
    print("Bayesian Inference")
    print("=" * 60)
    df_bayes = bayesian_inference(df_prep)
    print(df_bayes.to_string(index=False))

    # ── 2. Bootstrap Bayes factors ──
    print("\n" + "=" * 60)
    print("Bootstrap Bayes Factors (1,000 resamples)")
    print("=" * 60)
    boot_bf, boot_post = bootstrap_bayes_factors(df_prep)
    for s in BAYES_SYMPTOMS:
        bf_arr = np.array(boot_bf[s])
        lo, hi = np.percentile(bf_arr, [2.5, 97.5])
        print(f"  {s:<25} {bf_arr.mean():>5.1f}x  95% CI [{lo:.1f}, {hi:.1f}]")

    # ── 3. Bootstrap Kuppermann variants ──
    print("\n" + "=" * 60)
    print("Bootstrap Kuppermann Variants (1,000 resamples, >= 2 years)")
    print("=" * 60)
    variant_results = bootstrap_kuppermann_variants(df_prep)
    for name in ["Original", "+ Scalp hematoma", "+ Moderate headache", "+ Both"]:
        sens_arr = np.array(variant_results[name]["sens"])
        spec_arr = np.array(variant_results[name]["spec"])
        sens_lo, sens_hi = np.percentile(sens_arr, [2.5, 97.5])
        spec_lo, spec_hi = np.percentile(spec_arr, [2.5, 97.5])
        print(
            f"  {name:<25} Sens: {sens_arr.mean():.1f}% "
            f"[{sens_lo:.1f}, {sens_hi:.1f}]  "
            f"Spec: {spec_arr.mean():.1f}% "
            f"[{spec_lo:.1f}, {spec_hi:.1f}]"
        )

    # ── 4. Bootstrap Kuppermann overall ──
    print("\n" + "=" * 60)
    print("Bootstrap Kuppermann Overall (1,000 resamples)")
    print("=" * 60)
    k_sens, k_spec, k_prec = bootstrap_kuppermann(df_prep)
    for name, arr in [
        ("Sensitivity", k_sens),
        ("Specificity", k_spec),
        ("Precision", k_prec),
    ]:
        lo, hi = np.percentile(arr, [2.5, 97.5])
        print(f"  {name:<15} {arr.mean():>6.2f}%  95% CI [{lo:.2f}, {hi:.2f}]")

    # ── 5. Bootstrap LR ──
    print("\n" + "=" * 60)
    print("Bootstrap Logistic Regression (200 resamples)")
    print("=" * 60)
    X, y = prepare_features(df_prep)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    _, _, threshold_lr = train_logistic_regression(
        X_train, y_train, X_val, y_val
    )
    print(f"  LR threshold: {threshold_lr:.2f}")
    lr_sens, lr_spec, lr_prec, lr_coefs = bootstrap_logistic_regression(
        X_train, X_test, y_train, y_test, threshold_lr
    )
    for name, arr in [
        ("Sensitivity", lr_sens),
        ("Specificity", lr_spec),
        ("Precision", lr_prec),
    ]:
        lo, hi = np.percentile(arr, [2.5, 97.5])
        print(f"  {name:<15} {arr.mean():>6.2f}%  95% CI [{lo:.2f}, {hi:.2f}]")

    # ── 6. Bootstrap RF ──
    print("\n" + "=" * 60)
    print("Bootstrap Random Forest (200 resamples)")
    print("=" * 60)
    model_rf, threshold_rf = train_random_forest(
        X_train, y_train, X_val, y_val
    )
    best_params = {
        "n_estimators": model_rf.n_estimators,
        "max_depth": model_rf.max_depth,
        "min_samples_leaf": model_rf.min_samples_leaf,
    }
    print(f"  RF threshold: {threshold_rf:.2f}, params: {best_params}")
    rf_sens, rf_spec, rf_prec, rf_imps = bootstrap_random_forest(
        X_train, X_test, y_train, y_test, threshold_rf, best_params
    )
    for name, arr in [
        ("Sensitivity", rf_sens),
        ("Specificity", rf_spec),
        ("Precision", rf_prec),
    ]:
        lo, hi = np.percentile(arr, [2.5, 97.5])
        print(f"  {name:<15} {arr.mean():>6.2f}%  95% CI [{lo:.2f}, {hi:.2f}]")

    print("\nAll stability analyses complete.")
