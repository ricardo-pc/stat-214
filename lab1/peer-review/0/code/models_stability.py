'''
This file contains the code for training and evaluating the models for the stability analysis in Lab 1. It imports the necessary functions and classes from models.py, and then defines the specific models and evaluation procedures for the stability analysis.
'''
import os
from models import (
    make_cdr_data, make_lr_data, make_catboost_data, 
    build_best_lr_age2minus, build_best_lr_age2plus, 
    LR_Features_Age2minus, LR_Features_Age2plus,
    build_best_catboost_age2minus, build_best_catboost_age2plus, 
    CatBoost_Num_Features_Age2minus, CatBoost_Cat_Features_Age2minus,
    CatBoost_Num_Features_Age2plus, CatBoost_Cat_Features_Age2plus,
    metrics, PECARNDecisionRule,
)
import pandas as pd
import numpy as np

# CatBoost is optional
try:

    _HAS_CATBOOST = True
except Exception:
    _HAS_CATBOOST = False
    
#------
# CDR Prediction Stability Analysis
#------
def perturb_binary(series, flip_prob=0.05, rng=None):
    rng = rng or np.random.default_rng()
    s = series.copy()
    mask = rng.random(len(s)) < flip_prob
    # only flip known 0/1; leave missing as-is
    flip_idx = mask & s.isin([0, 1])
    s.loc[flip_idx] = 1 - s.loc[flip_idx]
    return s

def run_rule_perturbation_stability(df, rule,target="PosIntFinal",
                                    perturb_binary_cols=["AMS", "Vomit", "SFxBas", "ActNorm"],
                                   flip_prob=0.05, B=50, seed=42):
    rng = np.random.default_rng(seed)
    metrics = []

    base = df.copy()

    for b in range(B):
        d = base.copy()

        # Example: perturb a few subjective binary fields 
        for col in perturb_binary_cols:  
            if col in d.columns:
                d[col] = perturb_binary(d[col], flip_prob=flip_prob, rng=rng)

        y_true = d[target].astype(int).values
        y_pred = rule.predict(d)   # 1=not low risk, 0=very low risk (or your convention)

        # compute simple metrics
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()

        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        npv = tn / (tn + fn) if (tn + fn) else np.nan

        metrics.append({"rep": b, "sensitivity": sens, "specificity": spec, "npv": npv})

    return pd.DataFrame(metrics)




def main():
    # load data 
    tbi_cleaned_new = pd.read_csv('../data/TBI_cleaned.csv')
    output_dir = "../others"
    cdr_stability_metrics_file_name = "cdr_rule_perturbation_stability.csv"
    lr_stability_metrics_file_name = "lr_perturbation_stability.csv"
    cat_stability_metrics_file_name = "catboost_perturbation_stability.csv"

    target_col = 'PosIntFinal'

    cdr_rows = []
    lr_rows = []
    cat_rows = []

    def add_result(model_name, age_group, split_name, y_true, y_prob, rows, threshold=None, extra=None):
        m = metrics(y_true, y_prob, threshold=threshold)  # your existing function
        rec = {"model": model_name, 
                "age_group": age_group, 
                "threshold": threshold, 
                "split": split_name,
                **m}
        if extra:
            rec.update(extra)
        rows.append(rec)

    # CDR rule stability

    # ---- prepare CDR splits ----
    (
        train_cdr_data_age2minus,
        val_cdr_data_age2minus,
        test_cdr_data_age2minus,
        train_cdr_data_age2plus,
        val_cdr_data_age2plus,
        test_cdr_data_age2plus,
    ) = make_cdr_data(tbi_cleaned_new) 

    pecarn = PECARNDecisionRule() 

    # ---- run stability by split x agegroup ----
    experiments = [
        ("train", "<2", train_cdr_data_age2minus),
        ("val", "<2", val_cdr_data_age2minus),
        ("test", "<2", test_cdr_data_age2minus),
        ("train", ">=2", train_cdr_data_age2plus),
        ("val", ">=2", val_cdr_data_age2plus),
        ("test", ">=2", test_cdr_data_age2plus),
    ]

    # one can tweak these knobs
    flip_prob = 0.05
    B = 50
    seed = 42

    for split, age_group, df_sub in experiments:
        if df_sub is None or len(df_sub) == 0:
            continue

        stab = run_rule_perturbation_stability(
            df_sub,
            pecarn,
            target=target_col,
            flip_prob=flip_prob,
            B=B,
            seed=seed,
        )
        stab.insert(0, "model", "CDR")
        stab.insert(1, "split", split)
        stab.insert(2, "age_group", age_group)

        cdr_rows.append(stab)

    results = pd.concat(cdr_rows, ignore_index=True)
    
    out_path = os.path.join(output_dir, cdr_stability_metrics_file_name)
    results.to_csv(out_path, index=False)
    print(f"[OK] Saved CDR stability results to: {out_path}")

    # Logistic Regression stability

    # ----- prepare LR splits ----
    (
        train_lr_data_age2minus,
        val_lr_data_age2minus,
        test_lr_data_age2minus,
        train_lr_data_age2plus,
        val_lr_data_age2plus,
        test_lr_data_age2plus,
    ) = make_lr_data(tbi_cleaned_new, if_fill_missing_gcs= False) # not imputing missing GCS for LR stability since we want to see how perturbations affect the model trained on original data

    # ----- run stability by split x agegroup ----

    # ------ age < 2 ------
    lr_age2minus, threshold = build_best_lr_age2minus()
    lr_age2minus.fit(train_lr_data_age2minus[LR_Features_Age2minus], train_lr_data_age2minus[target_col])
    # save metrics for each split
    for split_name, d in [("train", train_lr_data_age2minus), ("val", val_lr_data_age2minus), ("test", test_lr_data_age2minus)]:
        y_true = d[target_col].astype(int).to_numpy()
        y_prob = lr_age2minus.predict_proba(d[LR_Features_Age2minus])[:, 1]  # probability of class 1
        add_result(
            model_name="Logistic Regression",
            age_group="Age < 2",
            split_name=split_name,
            y_true=y_true,
            rows=lr_rows,
            y_prob=y_prob,
            threshold=threshold,
            extra=None
        )

    # ------ age >= 2 ------
    lr_age2plus, threshold = build_best_lr_age2plus()
    lr_age2plus.fit(train_lr_data_age2plus[LR_Features_Age2plus], train_lr_data_age2plus[target_col])
    # save metrics for each split
    for split_name, d in [("train", train_lr_data_age2plus), ("val", val_lr_data_age2plus), ("test", test_lr_data_age2plus)]:
        y_true = d[target_col].astype(int).to_numpy()
        y_prob = lr_age2plus.predict_proba(d[LR_Features_Age2plus])[:, 1]  # probability of class 1
        add_result(
            model_name="Logistic Regression",
            age_group="Age >= 2",
            split_name=split_name,
            y_true=y_true,
            rows=lr_rows,
            y_prob=y_prob,
            threshold=threshold,
            extra=None
        )  

    lr_results = pd.DataFrame(lr_rows)
    out_path = os.path.join(output_dir, lr_stability_metrics_file_name)
    lr_results.to_csv(out_path, index=False)
    print(f"[OK] Saved Logistic Regression stability results to: {out_path}")


    # CatBoost stability 

    # ----- prepare CatBoost splits ----
    (
        train_cat_data_age2minus,
        val_cat_data_age2minus,
        test_cat_data_age2minus,
        train_cat_data_age2plus,
        val_cat_data_age2plus,
        test_cat_data_age2plus,
    ) = make_catboost_data(tbi_cleaned_new, if_fill_missing_gcs= False) # imputing missing GCS for CatBoost stability since tree-based models can handle imputed data and we want to see how perturbations affect the model trained on imputed data

    # ----- run stability by split x agegroup ----
    
    # ------ age < 2 ------
    if _HAS_CATBOOST:
        catboost_age2minus, threshold = build_best_catboost_age2minus()
        catboost_features_Age2minus = CatBoost_Num_Features_Age2minus + CatBoost_Cat_Features_Age2minus
        catboost_features_Age2minus.sort()  # sort features for consistent order
        catboost_age2minus.fit(
            train_cat_data_age2minus[catboost_features_Age2minus], 
            train_cat_data_age2minus[target_col],
            cat_features=CatBoost_Cat_Features_Age2minus,
            eval_set=(val_cat_data_age2minus[catboost_features_Age2minus], val_cat_data_age2minus[target_col]))


        for split_name, d in [("train", train_cat_data_age2minus), ("val", val_cat_data_age2minus), ("test", test_cat_data_age2minus)]:
            y_true = d[target_col].astype(int).to_numpy()
            y_prob = catboost_age2minus.predict_proba(d[catboost_features_Age2minus])[:, 1]
            add_result(
                model_name="CatBoost",
                age_group="Age < 2",
                split_name=split_name,
                y_true=y_true,
                rows=cat_rows,
                y_prob=y_prob,
                threshold=threshold,
                extra=None
            )
    # ------ age >= 2 ------
    if _HAS_CATBOOST:
        catboost_age2plus, threshold = build_best_catboost_age2plus()
        catboost_features_Age2plus = CatBoost_Num_Features_Age2plus + CatBoost_Cat_Features_Age2plus
        catboost_features_Age2plus.sort()  # sort features for consistent order
        catboost_age2plus.fit(
            train_cat_data_age2plus[catboost_features_Age2plus], 
            train_cat_data_age2plus[target_col],
            cat_features=CatBoost_Cat_Features_Age2plus,
            eval_set=(val_cat_data_age2plus[catboost_features_Age2plus], val_cat_data_age2plus[target_col]))

        for split_name, d in [("train", train_cat_data_age2plus), ("val", val_cat_data_age2plus), ("test", test_cat_data_age2plus)]:
            y_true = d[target_col].astype(int).to_numpy()
            y_prob = catboost_age2plus.predict_proba(d[catboost_features_Age2plus])[:, 1]
            add_result(
                model_name="CatBoost",
                age_group="Age >= 2",
                split_name=split_name,
                y_true=y_true,
                rows=cat_rows,
                y_prob=y_prob,
                threshold=threshold,
                extra=None
            )

    cat_results = pd.DataFrame(cat_rows)
    out_path = os.path.join(output_dir, cat_stability_metrics_file_name)
    cat_results.to_csv(out_path, index=False)
    print(f"[OK] Saved CatBoost stability results to: {out_path}")



if __name__ == "__main__":
    main()