import sys
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)
from clean import clean_data

#configs
TARGET_DEFAULT = "CTDone"         
RANDOM_STATE_DEFAULT = 42
TEST_SIZE_DEFAULT = 0.30
THRESHOLD = 0.5

FEATURES = [
    "AgeinYears",
    "AMS",                 # 0/1
    "LOCSeparate",         # 0/1/2 (2 = Suspected)
    "Vomit",               # 0/1/2 (2 = Unknown)
    "High_impact_InjSev",  # 1/2/3/4 (4=Undocumented)
    "SFxPalp",             # 0/1/2
    "SFxBas",              # 0/1/2
    "ActNorm",             # 0/1/2 (2 = Unknown)
]

@dataclass
class ModelResult:
    name: str
    y_pred: np.ndarray
    y_prob: Optional[np.ndarray] = None


def require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray], name: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["accuracy"] = accuracy_score(y_true, y_pred)
    out["precision"] = precision_score(y_true, y_pred, zero_division=0)
    out["recall"] = recall_score(y_true, y_pred, zero_division=0)

    if y_prob is not None:
        if len(np.unique(y_true)) == 2:
            out["auc"] = roc_auc_score(y_true, y_prob)
        else:
            out["auc"] = np.nan
    else:
        out["auc"] = np.nan

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["tn"] = tn
    out["fp"] = fp
    out["fn"] = fn
    out["tp"] = tp

    print(f"\n{name}")
    print(f"  Accuracy : {out['accuracy']:.4f}")
    print(f"  Precision: {out['precision']:.4f}")
    print(f"  Recall   : {out['recall']:.4f}")
    if not np.isnan(out["auc"]):
        print(f"  AUC      : {out['auc']:.4f}")
    print(f"  Confusion Matrix [tn fp; fn tp]: [{tn} {fp}; {fn} {tp}]")

    return out



#MODEL 1: Clinical decision rule
def pecarn_rule(dfX: pd.DataFrame) -> np.ndarray:
    """
    The clinical decision rule from Kuppermann et al.
    Returns: 1 = recommend CT, 0 = do not recommend
    """
    require_columns(dfX, FEATURES)
    high_risk = (
        (dfX["AMS"] == 1) |
        (dfX["SFxPalp"].isin([1, 2])) |
        (dfX["SFxBas"] == 1) |
        (dfX["High_impact_InjSev"] == 3) |
        (dfX["LOCSeparate"] == 1) |
        (dfX["Vomit"] == 1) |
        (dfX["ActNorm"] == 0)
    )

    return high_risk.astype(int).to_numpy()


#MODEL 2: Logistic regression
def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(penalty="l2", max_iter=2000))
    ])
    model.fit(X_train, y_train)
    return model


#MODEL 3: Random Forest
def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, random_state: int) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf



def prepare_X_y(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    require_columns(df, FEATURES + [target])

    X = df[FEATURES].copy()
    y = df[target].copy()

    return X, y


def plot_roc_curves(y_test: np.ndarray, results: List[ModelResult]) -> None:
    plt.figure(figsize=(7, 6))
    for res in results:
        if res.y_prob is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, res.y_prob)
        plt.plot(fpr, tpr, label=res.name)

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/model_accuracy.png", dpi=300)
    print("Saved ROC plot to: outputs/model_accuracy.png")

def feature_importance_plot(rf_model, X_train):
    label_map = {
    "ActNorm": "Not Acting Normal",
    "LOCSeparate": "Loss of Consciousness",
    "Vomit": "Vomiting",
    "AMS": "Altered Mental Status",
    "AgeinYears": "Age in Years",
    "High_impact_InjSev": "Severity of injury Mech",
    "SFxPalp": "Palpable Scalp Fracture",
    "SFxBas": "Basilar Skull Fracture"
}

    
    importances = rf_model.feature_importances_
    feat_importance = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
    feat_importance_named = feat_importance.rename(index=label_map)

    plt.figure(figsize=(8,6))
    feat_importance_named.head(15).plot(kind='bar')
    plt.title("Top Feature Importances (Random Forest)")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.savefig("outputs/rf_feature_importance.png", dpi=300)
    print("Saved ROC plot to: outputs/rf_feature_importance.png")

def run_models(
    df,
    target: str,
    test_size: float,
    random_state: int,
    threshold: float
) -> None:

    X, y = prepare_X_y(df, target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(np.unique(y)) == 2 else None
    )

    #CDR
    y_pred_cdr = pecarn_rule(X_test)
    res_cdr = ModelResult(name="PECARN-Style Rule", y_pred=y_pred_cdr, y_prob=None)

    #Logistic regression
    log_model = train_logistic_regression(X_train, y_train)
    y_prob_log = log_model.predict_proba(X_test)[:, 1]
    y_pred_log = (y_prob_log >= threshold).astype(int)
    res_log = ModelResult(name=f"Logistic Regression (thr={threshold})", y_pred=y_pred_log, y_prob=y_prob_log)

    #Random forest
    rf_model = train_random_forest(X_train, y_train, random_state=random_state)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    y_pred_rf = (y_prob_rf >= threshold).astype(int)
    res_rf = ModelResult(name=f"Random Forest (thr={threshold})", y_pred=y_pred_rf, y_prob=y_prob_rf)

    #evaluate model
    evaluate_model(y_test.to_numpy(), res_cdr.y_pred, None, res_cdr.name)
    evaluate_model(y_test.to_numpy(), res_log.y_pred, res_log.y_prob, res_log.name)
    evaluate_model(y_test.to_numpy(), res_rf.y_pred, res_rf.y_prob, res_rf.name)

    #ROC plot
    plot_roc_curves(y_test.to_numpy(), [res_cdr,res_log, res_rf])
    feature_importance_plot(rf_model, X_train)


if __name__ == "__main__":

    raw_df = pd.read_csv("../data/TBI PUD 10-08-2013.csv")
    df = clean_data(raw_df)
    if len(sys.argv) > 1 and sys.argv[1] == "perturb":
        # Change 4 (Undocumented) to 1 (Low)
        df["High_impact_InjSev"] = df["High_impact_InjSev"].replace({4: 1})
        
    try:
        run_models(
            df,
            target=TARGET_DEFAULT,
            test_size=TEST_SIZE_DEFAULT,
            random_state=RANDOM_STATE_DEFAULT,
            threshold=THRESHOLD,
        )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)