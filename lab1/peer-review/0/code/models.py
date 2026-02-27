from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score

from preprocess import preprocess_data
from features import (
    binary_cols,
    categorical_cols,
    multi_category_cols,
    numeric_cols,
    parent_child_groups
)

# CatBoost is optional
try:
    from catboost import CatBoostClassifier  # type: ignore

    _HAS_CATBOOST = True
except Exception:
    _HAS_CATBOOST = False

# selected features for logistic regression 
LR_Features_Age2minus= ['GCSMotor', 'GCSVerbal','GCSEye','HemaLoc_2.0','HemaLoc_3.0','High_impact_InjSev_3.0',
                   'High_impact_InjSev_missing','NeuroD_missing','AMS','AMSOth_1','ActNorm','ActNorm_missing',
                   'ClavOcc_1','ClavPar_1','SFxPalp_2.0','Seiz','LOCSeparate_missing','VomitStart_3.0']

LR_Features_Age2plus = ["SFxBas","GCSGroup","High_impact_InjSev_2.0","High_impact_InjSev_3.0","NeuroD",
    "NeuroD_missing","OSIOth_1","AMSAgitated_1","SFxPalp_2.0","AMS","ClavPar_1","LocLen_missing",
    "LOCSeparate_missing","ClavNeck_1","HAStart_3.0","HAStart_missing","Vomit","VomitStart_2.0",
    "VomitLast_missing","VomitStart_3.0","ActNorm","ActNorm_missing","Dizzy_missing","HemaSize_3.0","HASeverity_3.0"]

# selected features for CatBoost 
CatBoost_Cat_Features_Age2minus = ['ClavPar', 'AMSRepeat', 'HemaLoc', 'Vomit', 'SeizOccur', 'LocLen', 'OSIFlank', 'SFxBas', 'OSICspine', 'OSIAbdomen', 'OSI', 'Dizzy', 'Amnesia_verb', 'ClavNeck', 'Drugs', 'SFxPalp', 'AMS', 'HA_verb', 'Seiz', 'Hema', 'High_impact_InjSev', 'Clav', 'ClavOcc', 'SeizLen', 'FontBulg', 'Race', 'AMSSlow', 'NeuroD', 'VomitNbr', 'ActNorm', 'VomitLast', 'Intubated', 'HemaSize', 'VomitStart', 'LOCSeparate', 'OSIExtremity', 'AMSAgitated', 'OSIOth', 'AMSOth', 'ClavTem', 'InjuryMech', 'GCSGroup', 'OSIPelvis', 'OSICut', 'Sedated', 'AMSSleep']

CatBoost_Num_Features_Age2minus = ['GCSTotal', 'GCSEye', 'GCSVerbal', 'GCSMotor']


CatBoost_Cat_Features_Age2plus = ['ClavPar', 'AMSRepeat', 'HemaLoc', 'HASeverity', 'Vomit', 'NeuroDCranial', 'SeizOccur', 'LocLen', 'OSIFlank', 'SFxBas', 'NeuroDOth', 'OSICspine', 'OSIAbdomen', 'OSI', 'Dizzy', 'Amnesia_verb', 'ClavNeck', 'Gender', 'SFxBasRet', 'Drugs', 'OSICut', 'SFxPalp', 'SFxBasOto', 'NeuroDSensory', 'AMS', 'HA_verb', 'Seiz', 'Hema', 'High_impact_InjSev', 'Clav', 'ClavOcc', 'SeizLen', 'Race', 'SFxBasRhi', 'AMSSlow', 'NeuroD', 'VomitNbr', 'NeuroDMotor', 'ActNorm', 'VomitLast', 'SFxBasHem', 'Intubated', 'ClavFace', 'HemaSize', 'VomitStart', 'SFxBasPer', 'LOCSeparate', 'Paralyzed', 'OSIExtremity', 'AMSAgitated', 'OSIOth', 'AMSOth', 'ClavTem', 'NeuroDReflex', 'InjuryMech', 'GCSGroup', 'OSIPelvis', 'HAStart', 'ClavFro', 'Sedated', 'AMSSleep']


CatBoost_Num_Features_Age2plus = ['AgeinYears', 'GCSMotor', 'GCSEye', 'GCSVerbal', 'GCSTotal']

# Prepare data for each model

def make_cdr_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare data for clinical decision rule

    Returns a DataFrame ready for CDR prediction.
    """
    train_cdr_data, val_cdr_data, test_cdr_data = preprocess_data(
    df=df,
    numeric_cols=numeric_cols,
    categorical_cols=categorical_cols,
    multi_category_cols=multi_category_cols,
    binary_cols=binary_cols,
    parent_child_groups=parent_child_groups,
    target_col='PosIntFinal',
    test_size=0.2,
    val_size=0.1,
    random_state=42,
    stratify_col=['PosIntFinal', 'AgeTwoPlus'],
    if_exclude_gcs_under_13= True, # drop GCS 3-13 as paper
    if_fill_missing_gcs=True,
    gcs_fill_strategy="leave", # leave missing 
    if_one_hot_encode=False,
    drop_first_cat_in_ohe=True,
    if_drop_na_rows=False,
    if_handle_parent_child_missing=True,
    if_handle_missing_for_categories=True,
    missing_category_label="missing",
    if_add_flag_missing_cols=False,
    parent_missing_strategy="leave",
    child_missing_when_parent_yes="missing_category",
    binary_missing_strategy="leave",
    multi_missing_strategy="missing_category"
)
    train_cdr_data_age2minus = train_cdr_data[train_cdr_data['AgeTwoPlus'] == 1]
    train_cdr_data_age2plus = train_cdr_data[train_cdr_data['AgeTwoPlus'] == 2]
    val_cdr_data_age2minus = val_cdr_data[val_cdr_data['AgeTwoPlus'] == 1]
    val_cdr_data_age2plus = val_cdr_data[val_cdr_data['AgeTwoPlus'] == 2]
    test_cdr_data_age2minus = test_cdr_data[test_cdr_data['AgeTwoPlus'] == 1]
    test_cdr_data_age2plus = test_cdr_data[test_cdr_data['AgeTwoPlus'] == 2]
    return train_cdr_data_age2minus, val_cdr_data_age2minus, test_cdr_data_age2minus, train_cdr_data_age2plus, val_cdr_data_age2plus, test_cdr_data_age2plus


def make_lr_data(df: pd.DataFrame, 
                 if_fill_missing_gcs = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare data for logistic regression.
    Returns DataFrames ready for logistic regression.

    1.impute missing GCS values 
    2. flag binary categorical missingness with additional columns (e.g. AMS_missing) to allow the model to learn from missingness patterns, which can be informative in clinical data.
    3. Handle parent-child missingness: if a parent variable is missing, flag it and fill 0. If the parent is yes (e.g. 1), set child missing to a "missing_category" 
    4. Handle missing values for multi-category variables by adding a "missing" category and setting missing values to that category
    5. One-hot encode categorical variables, dropping the first category to avoid multicollinearity in logistic regression. 
    6. Drop any remaining rows with NA after all the above missing value handling steps
    7. Stratify the resulting train/val/test splits by both the target variable and the age group 
    """
    train_lr_data, val_lr_data, test_lr_data = preprocess_data(
        df=df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        multi_category_cols=multi_category_cols,
        binary_cols=binary_cols,
        parent_child_groups=parent_child_groups,
        target_col='PosIntFinal',
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        stratify_col=['PosIntFinal', 'AgeTwoPlus'],
        if_exclude_gcs_under_13= False, # include GCS 3-13 different from paper
        if_fill_missing_gcs=if_fill_missing_gcs,
        gcs_fill_strategy="leave", # leave missing 
        if_one_hot_encode=True,
        drop_first_cat_in_ohe=True,
        if_drop_na_rows=True, # after all the missing values are handled, drop any remaining rows with NA
        if_handle_parent_child_missing=True,
        if_handle_missing_for_categories=True,
        missing_category_label="missing",
        if_add_flag_missing_cols=True, # add flag columns to indicate missingness for each feature
        parent_missing_strategy="fill0",
        child_missing_when_parent_yes="missing_category",
        binary_missing_strategy="fill0",
        multi_missing_strategy="missing_category"
    )

    train_lr_data_age2minus = train_lr_data[train_lr_data['AgeTwoPlus'] == 1]
    train_lr_data_age2plus = train_lr_data[train_lr_data['AgeTwoPlus'] == 2]
    val_lr_data_age2minus = val_lr_data[val_lr_data['AgeTwoPlus'] == 1]
    val_lr_data_age2plus = val_lr_data[val_lr_data['AgeTwoPlus'] == 2]
    test_lr_data_age2minus = test_lr_data[test_lr_data['AgeTwoPlus'] == 1]
    test_lr_data_age2plus = test_lr_data[test_lr_data['AgeTwoPlus'] == 2]

    return train_lr_data_age2minus, val_lr_data_age2minus, test_lr_data_age2minus, train_lr_data_age2plus, val_lr_data_age2plus, test_lr_data_age2plus

def make_catboost_data(df: pd.DataFrame, 
                       if_fill_missing_gcs = True,
                       model_numeric_cols = list(set(CatBoost_Num_Features_Age2minus).union(set(CatBoost_Num_Features_Age2plus))),
                       model_categorical_cols = list(set(CatBoost_Cat_Features_Age2minus).union(set(CatBoost_Cat_Features_Age2plus)))
                       ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    """Prepare data for CatBoost.
    Returns a DataFrame ready for CatBoost.
    1. Preprocess data with the same steps as for logistic regression, but do NOT one-hot encode categorical variables (CatBoost can handle them natively). Handle missing values according to the same strategies as for logistic regression to ensure a fair comparison.
    2. After preprocessing, convert categorical columns to 'category' dtype and fill any remaining NaN in categorical columns with a "missing" category to ensure CatBoost can handle them properly.
    3. Stratify the resulting train/val/test splits by both the target variable and the age group 
    4. Return the processed train/val/test DataFrames for both age groups separately, ready for model training and evaluation.
    """

    train_cat_data, val_cat_data, test_cat_data = preprocess_data(
    df=df,
    numeric_cols=numeric_cols,
    categorical_cols=categorical_cols,
    multi_category_cols=multi_category_cols,
    binary_cols=binary_cols,
    parent_child_groups=parent_child_groups,
    target_col='PosIntFinal',
    test_size=0.2,
    val_size=0.1,
    random_state=42,
    stratify_col=['PosIntFinal', 'AgeTwoPlus'],
    if_exclude_gcs_under_13= False, # include GCS 3-13
    if_fill_missing_gcs=if_fill_missing_gcs,
    gcs_fill_strategy="leave", # leave missing 
    if_one_hot_encode=False,
    drop_first_cat_in_ohe=False,
    if_drop_na_rows=False,
    if_handle_parent_child_missing=False,
    if_handle_missing_for_categories=False,
    missing_category_label="missing",
    if_add_flag_missing_cols=False,
    parent_missing_strategy="leave",
    child_missing_when_parent_yes="missing_category",
    binary_missing_strategy="leave",
    multi_missing_strategy="missing_category"
)
    # convert categorical features to category dtype for CatBoost
    train_cat_data[model_numeric_cols] = train_cat_data[model_numeric_cols].apply(pd.to_numeric, errors='coerce')
    # convert category columns to string
    train_cat_data[model_categorical_cols] = (
        train_cat_data[model_categorical_cols]
            .apply(lambda col: col.astype("string").astype("category"))
    )
    # fill NaN with `missing`
    train_cat_data[model_categorical_cols] = (
        train_cat_data[model_categorical_cols]
            .apply(lambda col: col.cat.add_categories("missing").fillna("missing"))
    )
    # val
    val_cat_data[model_numeric_cols] = val_cat_data[model_numeric_cols].apply(pd.to_numeric, errors='coerce')
    val_cat_data[model_categorical_cols] = (
        val_cat_data[model_categorical_cols]
            .apply(lambda col: col.astype("string").astype("category"))
    )
    # fill NaN with `missing`
    val_cat_data[model_categorical_cols] = (
        val_cat_data[model_categorical_cols]
            .apply(lambda col: col.cat.add_categories("missing").fillna("missing"))
    )
    # test
    test_cat_data[model_numeric_cols] = test_cat_data[model_numeric_cols].apply(pd.to_numeric, errors='coerce')
    test_cat_data[model_categorical_cols] = (
        test_cat_data[model_categorical_cols]
            .apply(lambda col: col.astype("string").astype("category"))
    )
    # fill NaN with `missing`
    test_cat_data[model_categorical_cols] = (
        test_cat_data[model_categorical_cols]
            .apply(lambda col: col.cat.add_categories("missing").fillna("missing"))
    )
    # age stratification
    train_cat_data_age2minus = train_cat_data[train_cat_data['AgeTwoPlus'] == 1]
    train_cat_data_age2plus = train_cat_data[train_cat_data['AgeTwoPlus'] == 2]
    
    val_cat_data_age2minus = val_cat_data[val_cat_data['AgeTwoPlus'] == 1]
    val_cat_data_age2plus = val_cat_data[val_cat_data['AgeTwoPlus'] == 2]

    test_cat_data_age2minus = test_cat_data[test_cat_data['AgeTwoPlus'] == 1]
    test_cat_data_age2plus = test_cat_data[test_cat_data['AgeTwoPlus'] == 2]

    return train_cat_data_age2minus, val_cat_data_age2minus, test_cat_data_age2minus, train_cat_data_age2plus, val_cat_data_age2plus, test_cat_data_age2plus
    

    
# --------------------------
# Metrics
# --------------------------

def metrics(y_true: Union[pd.Series, np.ndarray], y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Calculate performance metrics for binary classification at a given threshold."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred = (y_prob >= threshold).astype(int) if threshold is not None else y_prob.astype(int)

    tp = int(((y_true_arr == 1) & (y_pred == 1)).sum())
    fn = int(((y_true_arr == 1) & (y_pred == 0)).sum())
    tn = int(((y_true_arr == 0) & (y_pred == 0)).sum())
    fp = int(((y_true_arr == 0) & (y_pred == 1)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    if threshold is None:
        # For probability metrics, we need a binary classification context. If no threshold is provided, we cannot compute these.
        roc_auc = np.nan
        pr_auc = np.nan
    else:
        roc_auc = roc_auc_score(y_true_arr, y_prob) if len(np.unique(y_true_arr)) > 1 else np.nan
        pr_auc = average_precision_score(y_true_arr, y_prob) if len(np.unique(y_true_arr)) > 1 else np.nan

    precision = precision_score(y_true_arr, y_pred, zero_division=0)

    return {
        "threshold": float(threshold) if threshold is not None else None,
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "NPV": float(npv),
        "ROC_AUC": float(roc_auc),
        "PR_AUC": float(pr_auc),
        "Precision": float(precision),
        "TP": tp,
        "FN": fn,
        "TN": tn,
        "FP": fp,
        "n": int(len(y_true_arr)),
        "pos": int(y_true_arr.sum()),
    }

def _safe_prob_from_pred(y_pred: np.ndarray) -> np.ndarray:
    """Convert hard 0/1 predictions to 'probabilities' for a unified interface."""
    y_pred = np.asarray(y_pred).astype(float)
    return y_pred  # 0/1 already



# ---------------------------
# Clinical Decision Rule
# ---------------------------


def _is_missing(x: Any) -> bool:
    if pd.isna(x):
        return True
    if isinstance(x, str) and x.strip() == '':
        return True
    return False

@dataclass
class PECARNColumns:
    """
    Default column mapping for your dataset (adjust if your names differ).

    Age group:
      - AgeTwoPlus: 1 => <2 years, 2 => >=2 years  (per your preprocessing)
    <2 years rule variables:
      - AMS
      - HemaLoc: location of scalp hematoma (needs to detect occipital/parietal/temporal vs none/frontal)
      - LOCSeparate + LocLen: LOC indicator and duration (seconds); rule uses >=5 seconds
      - High_impact_InjSev: severe mechanism indicator (1/0) or a coded severity you can threshold
      - SFxPalp: palpable skull fracture
      - ActNorm: acting normally per parent (1 yes / 0 no)
    >=2 years rule variables:
      - AMS
      - LOCSeparate (any LOC)
      - Vomit (history of vomiting)
      - High_impact_InjSev (severe mechanism)
      - SFxBas (basilar skull fracture signs)
      - HASeverity (severe headache)  (you can map to severe if HASeverity==1 etc)
    """

    AgeTwoPlus: str = "AgeTwoPlus"

    # shared
    AMS: str = "AMS"
    High_impact_InjSev: str = "High_impact_InjSev"

    # <2
    HemaLoc: str = "HemaLoc"
    LocLen: str = "LocLen"
    SFxPalp: str = "SFxPalp"
    ActNorm: str = "ActNorm"

    # >=2
    Vomit: str = "Vomit"
    SFxBas: str = "SFxBas"
    HASeverity: str = "HASeverity"
    LOCSeparate: str = "LOCSeparate"
    
class PECARNDecisionRule:
    """
    Replicates the PECARN 'very low risk' rules (Kuppermann et al., 2009):
      - Separate rules for age < 2 and age >= 2.
      - Output: 0 => very low risk (no predictors present)
                1 => not very low risk (>=1 predictor present OR unknown required field)

    IMPORTANT (clinical / modeling-safe):
      - Any unknown/missing on a required decision node is treated conservatively as "not very low risk".
    """

    def __init__(
        self,
        cols: Optional[PECARNColumns] = None,
        *,
        loc_severe_codes: Optional[Iterable[Any]] = {1,2},
        loc_len_ge_5s_code: {Any} = {2, 3, 4},
        severe_mech_codes: Optional[Iterable[Any]] = {3},
        severe_headache_codes: Optional[Iterable[Any]] = {3},
        hema_opt_codes: Optional[Iterable[Any]] = {2,3},
        palpalable_sfx_codes: Optional[Iterable[Any]] = {1,2},
    ):
        self.cols = cols or PECARNColumns()
        self.loc_len_ge_5s_code = loc_len_ge_5s_code
        self.loc_severe_codes = set(loc_severe_codes) if loc_severe_codes is not None else set()
        self.severe_mech_codes = set(severe_mech_codes) if severe_mech_codes is not None else set()
        self.severe_headache_codes = (
            set(severe_headache_codes) if severe_headache_codes is not None else set()
        )
        self.hema_opt_codes = set(hema_opt_codes) if hema_opt_codes is not None else set()
        self.palpalable_sfx_codes = set(palpalable_sfx_codes) if palpalable_sfx_codes is not None else set()    

    # --- helpers to interpret coded fields ---
    def _is_ams(self, x: Any) -> Optional[int]:
        if _is_missing(x):
            return None
        return int(x == 1)
    def _is_vomit(self, x: Any) -> Optional[int]:
        if _is_missing(x):
            return None
        return int(x == 1)
    def _is_bas_skull(self, x: Any) -> Optional[int]:
        if _is_missing(x):
            return None
        return int(x == 1)

    def _is_severe_mechanism(self, x: Any) -> Optional[int]:
        if _is_missing(x):
            return None
        if self.severe_mech_codes is None:
            # no mapping provided and not 0/1 -> unknown
            return None
        return int(x in self.severe_mech_codes)

    def _is_severe_headache(self, x: Any) -> Optional[int]:
        if _is_missing(x):
            return None
        if self.severe_headache_codes is None:
            return None
        return int(x in self.severe_headache_codes)

    def _hema_is_opt(self, x: Any) -> Optional[int]:
        if _is_missing(x):
            return None
        if self.hema_opt_codes is None:
            return None
        return int(x in self.hema_opt_codes)

    def _loc_ge_5s(self, loc_sep: Any, loc_len: Any) -> Optional[int]:
        if _is_missing(loc_len ):
            return None
        if self.loc_len_ge_5s_code is None:
            return None
        return int(loc_len in self.loc_len_ge_5s_code)
    
    def _loc_severe(self, loc_sep: Any) -> Optional[int]:
        if _is_missing(loc_sep):
            return None
        if self.loc_severe_codes is None:
            return None
        return int(loc_sep in self.loc_severe_codes)
    
    def _sfx_palp_severe(self, x: Any) -> Optional[int]:
        if _is_missing(x):
            return None
        if self.palpalable_sfx_codes is None:
            return None
        return int(x in self.palpalable_sfx_codes)
    
    def _act_norm_not_normal(self, x: Any) -> Optional[int]:
        if _is_missing(x):
            return None
        return int(x == 0)

    
    # --- core rules ---

    def _predict_row(self, row: pd.Series) -> Dict[str, Any]:
        c = self.cols

        agegrp = row.get(c.AgeTwoPlus, None)

        # Normalize age group to bool: <2 => 1, >=2 => 2 (your convention)
        is_under2 = (agegrp == 1) 

        reasons = []

        # Shared: AMS is a predictor for both trees
        ams = self._is_ams(row.get(c.AMS))
        if ams is None:
            reasons.append("AMS missing/unknown")
        elif ams == 1:
            reasons.append("Altered mental status = Yes")

        severe_mech = self._is_severe_mechanism(row.get(c.High_impact_InjSev))
        if severe_mech is None:
            # in the tree it's used later, but missing still prevents 'rule out' if reached
            pass

        if is_under2:
            # <2 years predictors (any one => not very low risk)
            hema_opt = self._hema_is_opt(row.get(c.HemaLoc))
            loc_ge5 = self._loc_ge_5s(row.get(c.LOCSeparate), row.get(c.LocLen))
            sfx_palp = self._sfx_palp_severe(row.get(c.SFxPalp))
            act_norm = self._act_norm_not_normal(row.get(c.ActNorm))

            # The derivation tree structure is sequential, but the published "no predictor present"
            # summary corresponds to: none of these predictors are present.
            # Missing in any predictor => cannot be confidently "no predictor present".
            predictors = {
                "AMS": ams,
                "Scalp hematoma (O/P/T)": hema_opt,
                "LOC >= 5s": loc_ge5,
                "Severe mechanism": severe_mech,
                "Palpable skull fracture": sfx_palp,
                "Not acting normally per parent": act_norm,
            }

            for name, val in predictors.items():
                if val is None:
                    reasons.append(f"{name} missing/unknown")
                elif val == 1:
                    reasons.append(f"{name} present")

            # very low risk only if ALL predictors are known and ==0
            if all(v == 0 for v in predictors.values() if v is not None):
                return {"very_low_risk": 1, "rule": "<2", "reasons": ["No PECARN predictors present"]}
            return {"very_low_risk": 0, "rule": "<2", "reasons": reasons}

        else:
            # >=2 years predictors
            loc_any = self._loc_severe(row.get(c.LOCSeparate))
            vomit = self._is_vomit(row.get(c.Vomit))
            sfx_bas = self._is_bas_skull(row.get(c.SFxBas))
            ha_severe = self._is_severe_headache(row.get(c.HASeverity))

            predictors = {
                "AMS": ams,
                "Any LOC": loc_any,
                "History of vomiting": vomit,
                "Severe mechanism": severe_mech,
                "Basilar skull fracture signs": sfx_bas,
                "Severe headache": ha_severe,
            }

            for name, val in predictors.items():
                if val is None:
                    reasons.append(f"{name} missing/unknown")
                elif val == 1:
                    reasons.append(f"{name} present")

            if all(v == 0 for v in predictors.values() if v is not None):
                return {"very_low_risk": 1, "rule": ">=2", "reasons": ["No PECARN predictors present"]}
            return {"very_low_risk": 0, "rule": ">=2", "reasons": reasons}

    # --- public API ---

    def predict(self, X: Union[pd.DataFrame, Dict[str, Any]]) -> np.ndarray:
        """
        Returns array y_hat where:
          0 => very low risk (no predictors present)   [safe to avoid CT under PECARN context]
          1 => not very low risk
        """
        if isinstance(X, dict):
            row = pd.Series(X)
            out = self._predict_row(row)
            return np.array([0 if out["very_low_risk"] == 1 else 1], dtype=int)

        preds = []
        for _, row in X.iterrows():
            out = self._predict_row(row)
            preds.append(0 if out["very_low_risk"] == 1 else 1)
        return np.array(preds, dtype=int)

    def predict_very_low_risk(self, X: Union[pd.DataFrame, Dict[str, Any]]) -> np.ndarray:
        """1 => very low risk, 0 => not very low risk."""
        if isinstance(X, dict):
            out = self._predict_row(pd.Series(X))
            return np.array([out["very_low_risk"]], dtype=int)
        return np.array([self._predict_row(r)["very_low_risk"] for _, r in X.iterrows()], dtype=int)

    def explain(self, x: Union[pd.Series, Dict[str, Any]]) -> Dict[str, Any]:
        """Return rule used and reasons (useful for reporting/debugging)."""
        row = x if isinstance(x, pd.Series) else pd.Series(x)
        return self._predict_row(row)


# ---------------------------
# Logistic Regression
# ---------------------------

def build_best_lr_age2minus():
    """Best logistic regression model for Age < 2."""

    threshold_for_very_low_risk = 0.02  # set threshold to achieve very high sensitivity (e.g., 0.97+)
    return LogisticRegression(
        l1_ratio=1,
        solver="saga",
        class_weight={0: 1, 1: 50},
        C=0.1,
        max_iter=20000,
        random_state=42
    ), threshold_for_very_low_risk

def build_best_lr_age2plus():
    """Best logistic regression model for Age >= 2."""

    threshold_for_very_low_risk = 0.1  # set threshold to achieve very high sensitivity (e.g., 0.97+)
    return LogisticRegression(
        l1_ratio=1,
        solver="saga",
        class_weight={0:1, 1:50},
        C=0.3,
        max_iter=20000,
        random_state=42
    ), threshold_for_very_low_risk

# ---------------------------
# CatBoost
# ---------------------------
def build_best_catboost_age2minus():
    """Best CatBoost model for Age < 2."""
    if not _HAS_CATBOOST:
        raise ImportError("CatBoost is not installed. Please install it to use this function.")
    
    threshold_for_very_low_risk = 0.2  
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        random_seed=42,
        auto_class_weights="Balanced",
        early_stopping_rounds=100,
        verbose=200,
    ), threshold_for_very_low_risk

def build_best_catboost_age2plus():
    """Best CatBoost model for Age >= 2."""
    if not _HAS_CATBOOST:
        raise ImportError("CatBoost is not installed. Please install it to use this function.")
    threshold_for_very_low_risk = 0.2 
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=2000,
        learning_rate=0.01,
        depth=4,
        l2_leaf_reg=7,
        border_count=254,
        random_seed=42,
        auto_class_weights="Balanced",
        early_stopping_rounds=100,
        verbose=200
    ), threshold_for_very_low_risk  




# --------------------------
# Main
# --------------------------


def main() -> None:
    # load data
    tbi_cleaned_new = pd.read_csv('../data/TBI_cleaned.csv')
    output_dir = "../others"
    metrics_file_name = "model_performance.csv"
    lr_coef_file_name = "lr_coefficients.csv"
    cat_fi_file_name = "catboost_feature_importances.csv"
    target_col = 'PosIntFinal'

    rows = []
    coef_rows = []   # logistic regression coefficients
    fi_rows   = []   # catboost feature importances

    def add_result(model_name, age_group, split_name, y_true, y_prob, threshold=None, extra=None):
        m = metrics(y_true, y_prob, threshold=threshold)  # your existing function
        rec = {"model": model_name, 
               "age_group": age_group, 
               "threshold": threshold, 
                "split": split_name,
               **m}
        if extra:
            rec.update(extra)
        rows.append(rec)

    #-----------------
    # CDR model
    #-----------------
    train_cdr_data_age2minus, val_cdr_data_age2minus, test_cdr_data_age2minus, train_cdr_data_age2plus, val_cdr_data_age2plus, test_cdr_data_age2plus = make_cdr_data(tbi_cleaned_new)

    # define the rule model
    pecarn = PECARNDecisionRule()

    for split_name, d in [("train", train_cdr_data_age2minus), ("val", val_cdr_data_age2minus), ("test", test_cdr_data_age2minus)]:
        y_true = d[target_col].astype(int).to_numpy()

        # If rule.predict gives hard labels:
        y_pred = pecarn.predict(d)  # shape (n,)
        y_prob = _safe_prob_from_pred(y_pred)
        add_result(
            model_name="CDR",
            age_group="Age < 2",
            split_name=split_name,
            y_true=y_true,
            y_prob=y_prob,
            threshold=None,
            extra=None
        )

    for split_name, d in [("train", train_cdr_data_age2plus), ("val", val_cdr_data_age2plus), ("test", test_cdr_data_age2plus)]:
        y_true = d[target_col].astype(int).to_numpy()

        # If rule.predict gives hard labels:
        y_pred = pecarn.predict(d)  # shape (n,)
        y_prob = _safe_prob_from_pred(y_pred)

        add_result(
            model_name="CDR",
            age_group="Age >= 2",
            split_name=split_name,
            y_true=y_true,
            y_prob=y_prob,
            threshold=None,
            extra=None
        )

    #-----------------
    # Logistic Regression (Age < 2)
    #-----------------
    train_lr_age2minus, val_lr_age2minus, test_lr_age2minus, train_lr_age2plus, val_lr_age2plus, test_lr_age2plus = make_lr_data(tbi_cleaned_new)

    lr_age2minus, threshold = build_best_lr_age2minus()
    lr_age2minus.fit(train_lr_age2minus[LR_Features_Age2minus], train_lr_age2minus[target_col])

    # Get coefficients for reporting
    
    feat_names = list(LR_Features_Age2minus)   # or LR_Features_Age2plus
    coefs = lr_age2minus.coef_.ravel()         # shape (n_features,)

    for f, c in zip(feat_names, coefs):
        coef_rows.append({
            "model": "Logistic Regression",
            "age_group": "Age < 2",
            "feature": f,
            "coef": float(c),
            "abs_coef": float(abs(c)),
        })

    # save metrics for each split
    for split_name, d in [("train", train_lr_age2minus), ("val", val_lr_age2minus), ("test", test_lr_age2minus)]:
        y_true = d[target_col].astype(int).to_numpy()
        y_prob = lr_age2minus.predict_proba(d[LR_Features_Age2minus])[:, 1]  # probability of class 1
        add_result(
            model_name="Logistic Regression",
            age_group="Age < 2",
            split_name=split_name,
            y_true=y_true,
            y_prob=y_prob,
            threshold=threshold,
            extra=None
        )

    # -----------------
    # Logistic Regression (Age >= 2)
    # -----------------
    lr_age2plus, threshold = build_best_lr_age2plus()
    lr_age2plus.fit(train_lr_age2plus[LR_Features_Age2plus], train_lr_age2plus[target_col])

    # --- save LR coefficients ---
    feat_names = list(LR_Features_Age2plus)   # or LR_Features_Age2plus
    coefs = lr_age2plus.coef_.ravel()         # shape (n_features,)

    for f, c in zip(feat_names, coefs):
        coef_rows.append({
            "model": "Logistic Regression",
            "age_group": "Age >= 2",
            "feature": f,
            "coef": float(c),
            "abs_coef": float(abs(c)),
        })
        
    for split_name, d in [("train", train_lr_age2plus), ("val", val_lr_age2plus), ("test", test_lr_age2plus)]:
        y_true = d[target_col].astype(int).to_numpy()
        y_prob = lr_age2plus.predict_proba(d[LR_Features_Age2plus])[:, 1]  # probability of class 1
        add_result(
            model_name="Logistic Regression",
            age_group="Age >= 2",
            split_name=split_name,
            y_true=y_true,
            y_prob=y_prob,
            threshold=threshold,
            extra=None
        )
    # -----------------
    # CatBoost     (Age < 2)
    # -----------------
    train_cat_age2minus, val_cat_age2minus, test_cat_age2minus, train_cat_age2plus, val_cat_age2plus, test_cat_age2plus = make_catboost_data(tbi_cleaned_new)

    if _HAS_CATBOOST:
        catboost_age2minus, threshold = build_best_catboost_age2minus()
        catboost_features_Age2minus = CatBoost_Num_Features_Age2minus + CatBoost_Cat_Features_Age2minus
        catboost_features_Age2minus.sort()  # sort features for consistent order
        catboost_age2minus.fit(
            train_cat_age2minus[catboost_features_Age2minus], 
            train_cat_age2minus[target_col],
            cat_features=CatBoost_Cat_Features_Age2minus,
            eval_set=(val_cat_age2minus[catboost_features_Age2minus], val_cat_age2minus[target_col]))
        # --- save CatBoost feature importance (interpretability) ---
        feat_names = catboost_features_Age2minus
        importances = catboost_age2minus.get_feature_importance(type="FeatureImportance")

        for f, imp in zip(feat_names, importances):
            fi_rows.append({
                "model": "CatBoost",
                "age_group": "Age < 2",
                "feature": f,
                "importance": float(imp),
            })

        for split_name, d in [("train", train_cat_age2minus), ("val", val_cat_age2minus), ("test", test_cat_age2minus)]:
            y_true = d[target_col].astype(int).to_numpy()
            y_prob = catboost_age2minus.predict_proba(d[catboost_features_Age2minus])[:, 1]
            add_result(
                model_name="CatBoost",
                age_group="Age < 2",
                split_name=split_name,
                y_true=y_true,
                y_prob=y_prob,
                threshold=threshold,
                extra=None
            )
    # -----------------
    # CatBoost     (Age >= 2)
    # -----------------
    if _HAS_CATBOOST:
        catboost_age2plus, threshold = build_best_catboost_age2plus()
        catboost_features_Age2plus = CatBoost_Num_Features_Age2plus + CatBoost_Cat_Features_Age2plus
        catboost_features_Age2plus.sort()  # sort features for consistent order
        
        catboost_age2plus.fit(
            train_cat_age2plus[catboost_features_Age2plus], 
            train_cat_age2plus[target_col],
            cat_features=CatBoost_Cat_Features_Age2plus,
            eval_set=(val_cat_age2plus[catboost_features_Age2plus], val_cat_age2plus[target_col])
        )

        # --- save CatBoost feature importance (interpretability) ---
        feat_names = catboost_features_Age2plus
        importances = catboost_age2plus.get_feature_importance(type="FeatureImportance")

        for f, imp in zip(feat_names, importances):
            fi_rows.append({
                "model": "CatBoost",
                "age_group": "Age >= 2",
                "feature": f,
                "importance": float(imp),
            })
        
        for split_name, d in [("train", train_cat_age2plus), ("val", val_cat_age2plus), ("test", test_cat_age2plus)]:
            y_true = d[target_col].astype(int).to_numpy()
            y_prob = catboost_age2plus.predict_proba(d[catboost_features_Age2plus])[:, 1]
            add_result(
                model_name="CatBoost",
                age_group="Age >= 2",
                split_name=split_name,
                y_true=y_true,
                y_prob=y_prob,
                threshold=threshold,
                extra=None
            )
    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(output_dir, metrics_file_name), index=False)

    # ---- save interpretability artifacts ----
    if len(coef_rows) > 0:
        coef_df = pd.DataFrame(coef_rows).sort_values(
            ["model", "age_group", "abs_coef"], ascending=[True, True, False]
        )
        coef_df.to_csv(os.path.join(output_dir, lr_coef_file_name), index=False)

    if len(fi_rows) > 0:
        fi_df = pd.DataFrame(fi_rows).sort_values(
            ["model", "age_group", "importance"], ascending=[True, True, False]
        )
        fi_df.to_csv(os.path.join(output_dir, cat_fi_file_name), index=False)

        

if __name__ == "__main__":
    main()
