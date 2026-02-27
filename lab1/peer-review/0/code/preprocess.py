from typing import Dict, List, Literal, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from features import numeric_cols, categorical_cols, binary_cols, multi_category_cols, parent_child_groups, identifier_cols

def impute_gcs(
    df: pd.DataFrame,
    motor_col: str = "GCSMotor",
    verbal_col: str = "GCSVerbal",
    eye_col: str = "GCSEye",
    total_col: str = "GCSTotal"
) -> pd.DataFrame:
    """
    Impute missing GCS components using GCSTotal = Motor + Verbal + Eye.

    Rules:
    - If total==15 and all components missing -> set (6,5,4).
    - If total known and exactly one component missing -> infer it if valid.
    - Invalid inferred values are left as NaN.

    Returns a copy of df with imputed columns.
    """
    out = df.copy()

    # Ensure numeric (preserve NaN)
    for c in [motor_col, verbal_col, eye_col, total_col]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Valid ranges
    bounds = {
        motor_col: (1, 6),
        verbal_col: (1, 5),
        eye_col: (1, 4),
        total_col: (3, 15),
    }

    def in_range(val, col):
        lo, hi = bounds[col]
        return pd.notna(val) and (lo <= val <= hi)

    # Row-wise imputation
    def fill_row(row):
        m, v, e, t = row[motor_col], row[verbal_col], row[eye_col], row[total_col]

        m_na, v_na, e_na, t_na = pd.isna(m), pd.isna(v), pd.isna(e), pd.isna(t)
        n_known_comp = sum([not m_na, not v_na, not e_na])

        # Case A: all components missing but total == 15 -> assume max components
        if (n_known_comp == 0) and (not t_na) and (t == 15):
            row[motor_col], row[verbal_col], row[eye_col] = 6, 5, 4
            return row

        # Case B: total known, exactly one component missing -> infer
        if (not t_na) and (n_known_comp == 2):
            if m_na:
                inferred = t - v - e
                row[motor_col] = inferred if in_range(inferred, motor_col) else np.nan
            elif v_na:
                inferred = t - m - e
                row[verbal_col] = inferred if in_range(inferred, verbal_col) else np.nan
            elif e_na:
                inferred = t - m - v
                row[eye_col] = inferred if in_range(inferred, eye_col) else np.nan
            return row
        return row

    out = out.apply(fill_row, axis=1)

    return out

FillingStrategy = Literal["mode", "median", "mean"]  

def fit_imputer(train: pd.DataFrame, 
                cols: List[str],
                filling_strategy: FillingStrategy = "mode") -> Dict[str, object]:
    """Learn filling values for specified columns using the specified strategy on TRAIN data."""
    filling_values = {}
    for c in cols:
        if filling_strategy == "mode":
            s = train[c]
            m = s.mode(dropna=True)
            filling_values[c] = m.iloc[0] if len(m) else np.nan
        elif filling_strategy == "median":
            s = train[c]
            # round median to nearest integer since we only have integer values in our dataset
            filling_values[c] = int(round(s.median()))
        elif filling_strategy == "mean":
            s = train[c]
            # round mean to nearest integer since we only have integer values in our dataset
            filling_values[c] = int(round(s.mean()))
        else:
            raise ValueError("filling_strategy must be mode/median/mean")
    return filling_values

def apply_imputer(df: pd.DataFrame, 
                  filling_values: Dict[str, object]) -> pd.DataFrame:
    """Fill NA using precomputed filling values."""
    out = df.copy()
    for c, v in filling_values.items():
        if c in out.columns and pd.notna(v):
            out[c] = out[c].fillna(v)

    return out



ParentMissingFill = Literal["leave", "fill0"]
ChildMissingStrategy = Literal["leave", "missing_category"]


def apply_parent_child_missingness(
    df: pd.DataFrame,
    groups: Dict[str, List[str]],
    missing_code: float = 92,
    parent_yes_value: float = 1,
    add_parent_missing_indicator: bool = True,
    parent_missing_strategy: ParentMissingFill = "fill0",  # A: "leave", B: "fill0"
    child_missing_when_parent_yes: ChildMissingStrategy = "missing_category",
    missing_category_label: str = "missing"
) -> pd.DataFrame:
    
    """Apply parent-child missingness handling for specified groups.
    For each parent-child group:
    - If parent is missing, set all children to missing_code (default 92) to create a consistent missingness pattern that captures the fact that these children are likely missing due to the parent being missing. Optionally add a binary indicator for parent missingness and optionally fill parent with 0.
    - For rows where parent == parent_yes_value (e.g. 1), handle child missingness according to child_missing_when_parent_yes strategy: either leave as is or set to a missing category (for categorical children).

    """
    out = df.copy()

    for parent, children in groups.items():
        # Ensure columns exist
        missing_cols = [c for c in [parent] + children if c not in out.columns]
        if missing_cols:
            raise KeyError(f"Columns not found in df: {missing_cols}")


        # parent missing -> children = missing_code (consistent with definition)
        parent_missing = out[parent].isna()
        out.loc[parent_missing, children] = missing_code

        # add missing indicator for parent missingness
        if add_parent_missing_indicator and parent_missing.any():
            out[f"{parent}_missing"] = parent_missing.astype(int)
        # parent missing fill strategy
        if parent_missing_strategy == "fill0":
            out.loc[parent_missing, parent] = 0
        elif parent_missing_strategy == "leave":
            pass  # do nothing, leave as missing

        # child missing strategy for rows where parent == yes
        parent_yes_mask = out[parent] == parent_yes_value
        if child_missing_when_parent_yes == "leave":
            continue

        if child_missing_when_parent_yes == "missing_category":
            for child in children:
                miss_mask = parent_yes_mask & out[child].isna()
                if not miss_mask.any():
                    continue
                # coerce to categorical if not already, since we need to add a category for missing
                if not isinstance(out[child].dtype, pd.CategoricalDtype):
                     out[child] = out[child].astype("category")

                # Add category then fill
                if missing_category_label not in out[child].cat.categories:
                    out[child] = out[child].cat.add_categories([missing_category_label])

                out.loc[miss_mask, child] = missing_category_label

        else:
            raise ValueError(f"Unknown child_missing_when_parent_yes={child_missing_when_parent_yes}")


    return out


BinaryMissing = Literal["leave", "fill0"]
MultiMissingStrategy = Literal["leave", "missing_category"]

def handle_missing_for_categories(
    df: pd.DataFrame,
    *,
    binary_cat_cols: List[str],
    multi_cat_cols: List[str],
    binary_missing_strategy: BinaryMissing = "fill0",   # A: "leave", B: "fill0"
    add_binary_missing_indicator: bool = True,
    missing_category_label: str = "missing",
    multi_missing_strategy: MultiMissingStrategy = "missing_category",
    force_to_category: bool = True
) -> pd.DataFrame:
    """Handle missing values for categorical variables.
    For binary categorical columns:
    - Optionally add a missing indicator column (colname_missing) for each binary cat column that indicates whether the original value was missing.
    - Handle missing values according to binary_missing_strategy: either leave as is or fill with 0 (assuming 0 is the "negative" category, which is a common and usually safe assumption for binary indicators in medical datasets, but should be verified for each specific column).
    For multi-category columns:
    - Handle missing values according to multi_missing_strategy: either leave as is or add a "missing" category and set missing values to that category.

    """

    out = df.copy()

    # Validate columns exist 
    all_cols = list(set(binary_cat_cols + multi_cat_cols))
    missing_cols = [c for c in all_cols if c not in out.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in df: {missing_cols}")

    # --- Force dtype to category if requested ---
    if force_to_category:
        # Only convert multi-category columns to category, since binary cats may be 0/1 numeric and that's fine
        for c in multi_cat_cols:
            if not isinstance(out[c].dtype, pd.CategoricalDtype):
                out[c] = out[c].astype("category")

    #  Binary cats: add indicator + fill0/leave 
    for c in binary_cat_cols:
        miss_mask = out[c].isna()
        
        if add_binary_missing_indicator and miss_mask.any():
            out[f"{c}_missing"] = miss_mask.astype(int)

        if binary_missing_strategy == "fill0":
            out.loc[miss_mask, c] = 0
        elif binary_missing_strategy == "leave":
            pass
        else:
            raise ValueError("binary_missing_strategy must be 'leave' or 'fill0'")

    # Multi-category: add 'missing' level + fill 
    if multi_missing_strategy == "missing_category":
        for c in multi_cat_cols:
            if out[c].isna().any():
                if missing_category_label not in out[c].cat.categories:
                    out[c] = out[c].cat.add_categories([missing_category_label])
                out[c] = out[c].fillna(missing_category_label)
    elif multi_missing_strategy == "leave":
        pass
    else:  
        raise ValueError(f"multi_missing_strategy must be 'missing_category' or 'leave', got {multi_missing_strategy}")

    return out


GCSFillingStrategy = Literal["mode", "median", "mean",'leave'] 

def preprocess_data(
    df: pd.DataFrame,
    numeric_cols = numeric_cols,
    categorical_cols = categorical_cols,
    multi_category_cols: List[str] = multi_category_cols,
    binary_cols: List[str] = binary_cols,
    parent_child_groups: Dict[str, List[str]] = parent_child_groups,
    target_col: str = "PosIntFinal",
    test_size: float = 0.2,
    val_size: float = 0.1,              # fraction of FULL data (not of train_temp)
    random_state: int = 42,
    stratify_col: List[str] = ["PosIntFinal", "AgeTwoPlus"],

    if_exclude_gcs_under_13: bool = True,  # if True, drop rows with gcs 3-13/GCSGroup =1 

    if_fill_missing_gcs: bool = True,      # if True, infer missing GCS components using a consistent strategy learned from TRAIN only (to avoid leakage)
    gcs_fill_strategy: GCSFillingStrategy = "mode",
    if_one_hot_encode: bool = False,
    gcs_cols: Optional[List[str]] = ["GCSEye","GCSVerbal","GCSMotor"],
    drop_first_cat_in_ohe: bool = True,

    if_drop_na_rows: bool = False,          # if True, drop rows with any missing values in numeric_cols or categorical_col

    if_handle_parent_child_missing: bool = True,  # if True, apply the parent-child missingness handling for specified groups
    if_handle_missing_for_categories: bool = True,  # if True, apply the standalone categorical missingness handling for specified columns
    missing_category_label: str = "missing",
    if_add_flag_missing_cols: bool = True,
    parent_missing_strategy: ParentMissingFill = "leave",
    child_missing_when_parent_yes: ChildMissingStrategy = "missing_category",
    binary_missing_strategy: BinaryMissing = "leave",
    multi_missing_strategy: MultiMissingStrategy = "missing_category",
) -> Dict[str, object]:
    """Preprocess the data and return train/val/test splits.
        Steps:

    1. Optional: Exclude rows with GCS under 13 (GCSGroup=1) if specified.
    2. Optional: Handle missing values in categorical variables using the handle_missing_for_categories function, which can add missing indicators for binary cats and/or add a "missing" category for multi-cats.
    3. Optional: Handle parent-child missingness for specified groups using the apply_parent_child_missingness function, which creates a consistent missingness pattern that captures the fact that children are likely missing due to the parent being missing, and optionally adds indicators for parent missingness and handles child missingness when parent == yes.
    4. Train/val/test split (stratify if specified).
    5. Optional: Exclude rows with GCS under 13 (GCSGroup=1) if specified (in case you want to do this after the split instead of before).
    6. Optional: Infer GCS components and fill missing GCS values using a consistent strategy learned from TRAIN only (to avoid leakage).
    7. Optional: One-hot encode multi-category variables.
    8. Optional: Drop rows with any missing values in numeric_cols or categorical_col.
    """
    
    out = df.copy()

    # optional: exclude rows with GCS under 13 (GCSGroup=1) if specified
    if if_exclude_gcs_under_13:
        out = out[out['GCSGroup'] != 1] 

    # optional: handle missing values in categorical variables
    if if_handle_missing_for_categories:
        out = handle_missing_for_categories(
            out,
            binary_cat_cols= binary_cols,
            multi_cat_cols= multi_category_cols,
            binary_missing_strategy=binary_missing_strategy,
            add_binary_missing_indicator=if_add_flag_missing_cols,
            missing_category_label=missing_category_label,
            multi_missing_strategy=multi_missing_strategy,
            force_to_category=True
        )


    # optional: handle parent-child missingness for specified groups
    if if_handle_parent_child_missing:
        out = apply_parent_child_missingness(out, parent_child_groups, missing_code=92, parent_yes_value=1, 
                                                  add_parent_missing_indicator=if_add_flag_missing_cols, 
                                                  parent_missing_strategy=parent_missing_strategy, 
                                                  child_missing_when_parent_yes=child_missing_when_parent_yes, 
                                                  missing_category_label=missing_category_label)

    # for missingness flag, we do it before train-test split to capture the missingness pattern in the whole dataset to make sure we have the same features across train/val/test. 
    
    
    #  train/val/test split (stratify if po)
    # do this BEFORE any imputation/encoding to avoid leakage

    train_temp, test_df = train_test_split(
        out,
        test_size=test_size,
        random_state=random_state,
        stratify=out[stratify_col] if stratify_col is not None else None,
    )

    # val_size is fraction of full dataset; convert to fraction of train_temp
    val_frac_of_train_temp = val_size / (1.0 - test_size)

    train_df, val_df = train_test_split(
        train_temp,
        test_size=val_frac_of_train_temp,
        random_state=random_state,
        stratify=train_temp[stratify_col] if stratify_col is not None else None,
    )

    # optional: exclude rows with GCS under 13 (GCSGroup=1) if specified
    if if_exclude_gcs_under_13:
        train_df = train_df[train_df['GCSGroup'] != 1]
        val_df   = val_df[val_df['GCSGroup'] != 1]
        test_df  = test_df[test_df['GCSGroup'] != 1]


    # optional: infer GCS components and fill missing GCS values using a consistent strategy learned from TRAIN only (to avoid leakage)
    if if_fill_missing_gcs and not gcs_cols:
        raise ValueError("if_fill_missing_gcs=True but gcs_cols is None/empty")
    if if_fill_missing_gcs:
        # first impute GCS components using the impute_gcs function we defined earlier, which uses the internal consistency rules of GCS to fill in some missing values
        train_df = impute_gcs(train_df)
        val_df   = impute_gcs(val_df)
        test_df  = impute_gcs(test_df)

        if gcs_fill_strategy != 'leave':
            # then, for any remaining missing GCS component values, fill them using a simple strategy (mode/median/mean) learned from TRAIN only
            imputer_values = fit_imputer(train_df, gcs_cols, filling_strategy=gcs_fill_strategy)
            train_df = apply_imputer(train_df, imputer_values)
            val_df   = apply_imputer(val_df, imputer_values)
            test_df  = apply_imputer(test_df, imputer_values)
        else:
            imputer_values = None

    # optional: one-hot encode multi-category variables
    
    if if_one_hot_encode:
        # pick which columns to one-hot (usually multi-cat)
        ohe_cols = [c for c in (multi_category_cols) if c in train_df.columns]

        train_ohe = pd.get_dummies(train_df, columns=ohe_cols, drop_first=drop_first_cat_in_ohe)
        val_ohe   = pd.get_dummies(val_df, columns=ohe_cols, drop_first=drop_first_cat_in_ohe)
        test_ohe  = pd.get_dummies(test_df, columns=ohe_cols, drop_first=drop_first_cat_in_ohe)

        # align val/test to train columns
        val_ohe  = val_ohe.reindex(columns=train_ohe.columns, fill_value=0)
        test_ohe = test_ohe.reindex(columns=train_ohe.columns, fill_value=0)

        train_df, val_df, test_df = train_ohe, val_ohe, test_ohe
    
    # optional: drop rows with any missing values in numeric_cols or categorical_col
    if if_drop_na_rows:
        train_df = train_df.dropna(subset= train_df.columns.difference(identifier_cols))  # drop rows with NA in any feature column (but not target)
        val_df   = val_df.dropna(subset= train_df.columns.difference(identifier_cols))
        test_df  = test_df.dropna(subset= train_df.columns.difference(identifier_cols))

    return train_df, val_df, test_df
