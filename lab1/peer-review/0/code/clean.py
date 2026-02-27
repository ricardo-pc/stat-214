from typing import List
import pandas as pd
from features import (
    numeric_cols,
    binary_cols,
    multi_category_cols
)


def flag_posintfinal_inconsistency(
    df: pd.DataFrame,
    outcome_col: str = "PosIntFinal",
    symptom_cols = [
        "DeathTBI",
        "HospHeadPosCT",
        "Intub24Head",
        "Neurosurgery"
    ],
    flag_col: str = "PosIntFinal_invalid"
) -> pd.DataFrame:
    """
    Flag logically inconsistent cases where:
    PosIntFinal == 1 but all symptom indicators are 0.

    Does NOT change PosIntFinal — only adds a flag column.

    Returns a modified copy.
    """
    out = df.copy()

    # Define "no symptoms present"
    no_symptoms = (out[symptom_cols] == 0).all(axis=1)

    # Invalid if outcome=1 but no symptoms
    out[flag_col] = (
        (out[outcome_col] == 1) & no_symptoms
    ).astype('int')  # 'c' for category (or use int if you prefer)

    return out




def clean_data(
        df: pd.DataFrame,
        numeric_cols: List[str],
        binary_cols: List[str],
        multi_category_cols: List[str],
        target_col: str = 'PosIntFinal',
        drop_target_na: bool = True
):
    '''
    Clean the TBI dataset by:
    1. Coercing data types for numeric, binary, and multi-category columns.
    2. Optionally dropping rows with missing target variable.
    3. Flagging inconsistencies in the target variable based on symptom indicators.

    Parameters:
    - df: Input DataFrame to clean.
    - numeric_cols: List of column names to coerce to numeric.
    - binary_cols: List of column names to coerce to binary (0/1).
    - multi_category_cols: List of column names to coerce to categorical.
    - target_col: Name of the target variable column (default 'PosIntFinal').
    - drop_target_na: Whether to drop rows with missing target variable (default True).
    Returns:
    - A cleaned DataFrame with coerced data types, optionally dropped rows, and inconsistency flags.
    '''

    # Step 1: coerce data types
    out = df.copy()
    out[numeric_cols] = out[numeric_cols].apply(pd.to_numeric, errors='coerce')  
    out[binary_cols] = out[binary_cols].astype('int', errors='ignore')  # if already numeric, keep as is; if not, try to convert to int (0/1), coercing errors to NaN
    out[multi_category_cols] = out[multi_category_cols].astype('category')


    # Step 2: Drop rows with missing target
    if drop_target_na:
        out = out.dropna(subset=[target_col])


    # Step 3: Flag inconsistencies in PosIntFinal
    out = flag_posintfinal_inconsistency(out)
    return out


if __name__ == "__main__":
    tbi_origin = pd.read_csv('../data/TBI PUD 10-08-2013.csv')
    tbi_cleaned_new = clean_data(
        df=tbi_origin,
        numeric_cols=numeric_cols,
        binary_cols=binary_cols,
        multi_category_cols=multi_category_cols,
        target_col='PosIntFinal',
        drop_target_na=True
    )
    # Save the cleaned dataset to a new CSV file
    tbi_cleaned_new.to_csv('../data/TBI_cleaned.csv', index=False)


