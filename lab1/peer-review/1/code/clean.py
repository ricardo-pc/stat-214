import pandas as pd


def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw PECARN TBI dataset.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw dataframe immediately after loading with pandas.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for analysis/modeling.
    """

    #copy to avoid changing the original
    df = raw_df.copy()

    #remove low GCS patients (GCSGroup == 1)
    df = df[df["GCSGroup"] != 1]

    #remove rows with missing outcome
    df = df[df["PosIntFinal"].notna()]

    #drop administrative columns
    df = df.drop(columns=["EmplType", "Certification"], errors="ignore")

    #injury mechanism fixes
    df.loc[
        (df["InjuryMech"] == 2) &
        (df["High_impact_InjSev"].isna()),
        "High_impact_InjSev"
    ] = 3

    df.loc[
        df["InjuryMech"].isna() &
        df["High_impact_InjSev"].isna(),
        ["InjuryMech", "High_impact_InjSev"]
    ] = [99, 4]

    #LOC and Seizure
    df.fillna({
        "LOCSeparate": 3,
        "LocLen": 5,
        "Seiz": 2,
        "SeizOccur": 4,
        "SeizLen": 5
    }, inplace=True)

    #amnesia, headache and diziness
    symptom_cols = ["HA_verb", "Amnesia_verb", "Dizzy"]

    df.loc[df["AgeinYears"] < 2, symptom_cols] = 91

    not_assessable_mask = (
        (df["Paralyzed"] == 1) |
        (df["Sedated"] == 1) |
        (df["Intubated"] == 1) |
        (df["ActNorm"] == 0) |
        (df["AMS"] == 1)
    )

    for col in symptom_cols:
        df.loc[df[col].isna() & not_assessable_mask, col] = 91
        df.loc[df[col].isna(), col] = 92

    #headache related variables
    df.loc[(df["HA_verb"] == 91) & (df["HASeverity"].isna()), "HASeverity"] = 92
    df.loc[(df["HA_verb"] == 91) & (df["HAStart"].isna()), "HAStart"] = 92
    df.loc[df["HASeverity"].isna(), "HASeverity"] = 4
    df.loc[df["HAStart"].isna(), "HAStart"] = 5

    #Outcome related variables
    df.loc[(df["PosIntFinal"] == 0) & (df["DeathTBI"].isna()), "DeathTBI"] = 0
    df.loc[(df["PosIntFinal"] == 0) & (df["HospHead"].isna()), "HospHead"] = 0
    df.loc[(df["PosIntFinal"] == 1) & (df["Intub24Head"].isna()), "Intub24Head"] = 1
    df.loc[(df["PosIntFinal"] == 0) & (df["Intub24Head"].isna()), "Intub24Head"] = 0
    df.loc[(df["PosIntFinal"] == 0) & (df["Neurosurgery"].isna()), "Neurosurgery"] = 0

    #AMS fix
    df.loc[(df["AMS"].isna()) & (df["GCSTotal"] == 15), "AMS"] = 0

    #Act Normal as per parent variable
    df.loc[df["ActNorm"].isna(), "ActNorm"] = 2

    #vomiting symptom and related variables
    df.loc[df["Vomit"].isna(), "Vomit"] = 2

    df.loc[df["Vomit"] == 0, ["VomitNbr", "VomitStart", "VomitLast"]] = 92

    df.loc[(df["Vomit"] == 1) & (df["VomitNbr"].isna()), "VomitNbr"] = 4
    df.loc[(df["Vomit"] == 1) & (df["VomitStart"].isna()), "VomitStart"] = 5
    df.loc[(df["Vomit"] == 1) & (df["VomitLast"].isna()), "VomitLast"] = 4

    df.loc[df["Vomit"] == 2, ["VomitNbr", "VomitStart", "VomitLast"]] = 92

    #skull fracture variables
    df.loc[df["SFxPalp"].isna(), "SFxPalp"] = 2
    df.loc[df["SFxPalp"] != 1, "SFxPalpDepress"] = 92
    df.loc[(df["SFxPalp"] == 1) & (df["SFxPalpDepress"].isna()), "SFxPalpDepress"] = 0

    df.loc[(df["AgeinYears"] > 2) & (df["FontBulg"].isna()), "FontBulg"] = 0
    df.loc[(df["AgeinYears"] <= 2) & (df["FontBulg"].isna()), "FontBulg"] = 2

    df.loc[df["SFxBas"].isna(), "SFxBas"] = 2

    #convert all categorical values in predictor columns from float to int
    categorical_cols = [
        "High_impact_InjSev", "InjuryMech", "LOCSeparate", "LocLen",
        "Seiz", "SeizOccur", "SeizLen", "HA_verb", "Amnesia_verb",
        "Dizzy", "HASeverity", "HAStart", "DeathTBI", "HospHead",
        "Intub24Head", "Neurosurgery", "AMS", "ActNorm",
        "Vomit", "VomitNbr", "VomitStart", "VomitLast",
        "SFxPalp", "SFxPalpDepress", "FontBulg", "SFxBas"
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("Int64").astype(int)

    return df
