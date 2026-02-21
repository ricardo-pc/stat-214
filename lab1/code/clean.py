"""
clean.py - Data cleaning and preprocessing for PECARN TBI dataset.

This module provides the clean_data function that takes the raw DataFrame
and returns the cleaned, preprocessed dataset ready for analysis.
"""

import re

import pandas as pd


# Column rename mapping: original PECARN names -> descriptive snake_case names
RENAME_MAP = {
    # Identifiers & provider info
    "PatNum": "patient_id",
    "EmplType": "physician_position",
    "Certification": "physician_certification",
    # Injury mechanism
    "InjuryMech": "injury_mechanism",
    "High_impact_InjSev": "injury_severity",
    # Symptoms: amnesia, LOC, seizure
    "Amnesia_verb": "amnesia",
    "LOCSeparate": "loc",
    "LocLen": "loc_duration",
    "Seiz": "seizure",
    "SeizOccur": "seizure_timing",
    "SeizLen": "seizure_duration",
    # Symptoms: general
    "ActNorm": "acting_normal",
    "HA_verb": "headache",
    "HASeverity": "headache_severity",
    "HAStart": "headache_onset",
    "Vomit": "vomiting",
    "VomitNbr": "vomiting_episodes",
    "VomitStart": "vomiting_onset",
    "VomitLast": "vomiting_last",
    "Dizzy": "dizziness",
    # Pre-evaluation interventions
    "Intubated": "intubated",
    "Paralyzed": "paralyzed",
    "Sedated": "sedated",
    # Glasgow Coma Scale
    "GCSEye": "gcs_eye",
    "GCSVerbal": "gcs_verbal",
    "GCSMotor": "gcs_motor",
    "GCSTotal": "gcs_total",
    "GCSGroup": "gcs_group",
    # Altered mental status
    "AMS": "altered_mental_status",
    "AMSAgitated": "ams_agitated",
    "AMSSleep": "ams_sleepy",
    "AMSSlow": "ams_slow_response",
    "AMSRepeat": "ams_repetitive_questions",
    "AMSOth": "ams_other",
    # Skull fracture signs
    "SFxPalp": "skull_fx_palpable",
    "SFxPalpDepress": "skull_fx_depressed",
    "FontBulg": "fontanelle_bulging",
    "SFxBas": "basilar_skull_fx",
    "SFxBasHem": "basilar_fx_hemotympanum",
    "SFxBasOto": "basilar_fx_csf_otorrhea",
    "SFxBasPer": "basilar_fx_raccoon_eyes",
    "SFxBasRet": "basilar_fx_battle_sign",
    "SFxBasRhi": "basilar_fx_csf_rhinorrhea",
    # Scalp hematoma
    "Hema": "scalp_hematoma",
    "HemaLoc": "hematoma_location",
    "HemaSize": "hematoma_size",
    # Trauma above clavicles
    "Clav": "trauma_above_clavicles",
    "ClavFace": "trauma_face",
    "ClavNeck": "trauma_neck",
    "ClavFro": "trauma_scalp_frontal",
    "ClavOcc": "trauma_scalp_occipital",
    "ClavPar": "trauma_scalp_parietal",
    "ClavTem": "trauma_scalp_temporal",
    # Neurological deficit
    "NeuroD": "neuro_deficit",
    "NeuroDMotor": "neuro_deficit_motor",
    "NeuroDSensory": "neuro_deficit_sensory",
    "NeuroDCranial": "neuro_deficit_cranial",
    "NeuroDReflex": "neuro_deficit_reflexes",
    "NeuroDOth": "neuro_deficit_other",
    # Other substantial injuries
    "OSI": "other_injuries",
    "OSIExtremity": "other_injury_extremity",
    "OSICut": "other_injury_laceration",
    "OSICspine": "other_injury_cspine",
    "OSIFlank": "other_injury_chest_flank",
    "OSIAbdomen": "other_injury_abdomen",
    "OSIPelvis": "other_injury_pelvis",
    "OSIOth": "other_injury_other",
    # Drugs
    "Drugs": "drug_intoxication",
    # CT ordering
    "CTForm1": "ct_ordered",
    "IndAge": "ct_ind_age",
    "IndAmnesia": "ct_ind_amnesia",
    "IndAMS": "ct_ind_mental_status",
    "IndClinSFx": "ct_ind_skull_fracture",
    "IndHA": "ct_ind_headache",
    "IndHema": "ct_ind_hematoma",
    "IndLOC": "ct_ind_loc",
    "IndMech": "ct_ind_mechanism",
    "IndNeuroD": "ct_ind_neuro_deficit",
    "IndRqstMD": "ct_ind_md_request",
    "IndRqstParent": "ct_ind_parent_request",
    "IndRqstTrauma": "ct_ind_trauma_team",
    "IndSeiz": "ct_ind_seizure",
    "IndVomit": "ct_ind_vomiting",
    "IndXraySFx": "ct_ind_xray_fracture",
    "IndOth": "ct_ind_other",
    # CT sedation
    "CTSed": "ct_sedation",
    "CTSedAgitate": "ct_sed_agitation",
    "CTSedAge": "ct_sed_age",
    "CTSedRqst": "ct_sed_tech_request",
    "CTSedOth": "ct_sed_other",
    # Demographics
    "AgeInMonth": "age_months",
    "AgeinYears": "age_years",
    "AgeTwoPlus": "age_group",
    "Gender": "gender",
    "Ethnicity": "ethnicity",
    "Race": "race",
    # ED management & disposition
    "Observed": "observed_in_ed",
    "EDDisposition": "ed_disposition",
    "CTDone": "ct_performed",
    "EDCT": "ct_performed_in_ed",
    "PosCT": "tbi_on_ct",
    # CT findings
    "Finding1": "finding_cerebellar_hemorrhage",
    "Finding2": "finding_cerebral_contusion",
    "Finding3": "finding_cerebral_edema",
    "Finding4": "finding_cerebral_hemorrhage",
    "Finding5": "finding_skull_diastasis",
    "Finding6": "finding_epidural_hematoma",
    "Finding7": "finding_extraaxial_hematoma",
    "Finding8": "finding_intraventricular_hemorrhage",
    "Finding9": "finding_midline_shift",
    "Finding10": "finding_pneumocephalus",
    "Finding11": "finding_skull_fracture",
    "Finding12": "finding_subarachnoid_hemorrhage",
    "Finding13": "finding_subdural_hematoma",
    "Finding14": "finding_traumatic_infarction",
    # Outcome
    "DeathTBI": "death_from_tbi",
    "HospHead": "hospitalized_head_injury",
    "HospHeadPosCT": "hospitalized_positive_ct",
    "Intub24Head": "intubated_24h_head",
    "Neurosurgery": "neurosurgery",
    "PosIntFinal": "clinically_important_tbi",
}


def _parse_value_mappings(filepath):
    """Parse the PECARN documentation Excel file to extract value label mappings.

    Reads the documentation spreadsheet and extracts coded value -> label
    mappings for each variable (e.g., 1 = "Yes", 0 = "No", 92 = "Not Applicable").

    Parameters
    ----------
    filepath : str
        Path to the TBI PUD Documentation Excel file.

    Returns
    -------
    dict
        Nested dictionary: {original_variable_name: {int_code: str_label}}.
    """
    df_mapping = pd.read_excel(filepath, skiprows=10, usecols=[0, 1, 2])
    all_variable_maps = {}

    for _, row in df_mapping.iterrows():
        var_name = row.iloc[0]
        messy_content = row.iloc[2]

        if pd.isna(messy_content):
            continue

        current_map = {}
        lines = str(messy_content).split("\n")

        for line in lines:
            line = line.strip()
            match = re.match(r"^(\d+)\s+(.*)$", line)
            if match:
                code = int(match.group(1))
                label = match.group(2).strip()
                current_map[code] = label

        if current_map:
            all_variable_maps[var_name] = current_map

    return all_variable_maps


def clean_data(df, mapping_file=None):
    """Clean and preprocess the raw PECARN TBI dataset.

    Applies the following steps:
    1. Copy the raw DataFrame to avoid modifying the original.
    2. Rename all 125 columns to descriptive snake_case names.
    3. Convert all columns to nullable Int64 dtype (supports NaN without
       float upcasting).
    4. Apply inclusion/exclusion criteria from Kuppermann et al. (2009):
       - Exclude trivial injury patients (low-mechanism + no clinical signs).
       - Include only GCS 14-15 (exclude moderate-to-severe GCS 3-13).
       - Exclude patients with missing outcome.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame loaded directly from TBI PUD 10-08-2013.csv.
    mapping_file : str, optional
        Path to the TBI PUD Documentation Excel file. If provided, a second
        DataFrame with human-readable value labels is also returned.

    Returns
    -------
    pd.DataFrame
        Cleaned and preprocessed DataFrame (df_prep) with integer-coded values.
    pd.DataFrame or None
        If mapping_file is provided, returns df_prep_mapped with human-readable
        labels. Otherwise returns None.
    """
    # Step 1: Copy the raw data
    df_clean = df.copy()

    # Step 2: Rename columns to descriptive snake_case
    df_clean = df_clean.rename(columns=RENAME_MAP)

    # Step 3: Convert all columns to nullable Int64 (supports NaN natively)
    for col in df_clean.columns:
        df_clean[col] = df_clean[col].astype("Int64")

    # Step 4: Apply inclusion/exclusion criteria (Kuppermann et al. 2009)

    # 4a. Exclude trivial injury patients:
    #     Low-energy mechanism (walked/ran into object OR fell from standing)
    #     AND no clinical signs of significance
    trivial_mask = (
        ((df_clean["injury_mechanism"] == 6) | (df_clean["injury_mechanism"] == 7))
        & (df_clean["trauma_above_clavicles"] == 1)
        & (df_clean["trauma_face"] == 0)
        & (df_clean["trauma_neck"] == 0)
        & df_clean[
            [
                "trauma_scalp_frontal",
                "trauma_scalp_occipital",
                "trauma_scalp_parietal",
                "trauma_scalp_temporal",
            ]
        ]
        .eq(1)
        .any(axis=1)
        & (df_clean["skull_fx_palpable"] == 0)
        & (df_clean["basilar_skull_fx"] == 0)
        & (df_clean["scalp_hematoma"] == 0)
        & (df_clean["fontanelle_bulging"] == 0)
        & (df_clean["neuro_deficit"] == 0)
        & (df_clean["altered_mental_status"] == 0)
        & (df_clean["gcs_total"] == 15)
        & ((df_clean["amnesia"] == 0) | (df_clean["amnesia"] == 91))
        & (df_clean["loc"] == 0)
        & (df_clean["seizure"] == 0)
        & ((df_clean["headache"] == 0) | (df_clean["headache"] == 91))
        & (df_clean["acting_normal"] == 1)
        & (df_clean["vomiting"] == 0)
        & (df_clean["dizziness"] == 0)
        & (df_clean["drug_intoxication"] == 0)
        & (df_clean["other_injuries"] == 0)
    )

    # 4b. Include only GCS 14-15 (minor head trauma)
    gcs_14_15_mask = df_clean["gcs_group"] == 2

    # 4c. Exclude patients with missing outcome
    has_outcome_mask = df_clean["clinically_important_tbi"].notna()

    # Combine: NOT trivial AND GCS 14-15 AND has outcome
    prep_mask = ~trivial_mask & gcs_14_15_mask & has_outcome_mask
    df_prep = df_clean[prep_mask].copy()

    # Step 5 (optional): Apply human-readable value mappings
    df_prep_mapped = None
    if mapping_file is not None:
        all_variable_maps = _parse_value_mappings(mapping_file)
        old_to_new = RENAME_MAP

        df_mapped = df_clean.copy()
        for var_name, mapping_dict in all_variable_maps.items():
            new_col = old_to_new.get(var_name, var_name)
            if new_col in df_mapped.columns:
                df_mapped[new_col] = df_mapped[new_col].map(mapping_dict)

        df_prep_mapped = df_mapped.loc[df_prep.index].copy()

    return df_prep, df_prep_mapped


if __name__ == "__main__":
    # Load the raw data
    raw_df = pd.read_csv("../data/TBI PUD 10-08-2013.csv")
    print(f"Raw data shape: {raw_df.shape}")

    # Clean without mapping
    df_prep, _ = clean_data(raw_df)
    print(f"Cleaned data shape: {df_prep.shape}")
    print(f"ciTBI prevalence: {df_prep['clinically_important_tbi'].mean():.4f}")

    # Clean with mapping
    df_prep2, df_prep_mapped = clean_data(
        raw_df,
        mapping_file="../data/TBI PUD Documentation 10-08-2013.xlsx",
    )
    print(f"\nMapped data shape: {df_prep_mapped.shape}")
    print(df_prep_mapped.head(3))
