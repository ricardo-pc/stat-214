# Here, I just keep all the feature categorization and parent-child group info in one place for easy reference.
# This is not strictly necessary but can be convenient for organizing the code and avoiding circular imports between
# clean.py and preprocess.py.


# categorize columns
target_col = 'PosIntFinal'
identifier_cols = ['PatNum','EmplType','Certification']
numeric_cols = ['GCSTotal','AgeInMonth','AgeinYears','GCSMotor','GCSVerbal','GCSEye']
categorical_cols = [
    'AMS', 'AMSAgitated', 'AMSOth', 'AMSRepeat', 'AMSSleep', 'AMSSlow',
    'ActNorm', 'AgeTwoPlus', 'Amnesia_verb', 'CTDone', 'CTForm1', 'CTSed',
    'CTSedAge', 'CTSedAgitate', 'CTSedOth', 'CTSedRqst', 'Clav', 'ClavFace',
    'ClavFro', 'ClavNeck', 'ClavOcc', 'ClavPar', 'ClavTem', 'DeathTBI',
    'Dizzy', 'Drugs', 'EDCT', 'EDDisposition', 'Ethnicity', 'Finding1',
    'Finding10', 'Finding11', 'Finding12', 'Finding13', 'Finding14',
    'Finding2', 'Finding20', 'Finding21', 'Finding22', 'Finding23',
    'Finding3', 'Finding4', 'Finding5', 'Finding6', 'Finding7', 'Finding8',
    'Finding9', 'FontBulg', 'GCSGroup', 'Gender', 'HASeverity', 'HAStart',
    'HA_verb', 'Hema', 'HemaLoc', 'HemaSize', 'High_impact_InjSev',
    'HospHead', 'HospHeadPosCT', 'IndAMS', 'IndAge', 'IndAmnesia',
    'IndClinSFx', 'IndHA', 'IndHema', 'IndLOC', 'IndMech', 'IndNeuroD',
    'IndOth', 'IndRqstMD', 'IndRqstParent', 'IndRqstTrauma', 'IndSeiz',
    'IndVomit', 'IndXraySFx', 'InjuryMech', 'Intub24Head', 'Intubated',
    'LOCSeparate', 'LocLen', 'NeuroD', 'NeuroDCranial', 'NeuroDMotor',
    'NeuroDOth', 'NeuroDReflex', 'NeuroDSensory', 'Neurosurgery', 'OSI',
    'OSIAbdomen', 'OSICspine', 'OSICut', 'OSIExtremity', 'OSIFlank',
    'OSIOth', 'OSIPelvis', 'Observed', 'Paralyzed', 'PosCT', 'PosIntFinal',
    'Race', 'SFxBas', 'SFxBasHem', 'SFxBasOto', 'SFxBasPer', 'SFxBasRet',
    'SFxBasRhi', 'SFxPalp', 'SFxPalpDepress', 'Sedated', 'Seiz', 'SeizLen',
    'SeizOccur', 'Vomit', 'VomitLast', 'VomitNbr', 'VomitStart'
]

binary_cols = [
    'AMS', 'ActNorm', 'AgeTwoPlus', 'CTDone', 'CTForm1', 'Clav', 'DeathTBI',
    'Dizzy', 'Drugs', 'Ethnicity', 'FontBulg', 'GCSGroup', 'Gender', 'Hema',
    'HospHead', 'HospHeadPosCT', 'Intub24Head', 'Intubated', 'NeuroD',
    'Neurosurgery', 'OSI', 'Observed', 'Paralyzed', 'PosIntFinal', 'SFxBas',
    'Sedated', 'Seiz', 'Vomit'
]
multi_category_cols = list(set(categorical_cols) - set(binary_cols))

# keep a dict for parent-child relationships
parent_child_groups = {
    # Symptoms / history
    "LOCSeparate": ["LocLen"],
    "Seiz": ["SeizOccur", "SeizLen"],
    "HA_verb": ["HASeverity", "HAStart"],
    "Vomit": ["VomitNbr", "VomitStart", "VomitLast"],

    # AMS and sub-symptoms
    "AMS": ["AMSAgitated", "AMSSleep", "AMSSlow", "AMSRepeat", "AMSOth"],

    # Skull fracture details
    "SFxPalp": ["SFxPalpDepress"],
    "SFxBas": ["SFxBasHem", "SFxBasOto", "SFxBasPer", "SFxBasRet", "SFxBasRhi"],

    # Hematoma details
    "Hema": ["HemaLoc", "HemaSize"],

    # Trauma above clavicles by region
    "Clav": ["ClavFace", "ClavNeck", "ClavFro", "ClavOcc", "ClavPar", "ClavTem"],

    # Neuro deficit details
    "NeuroD": ["NeuroDMotor", "NeuroDSensory", "NeuroDCranial", "NeuroDReflex", "NeuroDOth"],

    # Other substantial injury details
    "OSI": ["OSIExtremity", "OSICut", "OSICspine", "OSIFlank", "OSIAbdomen", "OSIPelvis", "OSIOth"],

    # CT ordered/obtained: indication checklist + sedation
    "CTForm1": [
        "IndAge", "IndAmnesia", "IndAMS", "IndClinSFx", "IndHA", "IndHema", "IndLOC",
        "IndMech", "IndNeuroD", "IndRqstMD", "IndRqstParent", "IndRqstTrauma",
        "IndSeiz", "IndVomit", "IndXraySFx", "IndOth",
        "CTSed"
    ],

    # Sedation reasons
    "CTSed": ["CTSedAgitate", "CTSedAge", "CTSedRqst", "CTSedOth"],

    # CT performed: CT-location + PI findings
    "CTDone": [
        "EDCT", "PosCT",
        "Finding1", "Finding2", "Finding3", "Finding4", "Finding5", "Finding6", "Finding7", "Finding8",
        "Finding9", "Finding10", "Finding11", "Finding12", "Finding13", "Finding14",
        "Finding20", "Finding21", "Finding22", "Finding23"
    ],
}