from pathlib import Path

import pandas as pd

EXPECTED_CLASSES = {1, 2, 3, 4, 5, 6}
TARGET = "class"

EXPECTED_FEATURES = {
    "erythema", "scaling", "definite_borders", "itching",
    "koebner_phenomenon", "polygonal_papules", "follicular_papules",
    "oral_mucosal_involvement", "knee_and_elbow_involvement",
    "scalp_involvement", "family_history", "melanin_incontinence",
    "eosinophils_in_the_infiltrate", "PNL_infiltrate",
    "fibrosis_of_the_papillary_dermis", "exocytosis", "acanthosis",
    "hyperkeratosis", "parakeratosis", "clubbing_of_the_rete_ridges",
    "elongation_of_the_rete_ridges",
    "thinning_of_the_suprapapillary_epidermis", "spongiform_pustule",
    "munro_microabcess", "focal_hypergranulosis",
    "disappearance_of_the_granular_layer",
    "vacuolisation_and_damage_of_basal_layer", "spongiosis",
    "saw-tooth_appearance_of_retes", "follicular_horn_plug",
    "perifollicular_parakeratosis",
    "inflammatory_monoluclear_inflitrate", "band-like_infiltrate", "Age",
}


def load_and_validate_dataset(path):
    """Load the project dataset and validate its expected structure."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path, na_values=["?"])

    if data.empty or data.shape != (366, 35):
        raise ValueError(
            f"Expected dataset shape (366, 35); received {data.shape}."
        )

    if data.columns.duplicated().any():
        raise ValueError("Duplicate column names detected.")

    if TARGET not in data.columns:
        raise ValueError("Target column 'class' is missing.")

    actual_features = set(data.columns) - {TARGET}
    if actual_features != EXPECTED_FEATURES:
        missing = sorted(EXPECTED_FEATURES - actual_features)
        extra = sorted(actual_features - EXPECTED_FEATURES)
        raise ValueError(
            f"Feature schema mismatch. Missing: {missing}; unexpected: {extra}"
        )

    if data[TARGET].isna().any():
        raise ValueError("Target column contains missing values.")

    if not pd.api.types.is_numeric_dtype(data[TARGET]):
        raise ValueError("Target column must contain numeric class labels.")

    observed_classes = set(data[TARGET].unique())
    if observed_classes != EXPECTED_CLASSES:
        raise ValueError(
            f"Unexpected target classes: {sorted(observed_classes)}"
        )

    for column in EXPECTED_FEATURES:
        converted = pd.to_numeric(data[column], errors="coerce")
        invalid = data[column].notna() & converted.isna()
        if invalid.any():
            raise ValueError(f"Feature '{column}' contains non-numeric values.")
        if converted.dropna().isin([float("inf"), float("-inf")]).any():
            raise ValueError(f"Feature '{column}' contains infinite values.")

    return data
