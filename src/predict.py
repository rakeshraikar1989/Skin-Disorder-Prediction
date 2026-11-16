from pathlib import Path
import argparse

import joblib
import pandas as pd


def predict_file(input_csv, model_path, output_csv):
    """Validate an input CSV, predict classes, and save the result."""
    input_csv = Path(input_csv)
    model_path = Path(model_path)
    output_csv = Path(output_csv)

    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    if not hasattr(model, "feature_names_in_"):
        raise ValueError("Loaded model does not expose its expected feature names.")

    try:
        incoming = pd.read_csv(input_csv, na_values=["?"])
    except pd.errors.EmptyDataError as exc:
        raise ValueError("Input CSV is empty or has no header.") from exc

    if incoming.empty:
        raise ValueError("Input CSV contains no prediction records.")
    if incoming.columns.duplicated().any():
        raise ValueError("Input CSV contains duplicate column names.")

    expected = list(model.feature_names_in_)
    missing = sorted(set(expected) - set(incoming.columns))
    extra = sorted(set(incoming.columns) - set(expected))

    if missing or extra:
        raise ValueError(
            f"Input schema mismatch. Missing columns: {missing}; "
            f"unexpected columns: {extra}"
        )

    incoming = incoming[expected].copy()

    for column in expected:
        converted = pd.to_numeric(incoming[column], errors="coerce")
        invalid = incoming[column].notna() & converted.isna()
        if invalid.any():
            raise ValueError(f"Input feature '{column}' contains non-numeric values.")
        if converted.dropna().isin([float("inf"), float("-inf")]).any():
            raise ValueError(f"Input feature '{column}' contains infinite values.")
        incoming[column] = converted

    predictions = model.predict(incoming)
    result = incoming.copy()
    result["predicted_class"] = predictions

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the educational skin-disorder classification model."
    )
    parser.add_argument(
        "--input", required=True,
        help="CSV containing the 34 predictor columns."
    )
    parser.add_argument(
        "--model", default="models/skin_disorder_best_model.joblib"
    )
    parser.add_argument(
        "--output",
        default="reports/model_evaluation/predictions_from_input.csv"
    )
    args = parser.parse_args()

    result = predict_file(args.input, args.model, args.output)
    print(f"Predictions saved: {args.output}")
    print(result["predicted_class"].value_counts().sort_index())
