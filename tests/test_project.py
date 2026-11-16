import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_validation import load_and_validate_dataset
from src.predict import predict_file

DATA = ROOT / "data" / "raw" / "dataset_35_dermatology.csv"
MODEL = ROOT / "models" / "skin_disorder_best_model.joblib"


class SkinDisorderProjectTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = load_and_validate_dataset(DATA)
        cls.model = joblib.load(MODEL)

    def test_dataset_schema_classes_and_missing_age(self):
        self.assertEqual(self.data.shape, (366, 35))
        self.assertEqual(self.data["class"].nunique(), 6)
        self.assertEqual(self.data["Age"].isna().sum(), 8)
        self.assertEqual(len(self.model.feature_names_in_), 34)

    def test_missing_dataset_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_and_validate_dataset(ROOT / "does_not_exist.csv")

    def test_dataset_with_wrong_row_count_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wrong_rows.csv"
            self.data.iloc[:-1].to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_and_validate_dataset(path)

    def test_dataset_with_missing_target_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing_target.csv"
            altered = self.data.copy()
            altered.loc[0, "class"] = None
            altered.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_and_validate_dataset(path)

    def test_dataset_with_invalid_class_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid_class.csv"
            altered = self.data.copy()
            altered.loc[0, "class"] = 9
            altered.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_and_validate_dataset(path)

    def test_dataset_with_non_numeric_feature_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid_feature.csv"
            altered = self.data.copy()
            altered["erythema"] = altered["erythema"].astype(object)
            altered.loc[0, "erythema"] = "invalid"
            altered.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_and_validate_dataset(path)

    def test_model_file_exists_and_loads(self):
        self.assertTrue(MODEL.is_file())
        self.assertEqual(len(self.model.feature_names_in_), 34)

    def test_model_predicts_valid_classes(self):
        X = self.data.drop(columns=["class"]).head(10)
        predictions = self.model.predict(X)
        self.assertEqual(len(predictions), 10)
        self.assertTrue(set(predictions).issubset({1, 2, 3, 4, 5, 6}))

    def test_prediction_file_is_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.csv"
            output_path = Path(tmp) / "nested" / "predictions.csv"
            self.data.drop(columns=["class"]).head(5).to_csv(
                input_path, index=False
            )
            result = predict_file(input_path, MODEL, output_path)
            self.assertTrue(output_path.is_file())
            self.assertEqual(len(result), 5)
            self.assertIn("predicted_class", result.columns)
            self.assertTrue(
                set(result["predicted_class"]).issubset({1, 2, 3, 4, 5, 6})
            )

    def test_prediction_rejects_missing_feature(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing_feature.csv"
            X = self.data.drop(columns=["class"]).head(2)
            X.drop(columns=["Age"]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                predict_file(path, MODEL, Path(tmp) / "out.csv")

    def test_prediction_rejects_extra_feature(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "extra_feature.csv"
            X = self.data.drop(columns=["class"]).head(2).copy()
            X["unexpected_feature"] = 1
            X.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                predict_file(path, MODEL, Path(tmp) / "out.csv")

    def test_prediction_rejects_non_numeric_feature(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "non_numeric.csv"
            X = self.data.drop(columns=["class"]).head(2).copy()
            X["erythema"] = X["erythema"].astype(object)
            X.loc[X.index[0], "erythema"] = "invalid"
            X.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "non-numeric"):
                predict_file(path, MODEL, Path(tmp) / "out.csv")

    def test_prediction_rejects_empty_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.csv"
            pd.DataFrame(columns=list(self.model.feature_names_in_)).to_csv(
                path, index=False
            )
            with self.assertRaisesRegex(ValueError, "no prediction records"):
                predict_file(path, MODEL, Path(tmp) / "out.csv")

    def test_prediction_rejects_missing_input_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                predict_file(
                    Path(tmp) / "missing.csv", MODEL, Path(tmp) / "out.csv"
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
