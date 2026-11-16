# Skin Disorder Prediction

**Institute project:** PRCP-1027 — Skin Disorder  
**Project folder/archive identifier:** PRCP-1028  
**Project type:** Educational multiclass machine-learning classification prototype

## Objective
Analyse the dermatology dataset, compare classification models for six disease classes, and develop cautious suggestions for supporting early identification. This project is not a clinical diagnostic system.

## Dataset
The project uses the Dermatology dataset from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/33/dermatology

Citation: Ilter, N. and Guvenir, H. (1998). Dermatology. UCI Machine Learning Repository. https://doi.org/10.24432/C5FK5P

The analysed dataset contains 366 records, 34 predictors and six target classes. Eight Age values are missing. One Age value is zero and was retained pending clarification. The raw source file is preserved.

## Model and evaluation
The selected estimator is an Extra Trees classifier in a preprocessing pipeline. The tuned configuration uses 300 estimators, max_features=0.7, min_samples_leaf=1 and balanced class weights.

| Evaluation | Result |
|---|---:|
| Tuned training-fold CV macro F1 | 0.984477 |
| CV standard deviation | 0.014563 |
| Held-out test accuracy | 0.945946 |
| Held-out test balanced accuracy | 0.947222 |
| Held-out test macro precision | 0.923737 |
| Held-out test macro recall | 0.947222 |
| Held-out test macro F1 | 0.928837 |
| Held-out test weighted F1 | 0.945118 |

The training-fold mean macro F1 for the selected configuration was 1.0. This indicates potential overfitting. The test set contains 74 records; the rarest class has only four test examples. Results therefore have substantial sampling uncertainty and require external validation.

## Repository structure
- `data/raw/` — original dataset
- `notebooks/` — complete analysis and modelling notebook
- `src/` — reusable validation and prediction modules
- `models/` — saved fitted model
- `reports/model_evaluation/` — model metrics, predictions and error analysis
- `reports/` — project conclusions and supporting reports
- `documentation/` — detailed project documentation
- `tests/` — automated integrity and prediction tests

## Environment
Python 3.13.17. Main libraries include pandas, NumPy, scikit-learn, matplotlib, seaborn and joblib. See `requirements.txt`.

## Run automated tests
From the project root:
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Run predictions
Prepare a CSV with exactly the 34 predictor columns, without the target column:
```powershell
.\.venv\Scripts\python.exe -m src.predict --input input.csv --output predictions.csv
```

## Important medical limitation
This model is an educational research prototype trained on a small historical dataset. It has not undergone prospective clinical validation, independent external validation, fairness assessment, regulatory review or clinical safety testing. It must not be used to diagnose patients, recommend treatment or replace a qualified dermatologist.
