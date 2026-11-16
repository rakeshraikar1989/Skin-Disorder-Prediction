# Project Documentation — Skin Disorder Prediction

## 1. Project identity and objective
Institute problem statement: PRCP-1027 — Skin Disorder. Project folder/archive identifier: PRCP-1028.

The project performs exploratory data analysis, develops a six-class classifier and derives cautious suggestions for structured clinical assessment and early identification.

## 2. Data source and structure
Source: UCI Machine Learning Repository, Dermatology dataset (Ilter and Guvenir, 1998), DOI 10.24432/C5FK5P.

The selected source file contains 366 observations, 34 predictors and one target column. Predictors include scored clinical attributes, a binary family-history attribute, histopathological attributes and age. The target has six disease classes.

## 3. Data quality
Eight Age entries are missing. These are handled using median imputation inside the modelling pipeline, so imputation is learned within training folds. One Age value of zero is retained because the source does not establish whether it is invalid. No rows or predictors were silently removed. The raw file is preserved.

The target distribution is imbalanced: class 1 has 112 records and class 6 has 20. Several predictors are sparse and some pairs are strongly correlated. These findings are documented in the notebook.

## 4. Methodology
A stratified 80:20 train/test split is used. Model development and tuning are performed using five-fold stratified cross-validation on the training partition. Preprocessing is contained in the pipeline to reduce leakage risk. The held-out test partition is used for final evaluation after model selection.

The baseline is a most-frequent DummyClassifier. Candidate models include Logistic Regression, Decision Tree, RBF SVM, Random Forest and Extra Trees. Macro F1 and balanced accuracy are reported alongside accuracy and other metrics to reflect class imbalance.

## 5. Selected model
The tuned Extra Trees configuration is:
- n_estimators: 300
- max_features: 0.7
- min_samples_leaf: 1
- class_weight: balanced
- random_state: 42

Tuned training-fold cross-validation macro F1: 0.984477 (standard deviation 0.014563). Mean training macro F1 was 1.000000, indicating possible overfitting.

## 6. Final held-out test results
- Accuracy: 0.945946
- Balanced accuracy: 0.947222
- Macro precision: 0.923737
- Macro recall: 0.947222
- Macro F1: 0.928837
- Weighted F1: 0.945118

The test partition contains 74 records. Per-class supports are 23, 12, 15, 10, 10 and 4. Performance for the rarest class is particularly uncertain because it is based on only four test observations.

## 7. Error analysis
Four test records were misclassified. Two Seborrheic dermatitis records were predicted as Pityriasis rosea; one Seborrheic dermatitis record was predicted as Pityriasis rubra pilaris; and one Lichen planus record was predicted as Chronic dermatitis. These errors show that high aggregate metrics do not eliminate class-level diagnostic ambiguity.

## 8. Model interpretation
Permutation importance on the held-out test set ranked fibrosis of the papillary dermis and perifollicular parakeratosis highest in this evaluation. Permutation importance is model- and sample-dependent; it does not establish causality or clinical importance. Correlated predictors may distribute or distort importance.

## 9. Suggestions for clinical workflow research
The project suggests consistent recording of clinical findings, complete patient records, appropriate use of histopathological evidence where clinically indicated, careful differential diagnosis and mandatory clinician oversight. These are research suggestions based on dataset attributes, not medical advice or a validated clinical protocol.

## 10. Limitations and future work
The dataset is small and historical, class distribution is imbalanced, and the rarest class has limited test support. The selected model's training score is perfect, suggesting potential overfitting. No external validation, prospective study, subgroup fairness analysis, calibration assessment, clinical utility evaluation or regulatory review has been completed. Future work should prioritise independent representative data, external validation, confidence and calibration analysis, and review by qualified dermatology professionals.

## 11. Reproducibility
The notebook records the analysis and modelling workflow. A fixed random seed of 42 is used. The fitted pipeline is saved under `models/`. The environment versions are recorded in `requirements.txt`. Automated integrity tests are provided under `tests/`.

## 12. Medical disclaimer
This is an educational machine-learning prototype only. It is not a medical device and must not be used to diagnose, triage or treat patients. Diagnosis and care decisions remain the responsibility of qualified healthcare professionals.
