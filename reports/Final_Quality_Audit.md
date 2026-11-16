# Final Quality Audit — Skin Disorder Prediction

**Project folder ID:** PRCP-1028
**Institute problem statement ID:** PRCP-1027
**Audit date:** 16 November 2026

## 1. Technical Quality Audit

| Check | Result |
|---|---|
| Automated unit tests | PASS — 14 tests |
| Original dataset dimensions | PASS — (366, 35) |
| Original dataset SHA256 | `4c398eba26590ea7daad1c8405f971a669da5fc6f880d69dfdeea610c1c2f68b` |
| Missing Age values | 8 |
| Saved model loads | PASS — Pipeline |
| Model feature count | 34 |
| Model feature schema matches dataset | PASS |
| Notebook | 46 cells; 16 code cells; all code cells executed; no execution errors |
| Final PDF | PASS — 332817 bytes |
| Supporting figures | 4 generated PNG files |
| Required project files | PASS |

### Technical audit observations

- All 14 automated tests passed, covering dataset validation, model loading, prediction outputs and invalid prediction inputs.
- The original dataset checksum matches the previously verified source file.
- The saved model's 34 input features match the dataset predictors.
- The prediction module validates input files, feature schemas and numeric values. Fourteen automated tests cover valid and invalid dataset and prediction scenarios.
- The training workflow remains primarily notebook-based; the project is not yet a fully packaged production training service.

## 2. Analytical Quality Audit

| Check | Verified result |
|---|---|
| Dataset size | 366 records, 34 predictors and one target |
| Disease classes | 6 |
| Class distribution | {1: 112, 2: 61, 3: 72, 4: 49, 5: 52, 6: 20} |
| Model-comparison Extra Trees CV Macro F1 (untuned) | 0.980673 |
| Test accuracy | 0.945946 |
| Test balanced accuracy | 0.947222 |
| Test macro precision | 0.923737 |
| Test macro recall | 0.947222 |
| Test macro F1 | 0.928837 |
| Test weighted F1 | 0.945118 |
| Held-out test records | 74 |
| Correctly counted disease-class support | 74 |
| Misclassified test records | 4 |

### Analytical audit observations and limitations

- The classification report contains six disease rows plus accuracy, macro-average and weighted-average summary rows. The six disease rows alone sum to 74 test records; summing every displayed row would incorrectly produce 222.
- The smallest original disease class contains only 20 records. Its held-out test support is four, so its apparent class-level performance is statistically fragile.
- The tuned Extra Trees model achieved a mean training Macro F1 of 1.0 during cross-validation. This is a warning sign for potential overfitting and should be disclosed.
- The held-out test results are encouraging but do not establish performance on new hospitals, populations, devices or clinical settings.
- Permutation importance is model- and sample-dependent. It is not evidence of causality or clinical importance.
- The single Age value of zero was retained pending source clarification; the eight missing Age values require transparent handling in the preprocessing pipeline.
- External validation, prospective clinical evaluation, confidence intervals and probability calibration are not established by this project.

## 3. Institute Requirements and Professional Compliance Audit

| Institute requirement | Evidence / status |
|---|---|
| Complete data analysis report | Notebook, project documentation and final PDF |
| Predict different skin disease classes | Six-class saved pipeline and prediction module |
| Suggestions to doctors for early identification | `doctor_early_identification_suggestions.csv` and notebook/report content |
| Model Comparison Report | `reports/Model_Comparison_Report.md` |
| Challenges Report | `reports/Challenges_Report.md` |
| All three main tasks in one notebook | `notebooks/Skin_Disorder_Prediction.ipynb` |
| Reproducibility and test evidence | requirements file, saved model, source modules and passing tests |
| Professional documentation | README, project documentation, reports and final PDF |

### Medical and corporate compliance

- This work is an educational machine-learning prototype, not a clinically validated diagnostic device.
- Model predictions must not replace a dermatologist's examination, clinical history, investigations or professional judgement.
- The model must not be used to prescribe treatment or to rule out a rare disease.
- Deployment would require independent external validation, clinical governance, privacy/security review and appropriate regulatory assessment.

## Final Audit Conclusion

**Technical status:** Automated tests, dataset integrity, model loading and feature-schema checks passed.

**Analytical status:** Core evaluation outputs are present and internally consistent with a 74-record held-out test set. Important limitations — small sample size, rare-class uncertainty, potential overfitting and absence of external clinical validation — must remain visible in the final presentation.

**Institute coverage:** The required notebook, analysis, six-class prediction, doctor suggestions, model comparison and challenges documentation are present.

**Overall assessment:** The project package is substantially complete for educational portfolio and institute-project purposes, with the final PDF and notebook presentation review completed. It is not approved or represented as a clinically deployable system.

## Audit execution

- Automated test exit code: `0`
- Missing required files: `None`
- Generated figures: `['final_classwise_f1.png', 'final_test_confusion_matrix.png', 'model_comparison_macro_f1.png', 'permutation_feature_importance.png']`


## Cross-validation results clarification

The two Extra Trees cross-validation results refer to different stages and must not be treated as interchangeable:

- **Initial model comparison:** Extra Trees CV Macro F1 = 0.980673.
- **Tuned Extra Trees model:** mean CV Macro F1 = 0.984477, with standard deviation = 0.014563.
- **Mean training-fold Macro F1 after tuning:** 1.000000. This indicates a possible overfitting/generalisation gap and is a limitation, not evidence of perfect generalisation.
- **Final held-out test Macro F1:** 0.928837, evaluated on 74 records.

The revised notebook was restarted and run from beginning to end. Verification recorded 46 total cells, 16 code cells, no unexecuted code cells, 82 outputs, and no execution errors. All 14 automated unit tests passed.

Final review completed on 16 November 2026.

- All four PDF charts were visually inspected for readability, labels and clipping.
- Notebook and PDF results were compared, including CV Macro F1, test accuracy, balanced accuracy, test Macro F1 and misclassification counts.
- Both remaining notebook output headings were corrected and verified.
- The notebook contains 46 cells and zero saved execution errors.
- All 14 automated tests passed in the recorded verification run.
- Final status: PASS for educational portfolio and institute-project purposes. The model is not clinically validated or approved for medical deployment.
