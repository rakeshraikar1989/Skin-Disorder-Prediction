# Challenges and Solutions Report

| Challenge | Evidence | Approach | Remaining limitation |
|---|---|---|---|
| Small dataset | 366 records across six classes | Stratified split and cross-validation | Generalisation to new populations is uncertain |
| Class imbalance | Largest class 112; smallest 20 | Macro metrics, balanced accuracy, class weighting | Rare-class estimates remain uncertain |
| Missing age | Eight missing values | Median imputation inside pipeline | Imputed values are estimates |
| Age equals zero | One record | Retained pending source clarification | Validity is unknown |
| Correlated predictors | Several high Spearman correlations | Investigated; not automatically removed | Importance can be difficult to interpret |
| Sparse predictors | Several attributes mostly zero | Retained and evaluated | Rare findings have limited support |
| Medical application risk | Historical dataset, no clinical validation | Educational prototype and human oversight | No clinical safety or deployment evidence |
| Generalisation | Small holdout | Training-only CV and untouched test partition | Independent external validation is required |

## Overall assessment
The workflow addresses major data-quality, imbalance and leakage risks, but the dataset size and provenance constrain conclusions. Future work should prioritise external validation, additional representative cases, calibration, subgroup analysis and clinical review.
