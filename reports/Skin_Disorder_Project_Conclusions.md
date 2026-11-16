# Skin Disorder Prediction — Project Conclusions

## Project scope
This project analyses 366 records with 34 predictors and six target classes.
It investigates clinical and histopathological attributes and develops a supervised classification prototype.

## Model comparison
The highest-performing tuned configuration was ExtraTreesClassifier.
Best training cross-validation macro F1: 0.9845.
Model selection used stratified cross-validation on the training partition.

## Final holdout evaluation
- Accuracy: 0.9459
- Balanced Accuracy: 0.9472
- Macro Precision: 0.9237
- Macro Recall: 0.9472
- Macro F1: 0.9288
- Weighted F1: 0.9451

## Interpretation
Permutation importance provides an exploratory view of which predictors influence this fitted model.
It does not establish medical causation or clinical importance.

## Limitations
- The dataset is small and class-imbalanced.
- The rarest class has only 20 records in the full dataset.
- Eight Age values are missing, and one record has Age = 0.
- The held-out test set is small, so class-wise metrics may be unstable.
- External validation, prospective clinical evaluation and fairness assessment have not been performed.

## Conclusion
The project demonstrates a reproducible machine-learning workflow for classifying six skin-disorder categories in this dataset.
The model is an educational prototype only and must not be used as an autonomous diagnostic or treatment system.

## Future improvements
- Validate using a larger, independent and more representative dataset.
- Obtain expert review of feature definitions and the Age = 0 record.
- Assess calibration, uncertainty and class-specific error costs.
- Conduct external and prospective validation before considering clinical use.
- Evaluate fairness across relevant patient subgroups when suitable data is available.