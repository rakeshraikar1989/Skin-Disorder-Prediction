# Model Comparison Report

## Evaluation design
Models were compared using five-fold stratified cross-validation on the training partition. Reported values are fold means. The independent 20% stratified holdout was reserved for final evaluation.

| Model | CV Accuracy | CV Balanced Accuracy | CV Macro Precision | CV Macro Recall | CV Macro F1 |
|---|---:|---:|---:|---:|---:|
| Extra Trees | 0.982876 | 0.980463 | 0.983039 | 0.980463 | 0.980673 |
| Logistic Regression | 0.979369 | 0.977368 | 0.978504 | 0.977368 | 0.976766 |
| RBF SVM | 0.979369 | 0.977368 | 0.978504 | 0.977368 | 0.976766 |
| Random Forest | 0.972589 | 0.961852 | 0.976604 | 0.961852 | 0.966425 |
| Decision Tree | 0.952075 | 0.945884 | 0.959374 | 0.945884 | 0.948438 |
| Dummy most frequent | 0.304793 | 0.166667 | 0.050799 | 0.166667 | 0.077859 |

## Hyperparameter tuning
The best grid-search configuration was Extra Trees with 300 estimators, max_features=0.7, min_samples_leaf=1 and balanced class weights. Its mean CV macro F1 was 0.984477 (standard deviation 0.014563). Mean training macro F1 was 1.000000, indicating potential overfitting.

## Final test evaluation
Accuracy 0.945946; balanced accuracy 0.947222; macro precision 0.923737; macro recall 0.947222; macro F1 0.928837; weighted F1 0.945118.

## Recommendation
Extra Trees is selected for this project because it achieved the highest tuned cross-validation macro F1 and strong held-out test performance. This recommendation is limited to this dataset and split. It is not evidence of clinical readiness. The gap between cross-validation and test macro F1, perfect training-fold score, small sample and four test examples in class 6 require cautious interpretation.
