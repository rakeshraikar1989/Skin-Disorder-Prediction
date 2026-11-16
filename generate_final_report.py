from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether
)
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
import reportlab

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
EVAL = REPORTS / "model_evaluation"
FIGURES = REPORTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

def read_csv(name):
    path = EVAL / name
    if not path.exists():
        raise FileNotFoundError(f"Required evaluation file missing: {path}")
    return pd.read_csv(path)

comparison = read_csv("model_comparison_cross_validation.csv")
metrics_df = read_csv("final_test_metrics.csv")
class_report = read_csv("final_classification_report.csv")
importance = read_csv("permutation_feature_importance.csv")
predictions = read_csv("final_test_predictions.csv")
errors = read_csv("misclassified_test_records.csv")

# Clean labels affected by the original CSV encoding.
comparison["Model"] = comparison["Model"].replace({
    "SVM ? RBF": "SVM (RBF)",
    "Dummy ? most frequent": "Dummy (most frequent)"
})

metric_values = dict(zip(metrics_df["Metric"], metrics_df["Test Result"]))
class_report = class_report[class_report["Unnamed: 0"].astype(str).str.lower()
                            .isin(["accuracy", "macro avg", "weighted avg"]) == False].copy()
class_report.rename(columns={"Unnamed: 0": "Disease"}, inplace=True)

# Chart 1: cross-validation model comparison.
plot_data = comparison.sort_values("CV Macro F1", ascending=True)
plt.figure(figsize=(9, 5.5))
plt.barh(plot_data["Model"], plot_data["CV Macro F1"])
plt.xlabel("Mean cross-validation Macro F1")
plt.title("Model Comparison — Stratified Cross-Validation")
plt.xlim(0, 1.05)
plt.tight_layout()
plt.savefig(FIGURES / "model_comparison_macro_f1.png", dpi=180)
plt.close()

# Chart 2: final test class-wise F1.
plot_data = class_report.sort_values("f1-score", ascending=True)
plt.figure(figsize=(9, 5.5))
plt.barh(plot_data["Disease"], plot_data["f1-score"])
plt.xlabel("F1-score")
plt.title("Final Test Performance by Disease Class")
plt.xlim(0, 1.05)
plt.tight_layout()
plt.savefig(FIGURES / "final_classwise_f1.png", dpi=180)
plt.close()

# Chart 3: permutation feature importance.
plot_data = importance.head(10).sort_values("Mean Importance", ascending=True)
plt.figure(figsize=(9, 6))
plt.barh(plot_data["Feature"], plot_data["Mean Importance"],
         xerr=plot_data["Importance Std"])
plt.xlabel("Mean permutation importance")
plt.title("Top 10 Features — Held-Out Test Set")
plt.tight_layout()
plt.savefig(FIGURES / "permutation_feature_importance.png", dpi=180)
plt.close()

# Chart 4: confusion matrix from the saved held-out predictions.
class_names = [
    "Psoriasis", "Seborrheic dermatitis", "Lichen planus",
    "Pityriasis rosea", "Chronic dermatitis",
    "Pityriasis rubra pilaris"
]
cm = confusion_matrix(
    predictions["Actual Class"],
    predictions["Predicted Class"],
    labels=[1, 2, 3, 4, 5, 6]
)
fig, ax = plt.subplots(figsize=(8, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
ax.set_title("Final Test Confusion Matrix")
plt.tight_layout()
plt.savefig(FIGURES / "final_test_confusion_matrix.png", dpi=180)
plt.close()

# Ensure requirements records the report generator dependency.
requirements_path = ROOT / "requirements.txt"
if requirements_path.exists():
    req_lines = requirements_path.read_text(encoding="utf-8").splitlines()
    req_lines = [line for line in req_lines if not line.lower().startswith("reportlab")]
    req_lines.append(f"reportlab=={reportlab.Version}")
    requirements_path.write_text("\n".join(req_lines) + "\n", encoding="utf-8")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="CoverTitle", parent=styles["Title"], fontSize=22,
    leading=28, alignment=TA_CENTER, spaceAfter=12
))
styles.add(ParagraphStyle(
    name="Section", parent=styles["Heading1"], fontSize=15,
    leading=19, spaceBefore=12, spaceAfter=7, textColor=colors.HexColor("#17365D")
))
styles.add(ParagraphStyle(
    name="SubSection", parent=styles["Heading2"], fontSize=11,
    leading=14, spaceBefore=8, spaceAfter=5
))
styles.add(ParagraphStyle(
    name="SmallBody", parent=styles["BodyText"], fontSize=8.5,
    leading=12, spaceAfter=5
))
styles.add(ParagraphStyle(
    name="SmallCell", parent=styles["BodyText"], fontSize=7,
    leading=9
))
styles.add(ParagraphStyle(
    name="Disclaimer", parent=styles["BodyText"], fontSize=9,
    leading=13, borderColor=colors.HexColor("#A61C1C"),
    borderWidth=0.8, borderPadding=8, spaceBefore=8, spaceAfter=8
))

def P(text, style="SmallBody"):
    return Paragraph(str(text), styles[style])

def make_table(rows, widths=None, font_size=7):
    converted = []
    for row in rows:
        converted.append([
            cell if isinstance(cell, Paragraph) else P(cell, "SmallCell")
            for cell in row
        ])
    t = Table(converted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AAB7C4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(18 * mm, 12 * mm, "Skin Disorder Prediction | Educational ML Project")
    canvas.drawRightString(192 * mm, 12 * mm, f"Page {doc.page}")
    canvas.restoreState()

pdf_path = REPORTS / "Skin_Disorder_Prediction_Final_Report.pdf"
doc = SimpleDocTemplate(
    str(pdf_path), pagesize=A4,
    rightMargin=17 * mm, leftMargin=17 * mm,
    topMargin=17 * mm, bottomMargin=20 * mm
)
story = []

# Cover.
story += [
    Spacer(1, 25 * mm),
    P("SKIN DISORDER PREDICTION", "CoverTitle"),
    P("Data Analysis, Machine Learning Evaluation and Decision-Support Recommendations",
      "Heading2"),
    Spacer(1, 8 * mm),
    P("<b>Institute project identifier:</b> PRCP-1027"),
    P("<b>Project folder/archive identifier:</b> PRCP-1028"),
    P("<b>Dataset:</b> UCI Dermatology dataset"),
    P("<b>Dataset size:</b> 366 records, 34 predictors and 6 target classes"),
    P("<b>Selected model:</b> Tuned Extra Trees Classifier"),
    Spacer(1, 10 * mm),
    P("Final Test Accuracy: 94.59%", "Heading2"),
    P("Final Test Macro F1: 92.88%", "Heading2"),
    Spacer(1, 12 * mm),
    P("Final Project Report", "Heading2"),
    P("Prepared as an educational machine-learning project. This report documents the dataset, analytical process, model evaluation, error analysis, limitations and potential clinical workflow considerations."),
    PageBreak()
]

story += [
    P("1. Executive Summary", "Section"),
    P(
        "This project analyses the UCI Dermatology dataset and develops a six-class "
        "classification model using structured clinical and histopathological attributes. "
        "The selected tuned Extra Trees model achieved 94.59% accuracy, 94.72% balanced "
        "accuracy and 92.88% macro F1 on the held-out test set of 74 records. Four test "
        "records were misclassified. These results are promising for an educational "
        "prototype, but the small dataset and limited rare-class test support prevent "
        "claims of clinical readiness or generalisation to real-world populations."
    ),
    P("2. Project Objectives and Scope", "Section"),
    P("The project addresses three institute objectives:"),
    P("1. Conduct exploratory data analysis and prepare a complete data analysis report."),
    P("2. Build and evaluate machine-learning models to predict the six skin-disorder classes."),
    P("3. Provide cautious, evidence-informed suggestions for supporting doctors in early identification."),
    P("The model is a research and learning prototype. It is not a diagnostic device and has not undergone prospective clinical or external validation."),
    P("3. Dataset Description", "Section"),
    P(
        "The dataset contains 366 observations and 35 columns: 34 predictor attributes "
        "and one target variable named class. The predictors include ordinal-coded "
        "clinical attributes, histopathological attributes, a binary family-history "
        "attribute and age. The six target classes are:"
    )
]

disease_rows = [["Class", "Disease"]]
for i, name in enumerate(class_names, 1):
    disease_rows.append([str(i), name])
story.append(make_table(disease_rows, [25 * mm, 140 * mm]))

story += [
    Spacer(1, 5),
    P(
        "Source: Ilter, N. and Guvenir, H. (1998). Dermatology. UCI Machine Learning "
        "Repository. https://doi.org/10.24432/C5FK5P"
    ),
    P("4. Data Quality and Exploratory Analysis", "Section"),
    P(
        "<b>Missing values:</b> Age contains 8 missing entries (2.19%). The source "
        "uses '?' to represent missing values. Missing-value handling is performed "
        "within the modelling workflow rather than by fitting preprocessing on the "
        "entire dataset."
    ),
    P(
        "<b>Potential age anomaly:</b> One record has Age = 0. The source documentation "
        "does not establish whether this is valid, so the value was retained and "
        "identified as a data-quality question rather than silently changed."
    ),
    P(
        "<b>Class distribution:</b> Class 1 has 112 records and class 6 has 20 records, "
        "giving an imbalance ratio of 5.6. The minority class therefore requires "
        "particular caution when interpreting performance."
    ),
    P(
        "<b>Duplicates:</b> No exact duplicate rows or duplicate predictor vectors "
        "were identified. No constant predictors were found."
    ),
    P(
        "<b>Correlations:</b> Several predictors are strongly correlated. Correlation "
        "may reflect shared underlying patterns or differences between disease classes; "
        "correlation alone is not sufficient justification for removing predictors."
    ),
    P(
        "<b>Sparse attributes:</b> Some predictors contain many zero values, including "
        "perifollicular parakeratosis and follicular horn plug. Their clinical and "
        "predictive relevance should be interpreted in the context of the dataset."
    ),
    P("5. Modelling Methodology", "Section"),
    P(
        "The workflow includes data validation, exploratory analysis, preprocessing, "
        "baseline comparison, multiple classification algorithms, stratified "
        "cross-validation, hyperparameter tuning and final evaluation on a held-out "
        "test partition. Preprocessing is fitted within the modelling process to "
        "reduce leakage risk. The target class is excluded from predictor inputs."
    ),
    P(
        "The model comparison uses stratified cross-validation on the training "
        "partition. The final test metrics below are calculated on the separate "
        "held-out test partition and are not cross-validation estimates."
    ),
    P("6. Cross-Validation Model Comparison", "Section"),
    P("The following table compares the saved cross-validation results using Macro F1."),
]

cv_rows = [["Model", "Accuracy", "Balanced Acc.", "Macro F1"]]
for _, r in comparison.iterrows():
    cv_rows.append([
        r["Model"],
        f'{r["CV Accuracy"]:.4f}',
        f'{r["CV Balanced Accuracy"]:.4f}',
        f'{r["CV Macro F1"]:.4f}'
    ])
story.append(make_table(cv_rows, [55 * mm, 30 * mm, 35 * mm, 25 * mm]))
story += [
    Spacer(1, 5),
    Image(str(FIGURES / "model_comparison_macro_f1.png"), width=165 * mm, height=100 * mm),
    P(
        "Extra Trees achieved the highest recorded cross-validation Macro F1 in the "
        "saved comparison. The Dummy classifier provides a baseline reference and "
        "shows why accuracy alone is insufficient for an imbalanced multiclass dataset."
    ),
    P("7. Hyperparameter Selection", "Section"),
    P(
        "The selected Extra Trees configuration uses 300 estimators, max_features=0.7, "
        "min_samples_leaf=1, class_weight='balanced', random_state=42 and n_jobs=1. "
        "The selected configuration achieved mean cross-validation Macro F1 of "
        "0.98448 (standard deviation 0.01456). The mean training Macro F1 was 1.0, "
        "which is a potential overfitting signal and should not be interpreted as "
        "evidence of perfect generalisation."
    ),
    P("8. Final Held-Out Test Results", "Section"),
]

metric_order = [
    "Accuracy", "Balanced Accuracy", "Macro Precision",
    "Macro Recall", "Macro F1", "Weighted F1"
]
metric_rows = [["Metric", "Test result"]]
for name in metric_order:
    if name in metric_values:
        metric_rows.append([name, f"{float(metric_values[name]):.4f}"])
story.append(make_table(metric_rows, [75 * mm, 45 * mm]))
story += [
    Spacer(1, 5),
    P(
        "The held-out test set contains 74 records. Macro averages give each class "
        "equal weight, while weighted F1 accounts for class support. Balanced accuracy "
        "is included because the target distribution is imbalanced."
    ),
    P("9. Class-Wise Test Performance", "Section"),
]

class_rows = [["Disease", "Precision", "Recall", "F1", "Support"]]
for _, r in class_report.iterrows():
    class_rows.append([
        r["Disease"],
        f'{r["precision"]:.3f}',
        f'{r["recall"]:.3f}',
        f'{r["f1-score"]:.3f}',
        str(int(r["support"]))
    ])
story.append(make_table(class_rows, [62 * mm, 23 * mm, 23 * mm, 23 * mm, 22 * mm]))
story += [
    Spacer(1, 5),
    Image(str(FIGURES / "final_classwise_f1.png"), width=165 * mm, height=100 * mm),
    Image(str(FIGURES / "final_test_confusion_matrix.png"), width=155 * mm, height=135 * mm),
    P(
        "The rarest class, Pityriasis rubra pilaris, has only four records in the test "
        "set. Its observed result is therefore highly uncertain. Correct classification "
        "of these four cases does not establish reliable performance for this disease."
    ),
    P("10. Error Analysis", "Section"),
    P(f"The saved test predictions contain {len(predictions)} records, of which "
      f"{len(errors)} were misclassified."),
]

error_rows = [["Actual disease", "Predicted disease", "Count"]]
if not errors.empty:
    grouped = errors.groupby(
        ["Actual Disease", "Predicted Disease"]
    ).size().reset_index(name="Count")
    for _, r in grouped.iterrows():
        error_rows.append([r["Actual Disease"], r["Predicted Disease"], str(r["Count"])])
story.append(make_table(error_rows, [65 * mm, 65 * mm, 20 * mm]))
story += [
    P(
        "The observed errors include confusion between seborrheic dermatitis and "
        "pityriasis rubra pilaris, between lichen planus and chronic dermatitis, and "
        "between seborrheic dermatitis and pityriasis rosea. These patterns identify "
        "areas for further investigation; they do not establish why the model made "
        "each individual error."
    ),
    P("11. Feature Importance", "Section"),
    P(
        "Permutation importance was calculated on the held-out test set. The leading "
        "features in this run were fibrosis of the papillary dermis, perifollicular "
        "parakeratosis, Koebner phenomenon and disappearance of the granular layer."
    ),
    Image(str(FIGURES / "permutation_feature_importance.png"), width=165 * mm, height=112 * mm),
    P(
        "Permutation importance measures the change in model performance when a "
        "feature is disrupted. It is sample-dependent, can be affected by correlated "
        "predictors and does not demonstrate causality or clinical importance."
    ),
    P("12. Suggestions for Doctor-Support Workflows", "Section"),
    P(
        "1. Use a structured clinical record to capture relevant observations such as "
        "erythema, scaling, lesion borders, itching, Koebner phenomenon, papules and "
        "the anatomical distribution of findings."
    ),
    P(
        "2. Record patient history and examination findings consistently. Missing "
        "information should be marked as missing rather than assumed to be absent."
    ),
    P(
        "3. Consider histopathological findings where clinically indicated and "
        "interpret them alongside the patient's history and clinical examination."
    ),
    P(
        "4. If an appropriately validated decision-support system is available, treat "
        "its output only as a prompt to review a differential diagnosis. A qualified "
        "clinician must independently assess and confirm the diagnosis."
    ),
    P(
        "5. Do not dismiss a rare disease solely because it is under-represented in "
        "a training dataset. Escalate uncertainty and clinically concerning findings "
        "for professional assessment."
    ),
    P("13. Challenges and Mitigation", "Section"),
    P(
        "The main constraints were the small sample size, class imbalance, eight "
        "missing age values, an unresolved age-zero record, correlated and sparse "
        "predictors, and the risk of overfitting. The modelling workflow uses "
        "stratified validation and a held-out test set, reports class-wise metrics, "
        "and retains the unresolved age value transparently. External validation "
        "remains necessary."
    ),
    P("14. Limitations and Future Work", "Section"),
    P(
        "The dataset is small and may not represent current clinical populations, "
        "different institutions, demographic groups or variations in clinical "
        "practice. Test estimates—especially for the smallest class—have substantial "
        "uncertainty. The project has no prospective clinical evaluation, external "
        "validation cohort or evidence of improved patient outcomes."
    ),
    P(
        "Future work should prioritise independent external validation, larger and "
        "more representative datasets, subgroup performance analysis, uncertainty "
        "assessment, calibration, clinician-led review and governance before any "
        "real-world use is considered."
    ),
    P("15. Reproducibility and Project Assets", "Section"),
    P(
        "The project includes the original dataset, analysis notebook, saved model, "
        "evaluation CSV outputs, source modules for data validation and prediction, "
        "automated tests, requirements file, README, project documentation, model "
        "comparison report and challenges report. The saved model can be loaded for "
        "schema-validated batch prediction using the project's prediction module."
    ),
    P("16. Medical and Intended-Use Disclaimer", "Section"),
    P(
        "<b>This project is an educational machine-learning prototype only.</b> It "
        "is not clinically validated, is not approved for medical use, and must not "
        "be used to diagnose, rule out, triage or treat any patient. It must not "
        "replace examination, histopathology where indicated, professional judgement "
        "or consultation with a qualified dermatologist. No patient-care decision "
        "should be made from this model's output.",
        "Disclaimer"
    ),
    Spacer(1, 5),
    P("End of Report", "Heading2")
]

doc.build(story, onFirstPage=footer, onLaterPages=footer)

print("\nFINAL REPORT GENERATED")
print("PDF:", pdf_path)
print("PDF SIZE:", pdf_path.stat().st_size, "bytes")
print("\nGENERATED FIGURES")
for path in sorted(FIGURES.glob("*.png")):
    print(path.name, "-", path.stat().st_size, "bytes")
print("\nREQUIREMENTS UPDATED")
print("ReportLab:", reportlab.Version)
