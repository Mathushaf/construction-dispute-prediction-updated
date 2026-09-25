# DisputeClassifier_RoBERTa_infer_with_cm_images.py

import os
from pathlib import Path
from datetime import datetime

# (fix the earlier typo: remove any "import numpy as pd")
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.metrics import ConfusionMatrixDisplay

# Plotting + Excel image embedding
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from openpyxl.drawing.image import Image as XLImage

# =========================
# Config (edit paths if needed)
# =========================
MODEL_DIR = "dispute_classification_model_RoBERTa"               # your fine-tuned model folder
TEST_XLSX = "Data sample - Dispute classification Testing.xlsx"   # test file path
TEXT_COL = "Input Feature"
GT_COL = "Correct Classification"  # ground truth column in the test file

OUT_DIR = Path(r"C:\Users\mathu\ML Model\Development1\DisputeClassificationModel")
METRICS_XLSX = OUT_DIR / "metrics_RoBERTabert.xlsx"
PREDICTIONS_XLSX = OUT_DIR / "Dispute Classification testing with predictionsRoBERTaBERTEvaluation.xlsx"

# Confusion matrix images
CM_PNG = OUT_DIR / "confusion_matrix_roberta.png"
CM_NORM_PNG = OUT_DIR / "confusion_matrix_roberta_normalized.png"

BATCH_SIZE = 32
MAX_LEN = 512

# =========================
# Helpers
# =========================
def save_excel_safely(df: pd.DataFrame, out_path: Path, **to_excel_kwargs):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_excel(out_path, index=False, **to_excel_kwargs)
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fb = out_path.with_name(out_path.stem + f"_{ts}" + out_path.suffix)
        print(f"⚠️ '{out_path.name}' is locked. Writing to '{fb.name}' instead.")
        df.to_excel(fb, index=False, **to_excel_kwargs)

def autofit_columns(ws, max_width=80):
    for col in ws.columns:
        try:
            col_letter = col[0].column_letter
        except Exception:
            continue
        max_len = 0
        for cell in col:
            v = "" if cell.value is None else str(cell.value)
            max_len = max(max_len, len(v))
        ws.column_dimensions[col_letter].width = min(max_len + 2, max_width)

def write_metrics_workbook(pred_df, overall_df, per_class_df_with_acc, cm_df, out_path: Path,
                           cm_png: Path | None = None, cm_norm_png: Path | None = None):
    """Write tables + predictions to Excel and embed CM images."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pred_df.to_excel(writer, sheet_name="predictions", index=False)
        overall_df.to_excel(writer, sheet_name="overall_metrics", index=False)
        per_class_df_with_acc.to_excel(writer, sheet_name="per_class_report_with_accuracy", index=False)
        cm_df.to_excel(writer, sheet_name="confusion_matrix", index=True)

        # Image sheets
        if cm_png is not None and cm_png.exists():
            ws_img = writer.book.create_sheet("confusion_matrix_image")
            ws_img.add_image(XLImage(str(cm_png)), "A1")
        if cm_norm_png is not None and cm_norm_png.exists():
            ws_imgn = writer.book.create_sheet("confusion_matrix_image_norm")
            ws_imgn.add_image(XLImage(str(cm_norm_png)), "A1")

        # Auto-fit columns for tabular sheets
        for name in ["predictions", "overall_metrics", "per_class_report_with_accuracy", "confusion_matrix"]:
            ws = writer.book[name]
            autofit_columns(ws)

# =========================
# Load model & tokenizer
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()
id2label = model.config.id2label

# =========================
# Load test data
# =========================
test_df = pd.read_excel(TEST_XLSX)
if TEXT_COL not in test_df.columns:
    raise ValueError(f"Column '{TEXT_COL}' not found in {TEST_XLSX}")
texts = test_df[TEXT_COL].astype(str).fillna("").tolist()

# =========================
# Batched prediction
# =========================
def predict_batch(texts, batch_size=BATCH_SIZE, max_length=MAX_LEN):
    all_pred_ids, all_probs = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            pred_ids = probs.argmax(dim=1)
        all_pred_ids.extend(pred_ids.cpu().numpy().tolist())
        all_probs.append(probs.cpu().numpy())
    probs_all = np.vstack(all_probs) if all_probs else np.zeros((0, model.config.num_labels))
    return all_pred_ids, probs_all

pred_ids, y_scores = predict_batch(texts)
pred_labels = [id2label[i] for i in pred_ids]
test_df["PredictionRoBERTaBERT"] = pred_labels

# (Optional) also save a standalone predictions file
save_excel_safely(
    test_df[[TEXT_COL, *( [GT_COL] if GT_COL in test_df.columns else [] ), "PredictionRoBERTaBERT"]],
    PREDICTIONS_XLSX
)

# =========================
# Build the tables + CM images
# =========================
if GT_COL in test_df.columns:
    true_labels = test_df[GT_COL].astype(str).fillna("").tolist()

    # ---- overall_metrics
    overall = {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "precision_macro": precision_score(true_labels, pred_labels, average='macro', zero_division=0),
        "recall_macro": recall_score(true_labels, pred_labels, average='macro', zero_division=0),
        "f1_macro": f1_score(true_labels, pred_labels, average='macro', zero_division=0),
        "precision_weighted": precision_score(true_labels, pred_labels, average='weighted', zero_division=0),
        "recall_weighted": recall_score(true_labels, pred_labels, average='weighted', zero_division=0),
        "f1_weighted": f1_score(true_labels, pred_labels, average='weighted', zero_division=0),
    }
    overall_df = pd.DataFrame({"metric": list(overall.keys()), "value": list(overall.values())})

    # ---- confusion_matrix (numeric table)
    labels_sorted = sorted(set(true_labels) | set(pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=labels_sorted)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true:{l}" for l in labels_sorted],
        columns=[f"pred:{l}" for l in labels_sorted]
    )

    # ---- render & save CM images
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Raw counts
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_sorted)\
        .plot(include_values=True, cmap="Greens", ax=ax, colorbar=True, xticks_rotation=45)
    ax.set_title("Confusion Matrix (counts)")
    plt.tight_layout()
    fig.savefig(CM_PNG, dpi=200)
    plt.close(fig)

    # Row-normalized
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_norm /= row_sums

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels_sorted)\
        .plot(include_values=True, cmap="Greens", ax=ax2, colorbar=True, xticks_rotation=45)
    ax2.set_title("Confusion Matrix - RoBERTa (row-normalized)")
    plt.tight_layout()
    fig2.savefig(CM_NORM_PNG, dpi=200)
    plt.close(fig2)

    # ---- per_class_report_with_accuracy
    report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
    per_class_rows = {k: v for k, v in report.items() if k in labels_sorted}
    per_class_df = (pd.DataFrame(per_class_rows).T
                    .reset_index()
                    .rename(columns={"index": "label", "f1-score": "f1"}))
    per_class_df = per_class_df[["label", "precision", "recall", "f1", "support"]]

    # per-class accuracy = TP/(TP+FN)
    acc_rows = []
    for i, cls in enumerate(labels_sorted):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        acc_rows.append({"label": cls, "accuracy": (TP / (TP + FN) if (TP + FN) > 0 else 0.0)})
    per_class_acc_df = pd.DataFrame(acc_rows)

    per_class_with_acc = per_class_df.merge(per_class_acc_df, on="label", how="left")

    # ---- Write one Excel with tables + predictions + images
    pred_sheet = test_df[[TEXT_COL, GT_COL, "PredictionRoBERTaBERT"]]
    write_metrics_workbook(
        pred_df=pred_sheet,
        overall_df=overall_df,
        per_class_df_with_acc=per_class_with_acc,
        cm_df=cm_df,
        out_path=METRICS_XLSX,
        cm_png=CM_PNG,
        cm_norm_png=CM_NORM_PNG
    )

    print("\n Wrote the following to Excel:")
    print(" - overall_metrics")
    print(" - confusion_matrix (numeric table)")
    print(" - per_class_report_with_accuracy (incl. per-class accuracy)")
    print(" - predictions")
    print(f" - embedded images: {CM_PNG.name}, {CM_NORM_PNG.name}")
    print(f"→ {METRICS_XLSX}\n")

else:
    print(f" Ground-truth column '{GT_COL}' not found; only predictions were saved to {PREDICTIONS_XLSX}.")
