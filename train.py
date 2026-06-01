import os
import json
import math
import pickle
import time

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)

from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

np.random.seed(42)
torch.manual_seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — pick ONE base model:
#
#   FAST  (~30-60 min CPU / ~5 min GPU):
#       "cross-encoder/ms-marco-MiniLM-L-6-v2"   ← default
#
#   BEST  (~2-4 hr CPU / ~15 min GPU):
#       "cross-encoder/quora-roberta-base"
#
# If you have a GPU, switch to quora-roberta-base — it starts closer to the
# task and can reach slightly higher accuracy.
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SAVE_DIR = "models/cross_encoder"
BATCH_SIZE = 16
NUM_EPOCHS = 1
WARMUP_FRAC = 0.1
MAX_LEN = 128


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def find_best_threshold(y_true, y_prob):
    best_threshold = 0.5
    best_f1 = 0.0
    best_acc = 0.0

    for t in np.arange(0.25, 0.76, 0.01):
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(round(t, 2))
            best_acc = accuracy_score(y_true, preds)

    return best_threshold, best_f1, best_acc


def train():
    # ── Load ─────────────────────────────────────────────────────────────────
    print("Loading dataset...")
    df = pd.read_csv(
        "data/train_snli.txt",
        sep="\t",
        header=None,
        names=["text1", "text2", "label"],
    )

    df = df.dropna()
    df["label"] = df["label"].astype(int)

    if len(df) > 5000:
        df = df.sample(n=5000, random_state=42)

    print(f"Dataset size : {len(df)}")
    print(f"Label balance: {df['label'].value_counts().to_dict()}")

    # ── Split ─────────────────────────────────────────────────────────────────
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"]
    )

    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    # ── TF-IDF vectorizer (display features only) ─────────────────────────────
    print("\nFitting TF-IDF vectorizer for display features...")
    tfidf_vec = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=20000,
        min_df=2,
        max_df=0.9,
    )
    tfidf_vec.fit(df["text1"].tolist() + df["text2"].tolist())

    os.makedirs("models", exist_ok=True)
    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf_vec, f)
    print("✔ Saved → models/tfidf_vectorizer.pkl")

    # ── Build InputExamples ───────────────────────────────────────────────────
    train_samples = [
        InputExample(texts=[row.text1, row.text2], label=float(row.label))
        for row in train_df.itertuples()
    ]
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=BATCH_SIZE)

    # ── Load CrossEncoder ─────────────────────────────────────────────────────
    print(f"\nLoading base model : {MODEL_NAME}")
    model = CrossEncoder(MODEL_NAME, num_labels=1, max_length=MAX_LEN)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training device    : {device.upper()}")
    if device == "cpu":
        print("  (tip: switch to quora-roberta-base if you have a GPU for best accuracy)")

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    warmup_steps = math.ceil(len(train_dataloader) * NUM_EPOCHS * WARMUP_FRAC)
    print(f"\nEpochs: {NUM_EPOCHS}  |  Batch: {BATCH_SIZE}  |  Warmup: {warmup_steps}\n")

    model.fit(
        train_dataloader=train_dataloader,
        epochs=NUM_EPOCHS,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
    )

    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save(SAVE_DIR)
    print(f"\n✔ Fine-tuned model saved → {SAVE_DIR}")

    # ── Threshold tuning on validation ────────────────────────────────────────
    print("\nEvaluating on validation set...")
    val_pairs = [(row.text1, row.text2) for row in val_df.itertuples()]
    val_scores = model.predict(val_pairs, show_progress_bar=True)
    val_prob = _sigmoid(np.array(val_scores, dtype=float))
    y_val = val_df["label"].values

    best_threshold, best_f1, best_acc = find_best_threshold(y_val, val_prob)
    val_pred = (val_prob >= best_threshold).astype(int)

    print(f"\nValidation results:")
    print(f"  Best threshold : {best_threshold}  (optimised on F1)")
    print(f"  Val F1         : {round(best_f1, 4)}")
    print(f"  Val Accuracy   : {round(best_acc, 4)}")
    print(f"  Val AUC        : {round(roc_auc_score(y_val, val_prob), 4)}")
    print("\nValidation Classification Report:")
    print(classification_report(y_val, val_pred))

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("Evaluating on test set...")
    test_pairs = [(row.text1, row.text2) for row in test_df.itertuples()]

    test_scores = model.predict(test_pairs, show_progress_bar=True)
    test_prob = _sigmoid(np.array(test_scores, dtype=float))
    y_test = test_df["label"].values
    test_pred = (test_prob >= best_threshold).astype(int)

    print(f"\nTest results:")
    print(f"  Test Accuracy : {round(accuracy_score(y_test, test_pred), 4)}")
    print(f"  Test F1       : {round(f1_score(y_test, test_pred), 4)}")
    print(f"  Test AUC      : {round(roc_auc_score(y_test, test_prob), 4)}")
    print("\nTest Classification Report:")
    print(classification_report(y_test, test_pred))

    # ── Save config ───────────────────────────────────────────────────────────
    config = {
        "best_threshold": best_threshold,
        "base_model": MODEL_NAME,
        "num_epochs": NUM_EPOCHS,
        "val_f1": round(best_f1, 4),
        "val_auc": round(roc_auc_score(y_val, val_prob), 4),
        "validation_report": classification_report(y_val, val_pred),
        "test_accuracy": round(accuracy_score(y_test, test_pred), 4),
        "test_f1": round(f1_score(y_test, test_pred), 4),
        "test_auc": round(roc_auc_score(y_test, test_prob), 4),
        "test_report": classification_report(y_test, test_pred),
    }

    with open(os.path.join(SAVE_DIR, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ── Save prediction arrays for plots ─────────────────────────────────────
    np.save("models/y_test.npy", y_test)
    np.save("models/test_pred.npy", test_pred)
    np.save("models/test_prob.npy", test_prob)

    # ── Confusion matrix + ROC curve ─────────────────────────────────────────
    os.makedirs("models/plots", exist_ok=True)

    cm = confusion_matrix(y_test, test_pred)
    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Plagiarized", "Plagiarized"],
    )
    disp.plot(cmap="Blues", values_format="d", ax=plt.gca())
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("models/plots/confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, test_prob)
    roc_auc_value = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc_value:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("models/plots/roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\n✔ Confusion Matrix saved → models/plots/confusion_matrix.png")
    print("✔ ROC Curve saved → models/plots/roc_curve.png")
    print(f"\n✔ Done.  Model → {SAVE_DIR}  |  Threshold → {best_threshold}")


if __name__ == "__main__":
    train()


