# ====================================================
# DisputeType Model - RoBERTa (Train + Validation + Logging)
# ====================================================

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os

# -----------------------------
# Config
# -----------------------------
DATA_PATH = r"Data Sample - Dispute Classification training.xlsx"
MODEL_NAME = "roberta-base"
SAVE_DIR = "./dispute_classification_roberta"
BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-5
MAX_LEN = 512
TEST_SIZE = 0.2
SEED = 42

torch.manual_seed(SEED)

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_excel(DATA_PATH).dropna(subset=['Input Feature', 'Target Feature'])

# -----------------------------
# Label encoding (STABLE)
# -----------------------------
unique_labels = sorted(df['Target Feature'].astype(str).unique())
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

df['label_id'] = df['Target Feature'].astype(str).map(label2id).astype(int)

print(" Labels:", label2id)

# -----------------------------
# Train/validation split
# -----------------------------
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['Input Feature'],
    df['label_id'],
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=df['label_id']
)

# -----------------------------
# Dataset class
# -----------------------------
class DisputeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# -----------------------------
# Loaders
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_loader = DataLoader(
    DisputeDataset(train_texts, train_labels, tokenizer, MAX_LEN),
    batch_size=BATCH_SIZE, shuffle=True
)

val_loader = DataLoader(
    DisputeDataset(val_texts, val_labels, tokenizer, MAX_LEN),
    batch_size=BATCH_SIZE
)

# -----------------------------
# Model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_loader) * EPOCHS
)

# -----------------------------
#  Track logs
# -----------------------------
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# -----------------------------
# Training loop
# -----------------------------
for epoch in range(EPOCHS):

    # ===== TRAIN =====
    model.train()
    total_loss = 0
    train_preds, train_true = [], []

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
        optimizer.zero_grad()

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_true.extend(batch["labels"].cpu().numpy())

    avg_train_loss = total_loss / len(train_loader)
    train_acc = accuracy_score(train_true, train_preds)

    # ===== VALIDATION =====
    model.eval()
    val_loss = 0
    val_preds, val_true = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            val_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_true.extend(batch["labels"].cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_acc = accuracy_score(val_true, val_preds)

    # Save logs
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f}")
    print(f"Val   Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f}")

# -----------------------------
# Save logs
# -----------------------------
log_df = pd.DataFrame({
    "epoch": range(1, EPOCHS + 1),
    "train_loss": train_losses,
    "val_loss": val_losses,
    "train_accuracy": train_accuracies,
    "val_accuracy": val_accuracies
})

log_df.to_excel("DisputeType_RoBERTa_training_logs.xlsx", index=False)

print(" Training logs saved!")

# -----------------------------
# Save model
# -----------------------------
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f" Model saved to: {SAVE_DIR}")
