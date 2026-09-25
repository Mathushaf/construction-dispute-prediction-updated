# DisputeIndex Model - RoBERTa-base Fine-Tuning
# WITH Validation + Logging
# Model Name: DisputeIndex_RoBERTa_Base
# ====================================================

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from tqdm.auto import tqdm

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -----------------------------
# Load dataset
# -----------------------------
train_path = r"Data Sample - Dispute Index Training.xlsx"
df = pd.read_excel(train_path)

required_cols = ["Input Feature", "Target Feature"]
df = df.dropna(subset=required_cols)

df["Target Feature"] = df["Target Feature"].astype(int)
df["label_id"] = df["Target Feature"] - 1

print("Data loaded")

# -----------------------------
# Label mapping
# -----------------------------
label2id = {str(i): i - 1 for i in range(1, 6)}
id2label = {i - 1: str(i) for i in range(1, 6)}

# -----------------------------
# Load tokenizer & model
# -----------------------------
model_name = "roberta-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=5,
    id2label=id2label,
    label2id=label2id
)

# -----------------------------
# Dataset class
# -----------------------------
class DisputeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# -----------------------------
# Prepare dataset
# -----------------------------
texts = df["Input Feature"].astype(str).tolist()
labels = df["label_id"].tolist()

full_dataset = DisputeDataset(texts, labels, tokenizer)

#  Train (80%) + Validation (20%)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

print(f" Train size: {train_size}, Validation size: {val_size}")

# -----------------------------
# Training setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

num_epochs = 5
total_steps = len(train_loader) * num_epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# -----------------------------
# Training + Validation Loop
# -----------------------------
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):

    # ===== TRAIN =====
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)

        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)

    avg_train_loss = total_loss / len(train_loader)
    train_acc = correct / total

    # ===== VALIDATION =====
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            val_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            val_correct += (preds == batch["labels"]).sum().item()
            val_total += batch["labels"].size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total

    # Store logs
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.4f}")

# -----------------------------
# Save logs
# -----------------------------
log_df = pd.DataFrame({
    "epoch": range(1, num_epochs + 1),
    "train_loss": train_losses,
    "val_loss": val_losses,
    "train_accuracy": train_accuracies,
    "val_accuracy": val_accuracies
})

log_df.to_excel("roberta_training_logs.xlsx", index=False)
print("Training logs saved!")

# -----------------------------
# Save model
# -----------------------------
save_dir = os.path.join(os.getcwd(), "DisputeIndex_RoBERTa_Base")
os.makedirs(save_dir, exist_ok=True)

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

torch.save(
    {"label2id": label2id, "id2label": id2label},
    os.path.join(save_dir, "label_mappings.pt")
)

print(f"Model saved to: {save_dir}")
