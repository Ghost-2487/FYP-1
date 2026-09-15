import pandas as pd
import numpy as np
import re
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, precision_recall_curve, auc,
                             confusion_matrix, classification_report)
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Data Loading & Cleaning
df1_path = BASE_DIR / "data" / "Suicide_Detection.csv"
df2_path = BASE_DIR / "data" / "suicide_only_clean.csv"

def load_and_standardize_data(path1, path2):
    if not os.path.exists(path1) or not os.path.exists(path2):
        print(f"❌ Error: One or both files not found.\nPath 1: {path1}\nPath 2: {path2}")
        print(f"👉 Run extract_suicide_data.py first.")
        return None
    
    print("Loading Suicide_Detection.csv (DF1)...")
    df1 = pd.read_csv(path1)
    df1 = df1.rename(columns={'class': 'label'})
    df1['label'] = df1['label'].map({'suicide': 1, 'non-suicide': 0})
    df1 = df1[['text', 'label']]

    print("Loading suicide_only_clean.csv (DF2)...")
    df2 = pd.read_csv(path2)
    df2 = df2[['text']]
    df2['label'] = 1 

    df = pd.concat([df1, df2], ignore_index=True)
    df = df.dropna(subset=['text', 'label'])
    return df

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df = load_and_standardize_data(df1_path, df2_path)
if df is not None:
    # USE ALL DATA: Shuffle but use 100%
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"Cleaning {len(df)} rows...")
    df['clean_text'] = df['text'].apply(clean_text)
    print(f"Dataset size: {len(df)}")

# 2. Configuration & Tokenization
BERT_MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 8  # Reduced from 16 for memory efficiency
EPOCHS = 3
LR = 2e-5

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

print("Splitting dataset...")
X_train_txt, X_test_txt, y_train, y_test = train_test_split(
    df['clean_text'].tolist(), df['label'].tolist(),
    test_size=0.2, random_state=42, stratify=df['label']
)

print(f"Train set size: {len(X_train_txt)} | Test set size: {len(X_test_txt)}")

# 3. Dataset & DataLoader (Tokenization on-the-fly)
class OptimizedDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text, max_length=self.max_len, padding='max_length',
            truncation=True, return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label)
        }

train_dataset = OptimizedDataset(X_train_txt, y_train, tokenizer, MAX_LEN)
test_dataset = OptimizedDataset(X_test_txt, y_test, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("DataLoaders ready! Training will tokenize batches on-the-fly.")

# 4. Model & Training Loop
print(f"Loading {BERT_MODEL_NAME} for fine-tuning...")
model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=2).to(device)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scaler = GradScaler()

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
)

def train_epoch(model, loader, optimizer, scheduler, device, scaler):
    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc="Training", leave=False)

    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with autocast(device_type=device.type):
            outputs = model(input_ids, attention_mask=mask, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    return total_loss / len(loader)

print("Starting Training on Full Dataset...")
history = {'train_loss': []}

for epoch in range(1, EPOCHS + 1):
    avg_loss = train_epoch(model, train_loader, optimizer, scheduler, device, scaler)
    history['train_loss'].append(round(avg_loss, 4))
    print(f"Epoch {epoch}/{EPOCHS} | Avg Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), SCRIPT_DIR / "bert_base_full_dataset.pt")

# 5. Evaluation
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=mask)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return all_preds, all_labels, all_probs

y_pred, y_true, y_probs = evaluate(model, test_loader, device)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Non-Suicide', 'Suicide'], digits=4))

# ── Confusion Matrix (%) ──────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred, normalize='true')
plt.figure(figsize=(8, 6))
sns.heatmap(cm * 100, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=['Non-Suicide', 'Suicide'],
            yticklabels=['Non-Suicide', 'Suicide'])
plt.title('BERT Confusion Matrix - Suicide Detection (%)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'bert_suicide_confusion_matrix.png', dpi=150)
plt.show()
print('✅ Confusion matrix saved → bert_suicide_confusion_matrix.png')

# ── Training History ──────────────────────────────────────────────────────────
plt.figure(figsize=(6, 5))
plt.plot(history['train_loss'], label='Train Loss', color='steelblue')
plt.title('Training Loss - BERT Suicide')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'bert_suicide_training_history.png', dpi=150)
plt.show()
print('✅ Training history saved → bert_suicide_training_history.png')

# ── ROC Curve ─────────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_true, y_probs)
auc_roc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC curve (AUC = {auc_roc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - BERT Suicide')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'bert_suicide_roc_curve.png', dpi=150)
plt.show()
print('✅ ROC curve saved → bert_suicide_roc_curve.png')

# ── Precision-Recall Curve ────────────────────────────────────────────────────
precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_probs)
auc_pr = auc(recall_arr, precision_arr)

plt.figure(figsize=(6, 5))
plt.plot(recall_arr, precision_arr, color='darkorange', lw=2,
         label=f'PR curve (AUC = {auc_pr:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - BERT Suicide')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'bert_suicide_pr_curve.png', dpi=150)
plt.show()
print('✅ PR curve saved → bert_suicide_pr_curve.png')

# ── Dashboard JSON ────────────────────────────────────────────────────────────
report_dict = classification_report(
    y_true, y_pred,
    target_names=['Non-Suicide', 'Suicide'],
    output_dict=True
)

cm_raw = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm_raw.ravel()

roc_indices = np.linspace(0, len(fpr) - 1, 12, dtype=int)
roc_points  = [[round(float(fpr[i]), 3), round(float(tpr[i]), 3)] for i in roc_indices]

pr_indices = np.linspace(0, len(precision_arr) - 1, 12, dtype=int)
pr_points  = [[round(float(recall_arr[i]), 3), round(float(precision_arr[i]), 3)] for i in pr_indices]

token_lengths = [len(str(t).split()) for t in df['clean_text']]

dashboard = {
    'model'    : 'BERT',
    'dataset'  : 'Suicide Detection',
    'accuracy' : round(report_dict['accuracy'] * 100, 2),
    'precision': round(report_dict['weighted avg']['precision'] * 100, 2),
    'recall'   : round(report_dict['weighted avg']['recall'] * 100, 2),
    'f1'       : round(report_dict['weighted avg']['f1-score'] * 100, 2),
    'auc_roc'  : round(float(auc_roc), 4),
    'auc_pr'   : round(float(auc_pr), 4),
    'class_metrics': {
        'Non-Suicide': {k: round(v * 100, 2) for k, v in report_dict['Non-Suicide'].items() if k != 'support'},
        'Suicide'    : {k: round(v * 100, 2) for k, v in report_dict['Suicide'].items()     if k != 'support'},
    },
    'confusion_matrix': {
        'tp': round(tp / (tp + fn) * 100, 2),
        'fp': round(fp / (fp + tn) * 100, 2),
        'fn': round(fn / (tp + fn) * 100, 2),
        'tn': round(tn / (fp + tn) * 100, 2),
    },
    'roc_curve'       : roc_points,
    'pr_curve'        : pr_points,
    'training_history': {
        'train_loss': history['train_loss'],
    },
    'dataset_stats': {
        'total_samples'  : len(df),
        'train_samples'  : len(X_train_txt),
        'test_samples'   : len(X_test_txt),
        'avg_text_length': round(float(np.mean(token_lengths)), 1),
        'max_text_length': int(np.max(token_lengths)),
        'min_text_length': int(np.min(token_lengths)),
    },
}

with open(SCRIPT_DIR / 'bert_suicide_dashboard.json', 'w') as f:
    json.dump(dashboard, f, indent=2)
print('✅ Dashboard JSON saved → bert_suicide_dashboard.json')