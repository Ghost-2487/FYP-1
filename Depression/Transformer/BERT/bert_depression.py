import os
import re
import html
import json
import random
import warnings
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import BertTokenizerFast, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
)
from pathlib import Path

warnings.filterwarnings('ignore')
print('✅ Imports complete.')

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

CFG = {
    'data_path'    : BASE_DIR / "data" / "training.1600000.processed.noemoticon.csv",
    'model_name'   : 'bert-base-uncased',
    'max_len'      : 48,
    'val_size'     : 0.05,
    'test_size'    : 0.10,
    'batch_size'   : 128,
    'num_workers'  : 0,
    'dropout_rate' : 0.1,
    'weight_decay' : 0.01,
    'grad_clip'    : 1.0,
    'warmup_ratio' : 0.06,
    'use_amp'      : True,
    'epochs'       : 5,
    'lr'           : 2e-5,
    'es_patience'  : 2,
    'save_path'    : SCRIPT_DIR / "best_bert_depression.pt",
    'json_out'     : SCRIPT_DIR / "bert_sentiment140_dashboard.json",
    'seed'         : 42,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(CFG['seed'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'✅ Seed set. Using device: {device}')

CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "i'm": "i am",
    "ain't": "is not", "i've": "i have", "i'd": "i would",
    "i'll": "i will", "it's": "it is", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "couldn't": "could not", "shouldn't": "should not", "wouldn't": "would not",
    "mustn't": "must not", "needn't": "need not", "that's": "that is",
    "there's": "there is", "they're": "they are", "they've": "they have",
    "they'd": "they would", "they'll": "they will", "we're": "we are",
    "we've": "we have", "we'd": "we would", "we'll": "we will",
    "you're": "you are", "you've": "you have", "you'd": "you would",
    "you'll": "you will", "he's": "he is", "she's": "she is",
    "he'd": "he would", "she'd": "she would", "he'll": "he will",
    "she'll": "she will", "let's": "let us", "what's": "what is",
    "who's": "who is", "where's": "where is", "how's": "how is",
    "it'd": "it would", "it'll": "it will",
}

_C_PAT = re.compile(r'\b(' + '|'.join(re.escape(k) for k in sorted(CONTRACTIONS, key=len, reverse=True)) + r')\b', flags=re.IGNORECASE)


# Normalise a raw tweet for BERT tokenisation.
def clean_text_for_bert(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ''
    t = html.unescape(text).lower()
    t = re.sub(r'https?://\S+|www\.\S+', ' [URL] ', t)
    t = re.sub(r'@\w+', ' [USER] ', t)
    t = _C_PAT.sub(
        lambda m: CONTRACTIONS.get(m.group(0).lower(), m.group(0)), t
    )
    return ' '.join(t.split())


print('✅ Preprocessing functions defined.')

def load_and_split_data(cfg: dict) -> tuple:
    """
    Load Sentiment140 dataset, clean, stratified-split, and cache to disk.
    Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    cache_files = [BASE_DIR / "data" / f for f in ['m_train.pkl', 'm_val.pkl', 'm_test.pkl']]

    if all(os.path.exists(f) for f in cache_files):
        print('✅ Loading cached splits...')
        with open(cache_files[0], 'rb') as f: X_train, y_train = pickle.load(f)
        with open(cache_files[1], 'rb') as f: X_val,   y_val   = pickle.load(f)
        with open(cache_files[2], 'rb') as f: X_test,  y_test  = pickle.load(f)
        return X_train, X_val, X_test, y_train, y_val, y_test

    print('⚙️  Processing raw data from scratch...')
    df = pd.read_csv(
        cfg['data_path'], encoding='latin-1', header=None,
        names=['sentiment', 'id', 'date', 'query', 'user_id', 'text'],
    )
    df = (
        df[df['sentiment'].isin([0, 4])][['sentiment', 'text']]
        .dropna()
        .drop_duplicates(subset=['text'])
    )
    df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})
    df['text']      = df['text'].apply(clean_text_for_bert)
    df = df[df['text'].str.strip() != ''].reset_index(drop=True)

    # Quick class-balance sanity check
    print('Class distribution:', df['sentiment'].value_counts().to_dict())

    texts, labels = df['text'].tolist(), df['sentiment'].tolist()
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        texts, labels,
        test_size=cfg['test_size'],
        random_state=cfg['seed'],
        stratify=labels,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp,
        test_size=cfg['val_size'] / (1 - cfg['test_size']),
        random_state=cfg['seed'],
        stratify=y_tmp,
    )

    with open(cache_files[0], 'wb') as f: pickle.dump((X_train, y_train), f)
    with open(cache_files[1], 'wb') as f: pickle.dump((X_val,   y_val),   f)
    with open(cache_files[2], 'wb') as f: pickle.dump((X_test,  y_test),  f)
    print('✅ Data cached to disk.')

    return X_train, X_val, X_test, y_train, y_val, y_test


X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(CFG)
print(f'   Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}')

# Tokenizer lives here — tightly coupled to TweetDataset
tokenizer = BertTokenizerFast.from_pretrained(CFG['model_name'])
tokenizer.add_tokens(['[URL]', '[USER]'], special_tokens=True)

# Maps raw tweet strings → BERT input tensors.
class TweetDataset(Dataset):

    def __init__(self, texts: list, labels: list):
        self.texts  = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = tokenizer(
            self.texts[idx],
            max_length=CFG['max_len'],
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids'     : enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label'         : torch.tensor(self.labels[idx], dtype=torch.long),
        }


# Wrap a TweetDataset in a DataLoader with project-wide settings.
def create_loader(texts: list, labels: list, shuffle: bool) -> DataLoader:
    return DataLoader(
        TweetDataset(texts, labels),
        batch_size=CFG['batch_size'],
        shuffle=shuffle,
        num_workers=CFG['num_workers'],
        pin_memory=True,
    )


train_loader = create_loader(X_train, y_train, shuffle=True)
val_loader   = create_loader(X_val,   y_val,   shuffle=False)
test_loader  = create_loader(X_test,  y_test,  shuffle=False)
print('✅ DataLoaders ready.')

class Bert(nn.Module):
    """
    BERT + dropout + linear classifier.
    Uses [CLS] pooler output for classification.
    """

    def __init__(self):
        super().__init__()
        self.bert    = BertModel.from_pretrained(CFG['model_name'])
        self.bert.resize_token_embeddings(len(tokenizer))
        self.dropout = nn.Dropout(CFG['dropout_rate'])
        self.fc      = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.fc(self.dropout(out.pooler_output))


print('✅ Bert architecture defined.')

# Saves the best checkpoint; signals stop when patience is exhausted.
class EarlyStopping:

    def __init__(self, patience: int = 2, path: str = 'best.pt'):
        self.patience  = patience
        self.path      = path
        self.best_loss = None
        self.counter   = 0

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.path)
            return False          # continue training
        self.counter += 1
        return self.counter >= self.patience   # True → stop


# One full pass over the training set. Returns (avg_loss, accuracy).
def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, scheduler, scaler: GradScaler, criterion: nn.Module) -> tuple:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in tqdm(loader, leave=False, desc='Training'):
        ids  = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        lbls = batch['label'].to(device)

        optimizer.zero_grad()
        with autocast(device_type='cuda', enabled=CFG['use_amp']):
            logits = model(ids, mask)
            loss   = criterion(logits, lbls)

        if CFG['use_amp']:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        correct    += (logits.argmax(dim=1) == lbls).sum().item()
        total      += lbls.size(0)

    return total_loss / len(loader), correct / total


# Evaluate on any loader. Returns (avg_loss, accuracy).
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for batch in tqdm(loader, leave=False, desc='Evaluating'):
        ids  = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        lbls = batch['label'].to(device)

        with autocast(device_type='cuda', enabled=CFG['use_amp']):
            logits = model(ids, mask)
            loss   = criterion(logits, lbls)

        total_loss += loss.item()
        correct    += (logits.argmax(dim=1) == lbls).sum().item()
        total      += lbls.size(0)

    return total_loss / len(loader), correct / total


print('✅ Training utilities defined.')

# Initialise model, run fine-tuning, return (model, history).
def run_training(cfg: dict) -> tuple:
    """
    The best checkpoint is saved to cfg['save_path'].
    """
    print(f"\n{'='*50}\n🚀 Starting Fine-Tuning\n{'='*50}")

    model     = Bert().to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'⚙️  Trainable params: {trainable:,}')

    optimizer    = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    total_steps  = len(train_loader) * cfg['epochs']
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * cfg['warmup_ratio']),
        num_training_steps=total_steps,
    )
    criterion    = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler       = GradScaler('cuda') if cfg['use_amp'] else None
    early_stop   = EarlyStopping(patience=cfg['es_patience'], path=cfg['save_path'])

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for ep in range(1, cfg['epochs'] + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, criterion)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        print(
            f"Epoch {ep}/{cfg['epochs']} | "
            f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f} | "
            f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f} "
            f"[{time.time()-t0:.0f}s]"
        )

        if early_stop(vl_loss, model):
            print('⏹️  Early stopping triggered.')
            break

    print(f'\n🎉 Training complete. Best checkpoint → {cfg["save_path"]}')
    model.load_state_dict(torch.load(cfg['save_path'], weights_only=True))
    return model, history


print('✅ Training loop defined.')

# Single inference pass over loader.
@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader) -> tuple:
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for batch in tqdm(loader, desc='Testing'):
        ids    = batch['input_ids'].to(device)
        mask   = batch['attention_mask'].to(device)
        logits = model(ids, mask)
        probs  = torch.softmax(logits, dim=1)[:, 1]

        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(batch['label'].numpy())
        all_probs.extend(probs.cpu().numpy())

    return all_preds, all_labels, all_probs


def plot_confusion_matrix(labels: list, preds: list) -> None:
    cm = confusion_matrix(labels, preds, normalize='true')
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm * 100, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('BERT Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / "bert_confusion_matrix.png", dpi=150)
    plt.show()


def plot_training_history(history: dict) -> None:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'],   label='Val')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'],   label='Val')
    plt.title('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / "bert_training_history.png", dpi=150)
    plt.show()

def plot_roc_curve(labels: list, probs: list) -> None:
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('BERT ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / "bert_roc_curve.png", dpi=150)
    plt.show()

def plot_pr_curve(labels: list, probs: list) -> None:
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('BERT Precision-Recall Curve')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / "bert_pr_curve.png", dpi=150)
    plt.show()

def save_dashboard_json(report: dict, labels: list, preds: list, probs: list, history: dict, cfg: dict) -> None:
    cm                  = confusion_matrix(labels, preds)
    tn, fp, fn, tp      = cm.ravel()

    # ROC
    fpr_arr, tpr_arr, _ = roc_curve(labels, probs)
    auc_roc             = auc(fpr_arr, tpr_arr)
    roc_indices         = np.linspace(0, len(fpr_arr) - 1, 12, dtype=int)
    roc_points          = [[round(float(fpr_arr[i]), 3), round(float(tpr_arr[i]), 3)] for i in roc_indices]

    # PR
    prec_arr, rec_arr, _ = precision_recall_curve(labels, probs)
    auc_pr               = auc(rec_arr, prec_arr)
    pr_indices           = np.linspace(0, len(prec_arr) - 1, 12, dtype=int)
    pr_points            = [[round(float(rec_arr[i]), 3), round(float(prec_arr[i]), 3)] for i in pr_indices]

    all_texts       = X_train + X_val + X_test
    token_lengths   = [len(t.split()) for t in all_texts]

    dashboard = {
        'model'     : 'BERT',
        'dataset'   : 'Sentiment140',
        'accuracy'  : round(report['accuracy'] * 100, 2),
        'precision' : round(report['weighted avg']['precision'] * 100, 2),
        'recall'    : round(report['weighted avg']['recall'] * 100, 2),
        'f1'        : round(report['weighted avg']['f1-score'] * 100, 2),
        'auc_roc'   : round(float(auc_roc), 4),
        'auc_pr'    : round(float(auc_pr), 4),
        'class_metrics': {
            'Negative' : {k: round(v * 100, 2) for k, v in report['Negative'].items() if k != 'support'},
            'Positive' : {k: round(v * 100, 2) for k, v in report['Positive'].items() if k != 'support'},
        },
        'confusion_matrix': {
            'tp': round(tp / (tp + fn) * 100, 2),
            'fp': round(fp / (fp + tn) * 100, 2),
            'fn': round(fn / (tp + fn) * 100, 2),
            'tn': round(tn / (fp + tn) * 100, 2),
        },
        'roc_curve'       : roc_points,
        'pr_curve'        : pr_points,
        'training_history': history,
        'dataset_stats': {
            'total_samples'  : len(all_texts),
            'train_samples'  : len(X_train),
            'val_samples'    : len(X_val),
            'test_samples'   : len(X_test),
            'avg_text_length': round(float(np.mean(token_lengths)), 1),
            'max_text_length': int(np.max(token_lengths)),
            'min_text_length': int(np.min(token_lengths)),
        },
    }

    with open(cfg['json_out'], 'w') as f:
        json.dump(dashboard, f, indent=2)
    print(f'✅ Dashboard JSON saved → {cfg["json_out"].name}')

# Full evaluation on the held-out test set: metrics, plots, JSON.
def run_evaluation(model: nn.Module, history: dict, cfg: dict) -> None:
    print(f"\n{'='*50}\n📊 Evaluating on Test Set\n{'='*50}")

    preds, labels, probs = collect_predictions(model, test_loader)

    acc    = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    report = classification_report(
        labels, preds,
        target_names=['Negative', 'Positive'],
        digits=4,
        output_dict=True,
    )

    print(f'\n📈 Test Accuracy: {acc*100:.2f}%')
    print(classification_report(labels, preds, target_names=['Negative', 'Positive'], digits=4))

    plot_confusion_matrix(labels, preds)
    plot_training_history(history)
    plot_roc_curve(labels, probs)
    plot_pr_curve(labels, probs)
    save_dashboard_json(report, labels, preds, probs, history, cfg)


print('✅ Evaluation functions defined.')

model, history = run_training(CFG)
run_evaluation(model, history, CFG)