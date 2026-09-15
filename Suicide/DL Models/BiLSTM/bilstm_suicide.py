import pandas as pd
import numpy as np
import re
import string
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, precision_recall_curve, auc,
                             confusion_matrix, classification_report)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

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

df = load_and_standardize_data(df1_path, df2_path)
if df is not None:
    print(f"Total samples: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#','', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Preprocessing text...")
df['clean_text'] = df['text'].apply(clean_text)

max_words = 20000
max_len = 250

tokenizer = Tokenizer(num_words=max_words, lower=True)
tokenizer.fit_on_texts(df['clean_text'])
sequences = tokenizer.texts_to_sequences(df['clean_text'])
X = pad_sequences(sequences, maxlen=max_len)
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


vocab_size = 20000
embedding_dim = 128
lstm_units = 64

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim),
    SpatialDropout1D(0.2),
    # Bidirectional layer wraps the LSTM layer
    Bidirectional(LSTM(lstm_units, dropout=0.2, recurrent_dropout=0)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=256,
    validation_split=0.1,
    callbacks=[early_stop]
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")


print("Generating predictions...")
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype("int32")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Suicide', 'Suicide']))

# ── Confusion Matrix (%) ──────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred, normalize='true')
plt.figure(figsize=(8, 6))
sns.heatmap(cm * 100, annot=True, fmt='.1f', cmap='Greens',
            xticklabels=['Non-Suicide', 'Suicide'],
            yticklabels=['Non-Suicide', 'Suicide'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Bi-LSTM Model (%)')
plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'bilstm_confusion_matrix.png', dpi=150)
plt.show()
print('✅ Confusion matrix saved → bilstm_confusion_matrix.png')

# ── Training History ──────────────────────────────────────────────────────────
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],     label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy - Bi-LSTM')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],     label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss - Bi-LSTM')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'bilstm_training_history.png', dpi=150)
plt.show()
print('✅ Training history saved → bilstm_training_history.png')

# ── ROC Curve ─────────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
auc_roc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC curve (AUC = {auc_roc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Bi-LSTM')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'bilstm_roc_curve.png', dpi=150)
plt.show()
print('✅ ROC curve saved → bilstm_roc_curve.png')

# ── Precision-Recall Curve ────────────────────────────────────────────────────
precision_arr, recall_arr, _ = precision_recall_curve(y_test, y_pred_prob)
auc_pr = auc(recall_arr, precision_arr)

plt.figure(figsize=(6, 5))
plt.plot(recall_arr, precision_arr, color='darkorange', lw=2,
         label=f'PR curve (AUC = {auc_pr:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Bi-LSTM')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'bilstm_pr_curve.png', dpi=150)
plt.show()
print('✅ PR curve saved → bilstm_pr_curve.png')

# ── Dashboard JSON ────────────────────────────────────────────────────────────
report_dict = classification_report(
    y_test, y_pred,
    target_names=['Non-Suicide', 'Suicide'],
    output_dict=True
)

# Raw CM for counts
cm_raw = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm_raw.ravel()

# Downsample ROC & PR to 12 points for the dashboard
roc_indices = np.linspace(0, len(fpr) - 1, 12, dtype=int)
roc_points  = [[round(float(fpr[i]), 3), round(float(tpr[i]), 3)] for i in roc_indices]

pr_indices = np.linspace(0, len(precision_arr) - 1, 12, dtype=int)
pr_points  = [[round(float(recall_arr[i]), 3), round(float(precision_arr[i]), 3)] for i in pr_indices]

# Convert Keras history to plain lists
train_acc  = [round(float(v), 4) for v in history.history['accuracy']]
val_acc    = [round(float(v), 4) for v in history.history['val_accuracy']]
train_loss = [round(float(v), 4) for v in history.history['loss']]
val_loss   = [round(float(v), 4) for v in history.history['val_loss']]

token_lengths = [len(str(t).split()) for t in df['clean_text']]

dashboard = {
    'model'    : 'BiLSTM',
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
        'train_acc' : train_acc,
        'val_acc'   : val_acc,
        'train_loss': train_loss,
        'val_loss'  : val_loss,
    },
    'dataset_stats': {
        'total_samples'  : len(df),
        'train_samples'  : len(X_train),
        'test_samples'   : len(X_test),
        'avg_text_length': round(float(np.mean(token_lengths)), 1),
        'max_text_length': int(np.max(token_lengths)),
        'min_text_length': int(np.min(token_lengths)),
    },
}

json_out = SCRIPT_DIR / 'bilstm_suicide_dashboard.json'
with open(json_out, 'w') as f:
    json.dump(dashboard, f, indent=2)
print(f'✅ Dashboard JSON saved → {json_out}')