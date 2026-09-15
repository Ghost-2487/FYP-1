import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, precision_recall_curve,
                             average_precision_score, auc,
                             confusion_matrix, classification_report)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, SpatialDropout1D
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


# 1. Text Cleaning
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

print("Building CNN model...")
model = Sequential([
    Embedding(max_words, 128),
    SpatialDropout1D(0.3),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

print("Training model...")
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=10,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

print("Evaluating model...")
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype("int32")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Suicide', 'Suicide']))

# ============================================
# CONFUSION MATRIX (NORMALIZED %)
# ============================================

cm = confusion_matrix(y_test, y_pred)
target_names = ['Non-Suicide', 'Suicide']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Raw counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names,
            cbar_kws={'label': 'Count'},
            ax=axes[0], annot_kws={'size': 14, 'weight': 'bold'})
axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold', pad=20)
axes[0].set_ylabel('True Label', fontsize=12)
axes[0].set_xlabel('Predicted Label', fontsize=12)

# Normalized %
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names,
            cbar_kws={'label': 'Percentage'},
            ax=axes[1], annot_kws={'size': 14, 'weight': 'bold'})
axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold', pad=20)
axes[1].set_ylabel('True Label', fontsize=12)
axes[1].set_xlabel('Predicted Label', fontsize=12)

plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'cnn_suicide_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"  True Negatives:  {tn:,}")
print(f"  False Positives: {fp:,}")
print(f"  False Negatives: {fn:,}")
print(f"  True Positives:  {tp:,}")

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue', marker='o')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='red', marker='o')
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='blue', marker='o')
    plt.plot(history.history['val_loss'], label='Val Loss', color='red', marker='o')
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / 'cnn_suicide_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()

if 'history' in locals():
    plot_training_history(history)
else:
    print("Error: 'history' object not found. Please train the model first.")

# ============================================
# ROC CURVE
# ============================================

print("\n" + "="*60)
print("ROC CURVE & AUC SCORE")
print("="*60)

# y_pred_prob is already 1D — sigmoid binary output
y_scores = y_pred_prob.flatten()

fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

print(f"\n🎯 ROC-AUC Score: {roc_auc:.4f}")

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='steelblue', lw=3,
         label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
         label='Random Classifier (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('ROC Curve — CNN (Suicide Dataset)',
          fontsize=14, fontweight='bold', pad=20)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'cnn_suicide_roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# PRECISION-RECALL CURVE
# ============================================

print("\n" + "="*60)
print("PRECISION-RECALL CURVE")
print("="*60)

precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_scores)
average_precision = average_precision_score(y_test, y_scores)

print(f"\n🎯 Average Precision Score: {average_precision:.4f}")

plt.figure(figsize=(10, 8))
plt.plot(recall_curve, precision_curve, color='darkorange', lw=3,
         label=f'PR Curve (AP = {average_precision:.4f})')
plt.axhline(y=np.sum(y_test) / len(y_test), color='navy', lw=2,
            linestyle='--', label='Baseline')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=12, fontweight='bold')
plt.ylabel('Precision', fontsize=12, fontweight='bold')
plt.title('Precision-Recall Curve — CNN (Suicide Dataset)',
          fontsize=14, fontweight='bold', pad=20)
plt.legend(loc='lower left', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'cnn_suicide_pr_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# SAVE DASHBOARD JSON
# ============================================

print("\n" + "="*60)
print("SAVE DASHBOARD JSON")
print("="*60)

def save_dashboard_json():
    report = classification_report(
        y_test, y_pred,
        target_names=target_names,
        output_dict=True,
    )

    # ROC curve — sample 12 points
    roc_indices = np.linspace(0, len(fpr) - 1, 12, dtype=int)
    roc_points  = [[round(float(fpr[i]), 3), round(float(tpr[i]), 3)] for i in roc_indices]

    # PR curve — sample 12 points
    pr_indices = np.linspace(0, len(precision_curve) - 1, 12, dtype=int)
    pr_points  = [[round(float(recall_curve[i]), 3), round(float(precision_curve[i]), 3)] for i in pr_indices]

    # Training history
    training_history = {
        'train_accuracy': [round(float(v), 4) for v in history.history['accuracy']],
        'val_accuracy'  : [round(float(v), 4) for v in history.history['val_accuracy']],
        'train_loss'    : [round(float(v), 6) for v in history.history['loss']],
        'val_loss'      : [round(float(v), 6) for v in history.history['val_loss']],
    }

    dashboard = {
        'model'     : 'CNN',
        'dataset'   : 'Suicide Detection',
        'accuracy'  : round(report['accuracy'] * 100, 2),
        'precision' : round(report['weighted avg']['precision'] * 100, 2),
        'recall'    : round(report['weighted avg']['recall'] * 100, 2),
        'f1'        : round(report['weighted avg']['f1-score'] * 100, 2),
        'auc_roc'   : round(float(roc_auc), 4),
        'auc_pr'    : round(float(average_precision), 4),
        'class_metrics': {
            'Non-Suicide': {k: round(v * 100, 2) for k, v in report['Non-Suicide'].items() if k != 'support'},
            'Suicide'    : {k: round(v * 100, 2) for k, v in report['Suicide'].items() if k != 'support'},
        },
        'confusion_matrix': {
            'tp': round(tp / (tp + fn) * 100, 2),
            'fp': round(fp / (fp + tn) * 100, 2),
            'fn': round(fn / (tp + fn) * 100, 2),
            'tn': round(tn / (fp + tn) * 100, 2),
        },
        'roc_curve'       : roc_points,
        'pr_curve'        : pr_points,
        'training_history': training_history,
        'dataset_stats': {
            'train_samples': int(X_train.shape[0]),
            'test_samples' : int(X_test.shape[0]),
            'vocab_size'   : max_words,
            'max_seq_len'  : max_len,
        },
    }

    with open(SCRIPT_DIR / 'cnn_suicide_dashboard.json', 'w') as f:
        json.dump(dashboard, f, indent=2)
    print('✅ Dashboard JSON saved → cnn_suicide_dashboard.json')

save_dashboard_json()