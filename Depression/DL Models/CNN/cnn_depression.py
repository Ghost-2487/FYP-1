import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve,
    classification_report, precision_recall_curve,
    average_precision_score, auc
)
import json
import time
import pickle
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

# ============================================
# 0. CONFIGURATION & REPRODUCIBILITY
# ============================================
np.random.seed(42)
tf.random.set_seed(42)
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("viridis")

print("="*70)
print("CONVOLUTIONAL NEURAL NETWORK (CNN)")
print("="*70)

# ============================================
# 1. LOAD PREPROCESSED DATA AND TOKENIZER
# ============================================
print("\n" + "="*70)
print("STEP 1: LOAD PREPROCESSED DATA AND TOKENIZER")
print("="*70)

try:
    x_train = np.load(BASE_DIR / "data" / "x_train_sequences.npy")
    x_test = np.load(BASE_DIR / "data" / "x_test_sequences.npy")
    y_train = np.load(BASE_DIR / "data" / "y_train.npy")
    y_test = np.load(BASE_DIR / "data" / "y_test.npy")

    with open(BASE_DIR / "data" / "tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
    with open(BASE_DIR / "data" / "label_encoder.pickle", "rb") as handle:
        encoder = pickle.load(handle)

    vocab_size = len(tokenizer.word_index) + 1
    max_sequence_length = x_train.shape[1]
    num_classes = len(encoder.classes_)

    print(f"✅ Data loaded successfully!")
    print(f"   Vocabulary Size: {vocab_size}")
    print(f"   Max Sequence Length: {max_sequence_length}")
except FileNotFoundError as e:
    print(f"❌ Error: {e}. Run preprocess_sentiment_data.py first.")
    exit()

# ============================================
# 2. PREPARE GLOVE EMBEDDINGS
# ============================================
print("\n" + "="*70)
print("STEP 2: PREPARE GLOVE EMBEDDINGS")
print("="*70)

GLOVE_PATH = BASE_DIR / "data" / 'glove.twitter.27B.100d.txt'
EMBEDDING_DIM = 100

def load_glove_embeddings(path, word_index, embedding_dim):
    print(f"Loading GloVe embeddings from {path}...")
    embeddings_index = {}
    try:
        if not os.path.exists(path):
            print(f"⚠️ GloVe file not found at {path}. Using random initialization.")
            return None
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                try:
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
                except ValueError: continue
        embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
        hits = 0
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                hits += 1
        print(f"✅ Converted {hits} words ({100*hits/len(word_index):.2f}% coverage)")
        return embedding_matrix
    except Exception as e:
        print(f"⚠️ Error loading GloVe: {e}.")
        return None

embedding_matrix = load_glove_embeddings(GLOVE_PATH, tokenizer.word_index, EMBEDDING_DIM)

# ============================================
# 3. BUILD CNN MODEL
# ============================================
print("\n" + "="*70)
print("STEP 3: BUILD CNN MODEL")
print("="*70)

# Builds a high-capacity CNN model with a hybrid embedding approach.
def build_cnn(vocab_size, max_length, num_classes, embedding_matrix=None,
              filters=128, kernel_size=5, dropout_rate=0.4, learning_rate=1e-3):
    
    inputs = Input(shape=(max_length,))

    # Branch 1: Pre-trained GloVe (Frozen)
    if embedding_matrix is not None:
        emb_glove = Embedding(
            input_dim=vocab_size,
            output_dim=EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=max_length,
            trainable=False,
            name="glove_static"
        )(inputs)
    else:
        emb_glove = Embedding(vocab_size, 100, input_length=max_length, name="glove_fallback")(inputs)

    # Branch 2: Fully Trainable Embedding
    emb_trainable = Embedding(
        input_dim=vocab_size,
        output_dim=200,
        input_length=max_length,
        trainable=True,
        name="domain_specific"
    )(inputs)

    # Concatenate both embedding spaces
    merged = Concatenate()([emb_glove, emb_trainable])
    x = Dropout(dropout_rate)(merged)

    # CNN Layer
    x = Conv1D(filters, kernel_size, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)

    # Dense head
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = build_cnn(
    vocab_size=vocab_size,
    max_length=max_sequence_length,
    num_classes=num_classes,
    embedding_matrix=embedding_matrix
)

print("✅ Hybrid CNN built successfully!")
model.summary()

# ============================================
# 4. TRAIN MODEL
# ============================================
print("\n" + "="*70)
print("STEP 4: TRAIN CNN MODEL")
print("="*70)

early_stopping = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=0.00001)

BATCH_SIZE = 512
EPOCHS = 15

print("\n🚀 Training started...")
start_time = time.time()
history = model.fit(
    x_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.05,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)
print(f"✅ Training completed in {time.time() - start_time:.2f}s")

# ============================================
# 5. EVALUATION
# ============================================

print("\n" + "="*70)
print("STEP 5: EVALUATION")
print("="*70)

print("\n📊 Predicting on test set...")
y_test_proba = model.predict(x_test, batch_size=BATCH_SIZE)
y_test_pred = np.argmax(y_test_proba, axis=1)
y_test_flat = y_test.flatten()

test_accuracy = accuracy_score(y_test_flat, y_test_pred)
print(f"\n📈 Test Set Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print("\nDETAILED CLASSIFICATION REPORT")
print(classification_report(y_test_flat, y_test_pred, target_names=encoder.classes_, digits=4))

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Model Accuracy")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Model Loss")
plt.legend()
plt.savefig(SCRIPT_DIR / "cnn_training_history.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# 6. CONFUSION MATRIX
# ============================================

print("\n" + "="*70)
print("STEP 6: CONFUSION MATRIX")
print("="*70)

cm = confusion_matrix(y_test_flat, y_test_pred)
target_names = list(encoder.classes_)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names,
            cbar_kws={'label': 'Count'},
            ax=axes[0], annot_kws={'size': 14, 'weight': 'bold'})
axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold', pad=20)
axes[0].set_ylabel('True Label', fontsize=12)
axes[0].set_xlabel('Predicted Label', fontsize=12)

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names,
            cbar_kws={'label': 'Percentage'},
            ax=axes[1], annot_kws={'size': 14, 'weight': 'bold'})
axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold', pad=20)
axes[1].set_ylabel('True Label', fontsize=12)
axes[1].set_xlabel('Predicted Label', fontsize=12)

plt.tight_layout()
plt.savefig(SCRIPT_DIR / "cnn_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"  True Negatives:  {tn:,}")
print(f"  False Positives: {fp:,}")
print(f"  False Negatives: {fn:,}")
print(f"  True Positives:  {tp:,}")

# ============================================
# 7. ROC CURVE
# ============================================

print("\n" + "="*70)
print("STEP 7: ROC CURVE & AUC SCORE")
print("="*70)

# y_test_proba[:, 1] is the probability for the positive class
fpr, tpr, _ = roc_curve(y_test_flat, y_test_proba[:, 1])
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
plt.title('ROC Curve — CNN', fontsize=14, fontweight='bold', pad=20)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "cnn_roc_curve.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# 8. PRECISION-RECALL CURVE
# ============================================

print("\n" + "="*70)
print("STEP 8: PRECISION-RECALL CURVE")
print("="*70)

precision_curve, recall_curve, _ = precision_recall_curve(y_test_flat, y_test_proba[:, 1])
average_precision = average_precision_score(y_test_flat, y_test_proba[:, 1])

print(f"\n🎯 Average Precision Score: {average_precision:.4f}")

plt.figure(figsize=(10, 8))
plt.plot(recall_curve, precision_curve, color='darkorange', lw=3,
         label=f'PR Curve (AP = {average_precision:.4f})')
plt.axhline(y=np.sum(y_test_flat) / len(y_test_flat), color='navy', lw=2,
            linestyle='--', label='Baseline')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=12, fontweight='bold')
plt.ylabel('Precision', fontsize=12, fontweight='bold')
plt.title('Precision-Recall Curve — CNN', fontsize=14, fontweight='bold', pad=20)
plt.legend(loc='lower left', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "cnn_pr_curve.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# 9. SAVE DASHBOARD JSON
# ============================================

print("\n" + "="*70)
print("STEP 9: SAVE DASHBOARD JSON")
print("="*70)

def save_dashboard_json():
    report = classification_report(
        y_test_flat, y_test_pred,
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
        'dataset'   : 'Sentiment140',
        'accuracy'  : round(test_accuracy * 100, 2),
        'precision' : round(report['weighted avg']['precision'] * 100, 2),
        'recall'    : round(report['weighted avg']['recall'] * 100, 2),
        'f1'        : round(report['weighted avg']['f1-score'] * 100, 2),
        'auc_roc'   : round(float(roc_auc), 4),
        'auc_pr'    : round(float(average_precision), 4),
        'class_metrics': {
            'Negative': {k: round(v * 100, 2) for k, v in report['Negative'].items() if k != 'support'},
            'Positive': {k: round(v * 100, 2) for k, v in report['Positive'].items() if k != 'support'},
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
            'train_samples': int(x_train.shape[0]),
            'test_samples' : int(x_test.shape[0]),
            'vocab_size'   : vocab_size,
            'max_seq_len'  : max_sequence_length,
        },
    }

    with open(SCRIPT_DIR / "cnn_sentiment140_dashboard.json", 'w') as f:
        json.dump(dashboard, f, indent=2)
    print('✅ Dashboard JSON saved → cnn_sentiment140_dashboard.json')

save_dashboard_json()