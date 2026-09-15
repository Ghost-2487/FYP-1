import re
import string
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (roc_curve, precision_recall_curve, precision_score,
                             average_precision_score, classification_report, auc,
                             accuracy_score, recall_score, f1_score, confusion_matrix)
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

try:
    df1 = pd.read_csv(BASE_DIR / "data" / "Suicide_Detection.csv")
    df2 = pd.read_csv(BASE_DIR / "data" / "suicide_only_clean.csv")

    print(f"✅ Data loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Error: {e}. Run extract_suicide_data.py first.")
    exit()

print("DF1 Columns:", df1.columns.tolist())
print("DF2 Columns:", df2.columns.tolist())

if "Unnamed: 0" in df1.columns:
    df1 = df1.drop(columns=["Unnamed: 0"])

df1 = df1.rename(columns={"class": "label"})

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)   # remove URLs
    text = re.sub(r'\@\w+|\#','', text)                   # remove @ and #
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df1["clean_text"] = df1["text"].apply(clean_text)

combined = pd.concat([
    df1[['text', 'label', 'clean_text']],
    df2[['text', 'label', 'clean_text']]
], ignore_index=True)

print("\nCombined Shape:", combined.shape)
print("\nLabel Counts:\n", combined["label"].value_counts())
combined.sample(5)

# Removing duplicates
combined = combined.drop_duplicates(subset=["clean_text"])
combined = combined.dropna(subset=["clean_text"])
combined = combined[combined["clean_text"].str.strip() != ""]

print("After removing duplicates & empty rows:", combined.shape)
print("\nClass Distribution:")
print(combined["label"].value_counts())


X = combined["clean_text"]
y = combined["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)


tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words="english")

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF train shape:", X_train_tfidf.shape)
print("TF-IDF test shape:", X_test_tfidf.shape)


print("Shapes:")
print("X_train:", X_train_tfidf.shape)
print("X_test :", X_test_tfidf.shape)
print("y_train:", len(y_train))
print("y_test :", len(y_test))


y_train = np.where(y_train == "suicide", 1, 0)
y_test = np.where(y_test == "suicide", 1, 0)

print("Original Data:")
print(f"X_train: {X_train_tfidf.shape}, X_test: {X_test_tfidf.shape}")
print(f"Train - Non-suicide: {np.sum(y_train == 0)}, Suicide: {np.sum(y_train == 1)}")
print(f"Test - Non-suicide: {np.sum(y_test == 0)}, Suicide: {np.sum(y_test == 1)}")
print("=" * 60)

# BETTER BALANCING WITH SMOTE-LIKE APPROACH
idx_0 = np.where(y_train == 0)[0]
idx_1 = np.where(y_train == 1)[0]

print(f"\nOriginal class distribution:")
print(f"Class 0: {len(idx_0)} samples")
print(f"Class 1: {len(idx_1)} samples")

# Upsample minority class to match majority
if len(idx_1) < len(idx_0):
    # Upsample class 1 (suicide)
    idx_1_upsampled = resample(idx_1,
                                replace=True,
                                n_samples=len(idx_0),
                                random_state=42)
    balanced_indices = np.concatenate([idx_0, idx_1_upsampled])
else:
    # Upsample class 0 (non-suicide)
    idx_0_upsampled = resample(idx_0,
                                replace=True,
                                n_samples=len(idx_1),
                                random_state=42)
    balanced_indices = np.concatenate([idx_0_upsampled, idx_1])

# Shuffle indices
np.random.seed(42)
np.random.shuffle(balanced_indices)

# Create balanced dataset
X_train_bal = X_train_tfidf[balanced_indices]
y_train_bal = y_train[balanced_indices]

print(f"\nBalanced class distribution:")
print(f"Class 0: {np.sum(y_train_bal == 0)} samples")
print(f"Class 1: {np.sum(y_train_bal == 1)} samples")
print("=" * 60)


class LinearSVM:
    def __init__(self, lr=0.1, lambda_param=0.0001, n_iters=1000, batch_size=None):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.w = None
        self.b = None

    def fit(self, X, y):
        y_ = np.where(y == 0, -1, 1).astype(float)
        n_samples, n_features = X.shape

        # Initialize weights with small random values
        self.w = np.random.randn(n_features) * 0.01
        self.b = 0.0

        is_sparse = hasattr(X, 'toarray')

        print(f"\nTraining SVM on {n_samples} samples with {n_features} features")
        print(f"Learning rate: {self.lr}, Lambda: {self.lambda_param}\n")

        for iteration in range(self.n_iters):
            # Mini-batch gradient descent
            if self.batch_size and self.batch_size < n_samples:
                indices = np.random.choice(n_samples, self.batch_size, replace=False)
                X_batch = X[indices]
                y_batch = y_[indices]
                batch_size = self.batch_size
            else:
                X_batch = X
                y_batch = y_
                batch_size = n_samples

            # Compute scores
            if is_sparse:
                scores = np.array(X_batch.dot(self.w)).ravel() + self.b
            else:
                scores = np.dot(X_batch, self.w) + self.b

            # Hinge loss subgradient
            margins = y_batch * scores
            sv_mask = margins < 1  # Support vectors

            # Compute gradients
            if is_sparse:
                if np.sum(sv_mask) > 0:
                    grad_w = self.lambda_param * self.w - (X_batch[sv_mask].T.dot(y_batch[sv_mask]) / batch_size)
                    grad_w = np.array(grad_w).ravel()
                    grad_b = -np.sum(y_batch[sv_mask]) / batch_size
                else:
                    grad_w = self.lambda_param * self.w
                    grad_b = 0.0
            else:
                if np.sum(sv_mask) > 0:
                    grad_w = self.lambda_param * self.w - (np.dot(X_batch[sv_mask].T, y_batch[sv_mask]) / batch_size)
                    grad_b = -np.sum(y_batch[sv_mask]) / batch_size
                else:
                    grad_w = self.lambda_param * self.w
                    grad_b = 0.0

            # Update weights
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

            if (iteration + 1) % 100 == 0 or iteration == 0:
                if is_sparse:
                    train_scores = np.array(X.dot(self.w)).ravel() + self.b
                else:
                    train_scores = np.dot(X, self.w) + self.b

                train_preds = np.where(train_scores >= 0, 1, -1)
                train_acc = np.mean(train_preds == y_)

                n_pos = np.sum(train_preds == 1)
                n_neg = np.sum(train_preds == -1)
                n_sv = np.sum(sv_mask)

                print(f"Iter {iteration+1:4d} | Acc: {train_acc:.4f} | "
                      f"SVs: {n_sv:6d} | Pred(+1): {n_pos:6d} | Pred(-1): {n_neg:6d}")

    def predict(self, X):
        if hasattr(X, 'toarray'):
            scores = np.array(X.dot(self.w)).ravel() + self.b
        else:
            scores = np.dot(X, self.w) + self.b
        return np.where(scores >= 0, 1, 0)
    
    def decision_scores(self, X):
        """Returns raw decision scores for ROC/PR curve computation."""
        if hasattr(X, 'toarray'):
            return np.array(X.dot(self.w)).ravel() + self.b
        else:
            return np.dot(X, self.w) + self.b


print("\nTRAINING CUSTOM SVM\n")

svm = LinearSVM(lr=0.5, lambda_param=0.00001, n_iters=1000, batch_size=10000)
svm.fit(X_train_bal, y_train_bal)

print("=" * 60)


print("\nMaking predictions on test set...")
preds = svm.predict(X_test_tfidf)

print("\nPREDICTION DISTRIBUTION")
print("\n")
print(f"Predicted Non-Suicide (0): {np.sum(preds == 0)} ({np.sum(preds == 0)/len(preds)*100:.1f}%)")
print(f"Predicted Suicide (1): {np.sum(preds == 1)} ({np.sum(preds == 1)/len(preds)*100:.1f}%)\n")

#metrics
accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, zero_division=0)
recall = recall_score(y_test, preds, zero_division=0)
f1 = f1_score(y_test, preds, zero_division=0)

print("\nCUSTOM SVM PERFORMANCE\n")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ============================================
# CONFUSION MATRIX (NORMALIZED %)
# ============================================

cm = confusion_matrix(y_test, preds)
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
plt.savefig(SCRIPT_DIR / 'svm_suicide_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"  True Negatives:  {tn:,}")
print(f"  False Positives: {fp:,}")
print(f"  False Negatives: {fn:,}")
print(f"  True Positives:  {tp:,}")

# ============================================
# PCA SCATTER PLOT
# ============================================

print("\nGenerating PCA visualization...")
pca = PCA(n_components=2)
sample_size = min(5000, X_test_tfidf.shape[0])
sample_indices = np.random.choice(X_test_tfidf.shape[0], sample_size, replace=False)
X_test_sample = X_test_tfidf[sample_indices].toarray()
preds_sample = preds[sample_indices]

X_test_2D = pca.fit_transform(X_test_sample)

plt.figure(figsize=(10, 8))
plt.scatter(X_test_2D[preds_sample==0, 0], X_test_2D[preds_sample==0, 1],
            marker='o', label="Non-Suicide (0)", alpha=0.5, s=20, c='blue')
plt.scatter(X_test_2D[preds_sample==1, 0], X_test_2D[preds_sample==1, 1],
            marker='x', label="Suicide (1)", alpha=0.5, s=20, c='red')
plt.title("PCA Plot — Predictions", fontsize=14, fontweight='bold', pad=20)
plt.xlabel("PCA Component 1", fontsize=12)
plt.ylabel("PCA Component 2", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'svm_suicide_pca.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# ROC CURVE
# ============================================

print("\n" + "="*60)
print("ROC CURVE & AUC SCORE")
print("="*60)

y_scores = svm.decision_scores(X_test_tfidf)

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
plt.title('ROC Curve — SVM (Suicide Dataset)',
          fontsize=14, fontweight='bold', pad=20)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'svm_suicide_roc_curve.png', dpi=150, bbox_inches='tight')
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
plt.title('Precision-Recall Curve — SVM (Suicide Dataset)',
          fontsize=14, fontweight='bold', pad=20)
plt.legend(loc='lower left', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'svm_suicide_pr_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# SAVE DASHBOARD JSON
# ============================================

print("\n" + "="*60)
print("SAVE DASHBOARD JSON")
print("="*60)

def save_dashboard_json():
    report = classification_report(
        y_test, preds,
        target_names=target_names,
        output_dict=True,
    )

    # ROC curve — sample 12 points
    roc_indices = np.linspace(0, len(fpr) - 1, 12, dtype=int)
    roc_points  = [[round(float(fpr[i]), 3), round(float(tpr[i]), 3)] for i in roc_indices]

    # PR curve — sample 12 points
    pr_indices = np.linspace(0, len(precision_curve) - 1, 12, dtype=int)
    pr_points  = [[round(float(recall_curve[i]), 3), round(float(precision_curve[i]), 3)] for i in pr_indices]

    dashboard = {
        'model'     : 'SVM',
        'dataset'   : 'Suicide Detection',
        'accuracy'  : round(float(accuracy) * 100, 2),
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
        'training_history': None,   # Custom SVM has no epoch-level loss tracking
        'dataset_stats': {
            'train_samples'         : int(X_train_bal.shape[0]),
            'test_samples'          : int(X_test_tfidf.shape[0]),
            'n_features'            : int(X_test_tfidf.shape[1]),
            'balancing_method'      : 'Upsampling (resample with replacement)',
        },
    }

    with open(SCRIPT_DIR / 'svm_suicide_dashboard.json', 'w') as f:
        json.dump(dashboard, f, indent=2)
    print('✅ Dashboard JSON saved → svm_suicide_dashboard.json')

save_dashboard_json()


print("\nTraining complete!")