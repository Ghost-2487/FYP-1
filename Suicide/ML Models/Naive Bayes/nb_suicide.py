import pandas as pd
import numpy as np
import re
import string
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

class NaiveBayesFromScratch:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.priors = None
        self.feature_log_probs = None
        self.classes = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        self.priors = np.zeros(n_classes)
        self.feature_log_probs = np.zeros((n_classes, n_features))
        
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.priors[i] = X_c.shape[0] / n_samples
            
            count_w_c = np.array(X_c.sum(axis=0)).flatten()
            total_count_c = count_w_c.sum()
            
            self.feature_log_probs[i, :] = np.log((count_w_c + self.alpha) / 
                                                 (total_count_c + self.alpha * n_features))
        
        self.priors = np.log(self.priors)

    def predict(self, X):
        joint_log_likelihood = X @ self.feature_log_probs.T + self.priors
        return self.classes[np.argmax(joint_log_likelihood, axis=1)]
    
    # Returns probability of positive class (class index 1).
    def predict_proba(self, X):
        joint_log_likelihood = X @ self.feature_log_probs.T + self.priors
        # Softmax to convert log-likelihoods to probabilities
        joint_ll = joint_log_likelihood - joint_log_likelihood.max(axis=1, keepdims=True)
        exp_ll   = np.exp(joint_ll)
        proba    = exp_ll / exp_ll.sum(axis=1, keepdims=True)
        return proba[:, 1]   # positive class probability


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


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#','', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return re.sub(r'\s+', ' ', text).strip()

df['clean_text'] = df['text'].apply(clean_text)
X_train_t, X_test_t, y_train, y_test = train_test_split(df['clean_text'], df['label'].values, test_size=0.2, random_state=42, stratify=df['label'])

tfidf = TfidfVectorizer(max_features=20000)
X_train = tfidf.fit_transform(X_train_t)
X_test = tfidf.transform(X_test_t)

print("Training from scratch model...")
nb = NaiveBayesFromScratch()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

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
plt.savefig(SCRIPT_DIR / 'nb_suicide_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"  True Negatives:  {tn:,}")
print(f"  False Positives: {fp:,}")
print(f"  False Negatives: {fn:,}")
print(f"  True Positives:  {tp:,}")

# ============================================
# ROC CURVE
# ============================================

print("\n" + "="*60)
print("ROC CURVE & AUC SCORE")
print("="*60)

y_test_proba = nb.predict_proba(X_test)

fpr, tpr, _ = roc_curve(y_test, y_test_proba)
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
plt.title('ROC Curve — Naive Bayes (Suicide Dataset)',
          fontsize=14, fontweight='bold', pad=20)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'nb_suicide_roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# PRECISION-RECALL CURVE
# ============================================

print("\n" + "="*60)
print("PRECISION-RECALL CURVE")
print("="*60)

precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_test_proba)
average_precision = average_precision_score(y_test, y_test_proba)

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
plt.title('Precision-Recall Curve — Naive Bayes (Suicide Dataset)',
          fontsize=14, fontweight='bold', pad=20)
plt.legend(loc='lower left', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'nb_suicide_pr_curve.png', dpi=150, bbox_inches='tight')
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

    dashboard = {
        'model'     : 'Naive Bayes',
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
        'training_history': None,   # NB trains in one pass — no epoch history
        'dataset_stats': {
            'train_samples': int(X_train.shape[0]),
            'test_samples' : int(X_test.shape[0]),
            'n_features'   : int(X_train.shape[1]),
        },
    }

    with open(SCRIPT_DIR / 'nb_suicide_dashboard.json', 'w') as f:
        json.dump(dashboard, f, indent=2)
    print('✅ Dashboard JSON saved → nb_suicide_dashboard.json')

save_dashboard_json()