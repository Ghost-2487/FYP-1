import re
import string
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import issparse
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import json
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


class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization='l2', lambda_reg=0.01, batch_size=None):

        self.lr = learning_rate
        self.n_iter = n_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        self.losses = []

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, X, y):
        m = len(y)

        linear_model = X.dot(self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

        if self.regularization == 'l2':
            loss += (self.lambda_reg / (2 * m)) * np.sum(self.weights ** 2)
        elif self.regularization == 'l1':
            loss += (self.lambda_reg / m) * np.sum(np.abs(self.weights))

        return loss

    def fit(self, X, y, verbose=True):

        n_samples = X.shape[0]
        n_features = X.shape[1]

        self.weights = np.zeros(n_features)
        self.bias = 0

        # Convert y to numpy array if it's a Series
        if hasattr(y, 'values'):
            y = y.values

        # Gradient descent
        for i in range(self.n_iter):
            linear_model = X.dot(self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            error = y_pred - y

            dw = (1 / n_samples) * X.T.dot(error)
            db = (1 / n_samples) * np.sum(error)

            if issparse(dw):
                dw = np.array(dw).flatten()

            if self.regularization == 'l2':
                dw += (self.lambda_reg / n_samples) * self.weights
            elif self.regularization == 'l1':
                dw += (self.lambda_reg / n_samples) * np.sign(self.weights)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Track loss every 100 iterations
            if i % 100 == 0:
                loss = self.compute_loss(X, y)
                self.losses.append(loss)
                if verbose:
                    print(f"Iteration {i}: Loss = {loss:.4f}")

        if verbose:
            final_loss = self.compute_loss(X, y)
            print(f"Training completed. Final loss: {final_loss:.4f}")

    def predict_proba(self, X):
        linear_model = X.dot(self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def evaluate(self, X, y, threshold=0.5):
        y_pred = self.predict(X, threshold)

        # Convert y to numpy if needed
        if hasattr(y, 'values'):
            y = y.values

        tp = np.sum((y_pred == 1) & (y == 1))
        tn = np.sum((y_pred == 0) & (y == 0))
        fp = np.sum((y_pred == 1) & (y == 0))
        fn = np.sum((y_pred == 0) & (y == 1))

        accuracy = (tp + tn) / len(y)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
        }


print("Train TF-IDF shape:", X_train_tfidf.shape)
print("Test TF-IDF shape:", X_test_tfidf.shape)
print("Is sparse?", issparse(X_train_tfidf))

# Normalize all label variants to strict 0/1 regardless of mixed types
y_train_binary = y_train.map(lambda x: 0 if str(x).lower() in ['non-suicide', '0'] else 1).values
y_test_binary  = y_test.map( lambda x: 0 if str(x).lower() in ['non-suicide', '0'] else 1).values

label_mapping = {'non-suicide': 0, 'suicide': 1}  # clean, JSON-safe, string keys only

print("\nLabel Mapping:", label_mapping)
print(f"\nTrain labels: {y_train_binary.shape}")
print(f"Test labels:  {y_test_binary.shape}")
print(f"Unique train values: {np.unique(y_train_binary)}")  # must be [0 1]
print(f"Unique test values:  {np.unique(y_test_binary)}")   # must be [0 1]
print(f"Class distribution in train: 0={np.sum(y_train_binary==0)}, 1={np.sum(y_train_binary==1)}")
print(f"Class distribution in test:  0={np.sum(y_test_binary==0)}, 1={np.sum(y_test_binary==1)}")

print("\nTRAINING LOGISTIC REGRESSION MODEL\n")

model = LogisticRegression(
    learning_rate=0.5,
    n_iterations=1000,
    regularization='l2',
    lambda_reg=0.01
)

model.fit(X_train_tfidf, y_train_binary, verbose=True)
print("\nMAKING PREDICTIONS\n")

y_train_pred = model.predict(X_train_tfidf)
y_test_pred = model.predict(X_test_tfidf)

print("Predictions completed!")

print("\nTRAINING SET EVALUATION\n")

train_metrics = model.evaluate(X_train_tfidf, y_train_binary)
print(f"Accuracy:  {train_metrics['accuracy']:.4f}")
print(f"Precision: {train_metrics['precision']:.4f}")
print(f"Recall:    {train_metrics['recall']:.4f}")
print(f"F1-Score:  {train_metrics['f1_score']:.4f}")

print("\nTEST SET EVALUATION\n")

test_metrics = model.evaluate(X_test_tfidf, y_test_binary)
print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
print(f"Precision: {test_metrics['precision']:.4f}")
print(f"Recall:    {test_metrics['recall']:.4f}")
print(f"F1-Score:  {test_metrics['f1_score']:.4f}")

print("\nConfusion Matrix:")
cm = test_metrics['confusion_matrix']
print(f"  True Positives:  {cm['tp']}")
print(f"  True Negatives:  {cm['tn']}")
print(f"  False Positives: {cm['fp']}")
print(f"  False Negatives: {cm['fn']}")

plt.figure(figsize=(10, 6))
plt.plot(range(0, len(model.losses) * 100, 100), model.losses, linewidth=2, marker='o')
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'lr_suicide_training_history.png', dpi=150, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
confusion = np.array([[cm['tn'], cm['fp']],
                      [cm['fn'], cm['tp']]])

im = ax.imshow(confusion, cmap='Blues')

binary_plot_labels = [label for label, idx in label_mapping.items() if idx in [0, 1]]
binary_plot_labels.sort(key=lambda x: label_mapping[x])

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(binary_plot_labels)
ax.set_yticklabels(binary_plot_labels)

for i in range(2):
    for j in range(2):
        text = ax.text(j, i, confusion[i, j],
                      ha="center", va="center", color="white" if confusion[i, j] > confusion.max()/2 else "black",
                      fontsize=20, fontweight='bold')

ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'lr_suicide_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nTHRESHOLD TUNING\n")

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
best_f1 = 0
best_thresh = 0.5

for thresh in thresholds:
    metrics = model.evaluate(X_test_tfidf, y_test_binary, threshold=thresh)
    print(f"\nThreshold: {thresh:.1f}")
    print(f"  Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1_score']:.4f}")

    if metrics['f1_score'] > best_f1:
        best_f1 = metrics['f1_score']
        best_thresh = thresh

print(f"\n\u2713 Best threshold: {best_thresh} with F1-Score: {best_f1:.4f}")

# ============================================
# ROC CURVE
# ============================================

print("\n" + "="*60)
print("ROC CURVE & AUC SCORE")
print("="*60)

y_test_proba = model.predict_proba(X_test_tfidf)

fpr, tpr, _ = roc_curve(y_test_binary, y_test_proba)
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
plt.title('ROC Curve — Logistic Regression (Suicide Dataset)',
          fontsize=14, fontweight='bold', pad=20)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'lr_suicide_roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# PRECISION-RECALL CURVE
# ============================================

print("\n" + "="*60)
print("PRECISION-RECALL CURVE")
print("="*60)

precision_curve, recall_curve, _ = precision_recall_curve(y_test_binary, y_test_proba)
average_precision = average_precision_score(y_test_binary, y_test_proba)

print(f"\n🎯 Average Precision Score: {average_precision:.4f}")

plt.figure(figsize=(10, 8))
plt.plot(recall_curve, precision_curve, color='darkorange', lw=3,
         label=f'PR Curve (AP = {average_precision:.4f})')
plt.axhline(y=np.sum(y_test_binary) / len(y_test_binary), color='navy', lw=2,
            linestyle='--', label='Baseline')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=12, fontweight='bold')
plt.ylabel('Precision', fontsize=12, fontweight='bold')
plt.title('Precision-Recall Curve — Logistic Regression (Suicide Dataset)',
          fontsize=14, fontweight='bold', pad=20)
plt.legend(loc='lower left', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'lr_suicide_pr_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# SAVE DASHBOARD JSON
# ============================================

print("\n" + "="*60)
print("SAVE DASHBOARD JSON")
print("="*60)

def save_dashboard_json():
    # ROC curve — sample 12 points
    roc_indices = np.linspace(0, len(fpr) - 1, 12, dtype=int)
    roc_points  = [[round(float(fpr[i]), 3), round(float(tpr[i]), 3)] for i in roc_indices]

    # PR curve — sample 12 points
    pr_indices = np.linspace(0, len(precision_curve) - 1, 12, dtype=int)
    pr_points  = [[round(float(recall_curve[i]), 3), round(float(precision_curve[i]), 3)] for i in pr_indices]

    # Training history (loss per 100 iterations)
    training_history = {
        'iterations': list(range(0, len(model.losses) * 100, 100)),
        'train_loss': [round(float(l), 6) for l in model.losses],
    }

    tp = cm['tp']
    tn = cm['tn']
    fp = cm['fp']
    fn = cm['fn']

    dashboard = {
        'model'     : 'Logistic Regression',
        'dataset'   : 'Suicide Detection',
        'accuracy'  : round(test_metrics['accuracy'] * 100, 2),
        'precision' : round(test_metrics['precision'] * 100, 2),
        'recall'    : round(test_metrics['recall'] * 100, 2),
        'f1'        : round(test_metrics['f1_score'] * 100, 2),
        'auc_roc'   : round(float(roc_auc), 4),
        'auc_pr'    : round(float(average_precision), 4),
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
            'train_samples': int(X_train_tfidf.shape[0]),
            'test_samples' : int(X_test_tfidf.shape[0]),
            'n_features'   : int(X_train_tfidf.shape[1]),
        },
    }

    with open(SCRIPT_DIR / 'lr_suicide_dashboard.json', 'w') as f:
        json.dump(dashboard, f, indent=2)
    print('✅ Dashboard JSON saved → lr_suicide_dashboard.json')

save_dashboard_json()