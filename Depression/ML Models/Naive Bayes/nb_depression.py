# ============================================
# NAIVE BAYES
# ============================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import issparse, csr_matrix, load_npz
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
import json
import time
import gc
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================
# NAIVE BAYES CLASSIFIER
# ============================================

class NaiveBayes:

    def __init__(self, alpha=0.1):
        """
        Initialize Naive Bayes with Laplace smoothing

        Parameters:
        -----------
        alpha : float, default=0.1
            Smoothing parameter
        """
        self.alpha = alpha
        self.classes_ = None
        self.class_prior_ = None
        self.feature_log_prob_ = None
        self.n_features_ = None

    # Fit classifier using sparse matrices
    def fit(self, X, y):
        print("\n" + "="*60)
        print("TRAINING NAIVE BAYES")
        print("="*60)

        start_time = time.time()

        # Ensure X is sparse CSR format
        if not issparse(X):
            X = csr_matrix(X)

        # Flatten y
        y = y.ravel()

        # Get unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, self.n_features_ = X.shape

        print(f"Classes: {n_classes}")
        print(f"Features: {self.n_features_}")
        print(f"Training samples: {n_samples:,}")
        print(f"Smoothing (alpha): {self.alpha}")

        # Initialize arrays
        self.class_prior_ = np.zeros(n_classes, dtype=np.float32)
        self.feature_log_prob_ = np.zeros((n_classes, self.n_features_), dtype=np.float32)

        # Calculate for each class separately
        for idx, c in enumerate(self.classes_):
            print(f"Processing class {c}...", end=" ")

            # Boolean mask for this class
            mask = (y == c)

            # Class prior
            self.class_prior_[idx] = mask.sum() / n_samples

            # Feature counts using sparse operations
            X_c = X[mask]
            feature_count = np.asarray(X_c.sum(axis=0)).ravel() + self.alpha

            # Calculate log probabilities
            total_count = feature_count.sum()
            self.feature_log_prob_[idx, :] = np.log(feature_count / total_count)

            print("✓")

            # Clear memory
            del X_c, feature_count, mask
            gc.collect()

        # Convert priors to log
        self.class_prior_ = np.log(self.class_prior_)

        training_time = time.time() - start_time
        print(f"\n✓ Training completed in {training_time:.2f} seconds")

        return self

    # Calculate log probabilities in batches
    def predict_log_proba_batch(self, X, batch_size=10000):
        if not issparse(X):
            X = csr_matrix(X)

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_proba = np.zeros((n_samples, n_classes), dtype=np.float32)

        # Process in batches
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            X_batch = X[start_idx:end_idx]

            # Calculate: log(P(class)) + sum(log(P(feature|class)) * feature_value)
            log_proba[start_idx:end_idx] = (
                X_batch @ self.feature_log_prob_.T + self.class_prior_
            )

        return log_proba

    # Predict class labels in batches
    def predict(self, X, batch_size=10000):
        log_proba = self.predict_log_proba_batch(X, batch_size)
        predictions = self.classes_[np.argmax(log_proba, axis=1)]

        # Clear memory
        del log_proba
        gc.collect()

        return predictions

    # Calculate probabilities in batches
    def predict_proba_batch(self, X, batch_size=10000):
        log_proba = self.predict_log_proba_batch(X, batch_size)

        # Convert to probabilities using log-sum-exp trick
        log_proba_max = log_proba.max(axis=1, keepdims=True)
        exp_log_proba = np.exp(log_proba - log_proba_max)
        proba = exp_log_proba / exp_log_proba.sum(axis=1, keepdims=True)

        # Clear memory
        del log_proba, log_proba_max, exp_log_proba
        gc.collect()

        return proba

# ============================================
# ROC COMPUTATION
# ============================================

# Compute ROC curve
def compute_roc_auc(y_true, y_proba, batch_size=50000):

    # Sample if dataset is too large
    n_samples = len(y_true)
    if n_samples > batch_size:
        print(f"  Sampling {batch_size:,} points for ROC calculation...")
        indices = np.random.choice(n_samples, batch_size, replace=False)
        y_true_sample = y_true[indices]
        y_proba_sample = y_proba[indices]
    else:
        y_true_sample = y_true
        y_proba_sample = y_proba

    fpr, tpr, _ = roc_curve(y_true_sample, y_proba_sample)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc

# Compute Precision-Recall curve
def compute_pr_curve(y_true, y_proba, batch_size=50000):

    # Sample if dataset is too large
    n_samples = len(y_true)
    if n_samples > batch_size:
        print(f"  Sampling {batch_size:,} points for PR calculation...")
        indices = np.random.choice(n_samples, batch_size, replace=False)
        y_true_sample = y_true[indices]
        y_proba_sample = y_proba[indices]
    else:
        y_true_sample = y_true
        y_proba_sample = y_proba

    precision, recall, _ = precision_recall_curve(y_true_sample, y_proba_sample)
    ap = average_precision_score(y_true_sample, y_proba_sample)

    return precision, recall, ap

# ============================================
# LOAD PREPROCESSED DATA
# ============================================
print("\n" + "="*60)
print("LOADING PREPROCESSED DATA")
print("="*60)

try:
    X_train_tfidf = load_npz(BASE_DIR / "data" / "X_train_tfidf.npz")
    y_train = np.load(BASE_DIR / "data" / "y_train.npy")
    X_test_tfidf = load_npz(BASE_DIR / "data" / "X_test_tfidf.npz")
    y_test = np.load(BASE_DIR / "data" / "y_test.npy")

    print(f"✅ Data loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Error: {e}. Run preprocess_sentiment_data.py first.")
    exit()

# ============================================
# TRAIN THE MODEL
# ============================================

print("\n" + "="*60)
print("INITIALIZING MODEL")
print("="*60)

# Clear memory before training
gc.collect()

# Initialize classifier
nb_classifier = NaiveBayes(alpha=0.5)

# Train the model
nb_classifier.fit(X_train_tfidf, y_train)

# Clear memory
gc.collect()

# ============================================
# MAKE PREDICTIONS (BATCH PROCESSING)
# ============================================

print("\n" + "="*60)
print("MAKING PREDICTIONS")
print("="*60)

BATCH_SIZE = 50000

start_time = time.time()

print("Predicting on training set...")
y_pred_train = nb_classifier.predict(X_train_tfidf, batch_size=BATCH_SIZE)

print("Predicting on test set...")
y_pred_test = nb_classifier.predict(X_test_tfidf, batch_size=BATCH_SIZE)

prediction_time = time.time() - start_time
print(f"✓ Predictions completed in {prediction_time:.2f} seconds")

# Get probabilities for positive class (for ROC/PR curves)
print("\nCalculating probabilities for test set...")
y_proba_test = nb_classifier.predict_proba_batch(X_test_tfidf, batch_size=BATCH_SIZE)[:, 1]

gc.collect()

# ============================================
# EVALUATION METRICS
# ============================================

print("\n" + "="*60)
print("EVALUATION METRICS")
print("="*60)

# Calculate metrics
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

train_precision = precision_score(y_train, y_pred_train, average='weighted')
test_precision = precision_score(y_test, y_pred_test, average='weighted')

train_recall = recall_score(y_train, y_pred_train, average='weighted')
test_recall = recall_score(y_test, y_pred_test, average='weighted')

train_f1 = f1_score(y_train, y_pred_train, average='weighted')
test_f1 = f1_score(y_test, y_pred_test, average='weighted')

# Print metrics
print("\n📊 TRAINING SET PERFORMANCE:")
print("-" * 60)
print(f"Accuracy:  {train_accuracy:.4f}")
print(f"Precision: {train_precision:.4f}")
print(f"Recall:    {train_recall:.4f}")
print(f"F1-Score:  {train_f1:.4f}")

print("\n📊 TEST SET PERFORMANCE:")
print("-" * 60)
print(f"Accuracy:  {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1-Score:  {test_f1:.4f}")

# Detailed classification report
print("\n📋 DETAILED CLASSIFICATION REPORT (TEST SET):")
print("-" * 60)
target_names = ['Negative', 'Positive']
print(classification_report(y_test, y_pred_test, target_names=target_names))

# ============================================
# CONFUSION MATRIX
# ============================================

print("\n" + "="*60)
print("CONFUSION MATRIX")
print("="*60)

# Calculate confusion matrices
cm = confusion_matrix(y_test, y_pred_test)

# Plot confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Raw counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names,
            cbar_kws={'label': 'Count'},
            ax=axes[0],
            annot_kws={'size': 14, 'weight': 'bold'})
axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold', pad=20)
axes[0].set_ylabel('True Label', fontsize=12)
axes[0].set_xlabel('Predicted Label', fontsize=12)

# Normalized
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names,
            cbar_kws={'label': 'Percentage'},
            ax=axes[1],
            annot_kws={'size': 14, 'weight': 'bold'})
axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold', pad=20)
axes[1].set_ylabel('True Label', fontsize=12)
axes[1].set_xlabel('Predicted Label', fontsize=12)

plt.tight_layout()
plt.savefig(SCRIPT_DIR / "nb_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()

print("✓ Confusion matrices plotted")

print("\nConfusion Matrix Details:")
print(f"  True Negatives:  {cm[0, 0]:,}")
print(f"  False Positives: {cm[0, 1]:,}")
print(f"  False Negatives: {cm[1, 0]:,}")
print(f"  True Positives:  {cm[1, 1]:,}")

tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

print(f"\n  Specificity: {specificity:.4f}")
print(f"  Sensitivity: {sensitivity:.4f}")

# ============================================
# ROC CURVE
# ============================================

print("\n" + "="*60)
print("ROC CURVE AND AUC")
print("="*60)

print("Computing ROC for test set...")
fpr_test, tpr_test, roc_auc_test = compute_roc_auc(
    y_test, y_proba_test, batch_size=50000
)

print(f"Test AUC:  {roc_auc_test:.4f}")

# Plot ROC curves
plt.figure(figsize=(10, 6))

plt.plot(fpr_test, tpr_test, color='darkorange', lw=2,
         label=f'ROC Curve (AUC = {roc_auc_test:.4f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--',
         label='Random Classifier (AUC = 0.50)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Naive Bayes Sentiment Classifier', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "nb_roc_curve.png", dpi=150, bbox_inches='tight')
plt.show()

print("✓ ROC curve plotted")

gc.collect()

# ============================================
# PRECISION-RECALL CURVE
# ============================================

print("\n" + "="*60)
print("PRECISION-RECALL CURVE")
print("="*60)

print("Computing PR curve for test set...")
precision_test, recall_test, ap_test = compute_pr_curve(
    y_test, y_proba_test, batch_size=50000
)

print(f"Test Average Precision:  {ap_test:.4f}")

# Plot Precision-Recall curves
plt.figure(figsize=(10, 6))

plt.plot(recall_test, precision_test, color='blue', lw=2,
         label=f'PR Curve (AP = {ap_test:.4f})')

# Baseline
baseline = y_test.sum() / len(y_test)
plt.plot([0, 1], [baseline, baseline], color='red', lw=2,
         linestyle='--', label=f'Baseline (AP = {baseline:.4f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower left", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "nb_pr_curve.png", dpi=150, bbox_inches='tight')
plt.show()

print("✓ Precision-Recall curve plotted")

gc.collect()

# ============================================
# PERFORMANCE COMPARISON
# ============================================

print("\n" + "="*60)
print("PERFORMANCE COMPARISON")
print("="*60)

# Create performance comparison plot
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
train_scores = [train_accuracy, train_precision, train_recall, train_f1]
test_scores = [test_accuracy, test_precision, test_recall, test_f1]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, train_scores, width, label='Train',
               color='skyblue', edgecolor='black')
bars2 = ax.bar(x + width/2, test_scores, width, label='Test',
               color='lightcoral', edgecolor='black')

ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Performance Metrics (Train vs Test)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(fontsize=11)
ax.set_ylim([0, 1.05])
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

print("✓ Performance comparison plotted")

# ============================================
# FINAL PERFORMANCE SUMMARY
# ============================================

print("\n" + "="*60)
print("📊 FINAL PERFORMANCE SUMMARY")
print("="*60)

print(f"\n{'METRIC':<20} {'TRAIN':<15} {'TEST':<15} {'DIFFERENCE':<15}")
print("-" * 65)
print(f"{'Accuracy':<20} {train_accuracy:<15.4f} {test_accuracy:<15.4f} {abs(train_accuracy-test_accuracy):<15.4f}")
print(f"{'Precision':<20} {train_precision:<15.4f} {test_precision:<15.4f} {abs(train_precision-test_precision):<15.4f}")
print(f"{'Recall':<20} {train_recall:<15.4f} {test_recall:<15.4f} {abs(train_recall-test_recall):<15.4f}")
print(f"{'F1-Score':<20} {train_f1:<15.4f} {test_f1:<15.4f} {abs(train_f1-test_f1):<15.4f}")
print(f"{'ROC AUC':<20} {'N/A':<15} {roc_auc_test:<15.4f} {'N/A':<15}")
print(f"{'Avg Precision':<20} {'N/A':<15} {ap_test:<15.4f} {'N/A':<15}")
print("-" * 65)

# Overfitting analysis
avg_diff = np.mean([
    abs(train_accuracy-test_accuracy),
    abs(train_precision-test_precision),
    abs(train_recall-test_recall),
    abs(train_f1-test_f1)
])

print(f"\n📈 MODEL ANALYSIS:")
print(f"   • Average Train-Test Difference: {avg_diff:.4f}")
if avg_diff < 0.02:
    print(f"   • Status: ✓ Excellent generalization")
elif avg_diff < 0.05:
    print(f"   • Status: ✓ Good generalization")
elif avg_diff < 0.10:
    print(f"   • Status: ⚠ Moderate overfitting")
else:
    print(f"   • Status: ⚠ Significant overfitting")

print(f"\n🎯 KEY FINDINGS:")
print(f"   • Test Accuracy: {test_accuracy:.2%}")
print(f"   • Test F1-Score: {test_f1:.4f}")
print(f"   • ROC AUC: {roc_auc_test:.4f}")
print(f"   • Model shows {'strong' if test_accuracy > 0.75 else 'moderate'} performance")

print("\n" + "="*60)
print("✓ ANALYSIS COMPLETE")
print("="*60)

# ============================================
# SAVE DASHBOARD JSON
# ============================================

def save_dashboard_json():
    report = classification_report(
        y_test, y_pred_test,
        target_names=['Negative', 'Positive'],
        output_dict=True,
    )

    # ROC curve — sample 12 points
    roc_indices = np.linspace(0, len(fpr_test) - 1, 12, dtype=int)
    roc_points  = [[round(float(fpr_test[i]), 3), round(float(tpr_test[i]), 3)] for i in roc_indices]

    # PR curve — sample 12 points
    pr_indices = np.linspace(0, len(precision_test) - 1, 12, dtype=int)
    pr_points  = [[round(float(recall_test[i]), 3), round(float(precision_test[i]), 3)] for i in pr_indices]

    tn, fp, fn, tp = cm.ravel()

    dashboard = {
        'model'     : 'Naive Bayes',
        'dataset'   : 'Sentiment140',
        'accuracy'  : round(test_accuracy * 100, 2),
        'precision' : round(report['weighted avg']['precision'] * 100, 2),
        'recall'    : round(report['weighted avg']['recall'] * 100, 2),
        'f1'        : round(report['weighted avg']['f1-score'] * 100, 2),
        'auc_roc'   : round(float(roc_auc_test), 4),
        'auc_pr'    : round(float(ap_test), 4),
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
        'training_history': None,   # NB trains in one pass — no epoch history
        'dataset_stats': {
            'train_samples': int(X_train_tfidf.shape[0]),
            'test_samples' : int(X_test_tfidf.shape[0]),
            'n_features'   : int(X_train_tfidf.shape[1]),
        },
    }

    with open(SCRIPT_DIR / "nb_sentiment140_dashboard.json", 'w') as f:
        json.dump(dashboard, f, indent=2)
    print('✅ Dashboard JSON saved → nb_sentiment140_dashboard.json')


save_dashboard_json()

# Final cleanup
gc.collect()
