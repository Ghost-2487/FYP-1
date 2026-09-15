# ============================================
# SVM USING SKLEARN
# ============================================

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report,
    precision_recall_curve, average_precision_score
)
from scipy.sparse import load_npz
import time
import gc
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

print("="*70)
print("SUPPORT VECTOR MACHINE (SKLEARN)")
print("="*70)

# ============================================
# 1. LOAD PREPROCESSED DATA
# ============================================
print("\n" + "="*70)
print("STEP 1: LOADING PREPROCESSED DATA")
print("="*70)

try:
    X_train_tfidf = load_npz(BASE_DIR / "data" / "X_train_tfidf.npz")
    y_train = np.load(BASE_DIR / "data" / "y_train.npy")
    X_test_tfidf = load_npz(BASE_DIR / "data" / "X_test_tfidf.npz")
    y_test = np.load(BASE_DIR / "data" / "y_test.npy")

    print(f"✅ Data loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Error: {e}. Run preprocess_sentiment_data.py first.")
    exit()

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================
# 2. TRAIN MODEL
# ============================================

print("\n" + "="*70)
print("STEP 2: TRAIN SVM MODEL")
print("="*70)

# Clear memory before training
gc.collect()

print("\n🚀 Training started...")
print(f"   Training samples: {X_train_tfidf.shape[0]:,}")
print(f"   Features: {X_train_tfidf.shape[1]:,}")
print(f"   Algorithm: LinearSVC (Linear Support Vector Classification)")
print("-" * 70)

start_time = time.time()

# Initialize LinearSVC
svm_base = LinearSVC(
    C=0.1,
    class_weight='balanced',
    max_iter=3000,
    random_state=42,
    dual=False,
    verbose=0
)

print("\n🚀 Training SVM...")
svm_base.fit(X_train_tfidf, y_train.ravel())

training_time = time.time() - start_time
print(f"✅ Training completed in {training_time:.2f} seconds")

# Calibrate for probability estimates
print("\n📊 Calibrating for probability estimates...")
calibration_start = time.time()

svm_model = CalibratedClassifierCV(
    estimator=FrozenEstimator(svm_base),
    method='sigmoid'
)

# Calibrate on a subset
sample_size = min(100000, len(y_train))
if len(y_train) > sample_size:
    print(f"   Using {sample_size:,} samples for calibration")
    indices = np.random.choice(len(y_train), sample_size, replace=False)
    X_calibrate = X_train_tfidf[indices]
    y_calibrate = y_train[indices]
else:
    X_calibrate = X_train_tfidf
    y_calibrate = y_train

svm_model.fit(X_calibrate, y_calibrate.ravel())

calibration_time = time.time() - calibration_start
print(f"✅ Calibration completed in {calibration_time:.2f} seconds")

# Clear memory
gc.collect()

# ============================================
# 3. MAKE PREDICTIONS
# ============================================

print("\n" + "="*70)
print("STEP 3: MAKING PREDICTIONS")
print("="*70)

# Predict on training set
print("\n📊 Predicting on training set...")
pred_start = time.time()
# For prefit CalibratedClassifierCV, predict uses the calibrated probabilities
y_train_pred = svm_model.predict(X_train_tfidf)
print("📊 Calculating probabilities for training set...")
y_train_proba = svm_model.predict_proba(X_train_tfidf)[:, 1]
train_pred_time = time.time() - pred_start
print(f"   Training predictions: {train_pred_time:.2f}s")
gc.collect()

# Predict on test set
print("📊 Predicting on test set...")
pred_start = time.time()
y_test_pred = svm_model.predict(X_test_tfidf)
print("📊 Calculating probabilities for test set...")
y_test_proba = svm_model.predict_proba(X_test_tfidf)[:, 1]
test_pred_time = time.time() - pred_start
print(f"   Test predictions: {test_pred_time:.2f}s")
gc.collect()

print("✅ Predictions completed!")

# ============================================
# 4. EVALUATION METRICS
# ============================================

print("\n" + "="*70)
print("STEP 4: MODEL EVALUATION - TRAINING SET")
print("="*70)

train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

print(f"\n📈 Training Set Performance:")
print(f"   Accuracy:  {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"   Precision: {train_precision:.4f}")
print(f"   Recall:    {train_recall:.4f}")
print(f"   F1-Score:  {train_f1:.4f}")

print("\n" + "="*70)
print("STEP 5: MODEL EVALUATION - TEST SET")
print("="*70)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"\n📈 Test Set Performance:")
print(f"   Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"   Precision: {test_precision:.4f}")
print(f"   Recall:    {test_recall:.4f}")
print(f"   F1-Score:  {test_f1:.4f}")

print("\n" + "="*70)
print("DETAILED CLASSIFICATION REPORT")
print("="*70)
print("\n", classification_report(y_test, y_test_pred, target_names=['Negative', 'Positive'], digits=4))

# ============================================
# 5. CONFUSION MATRIX
# ============================================

print("\n" + "="*70)
print("STEP 6: CONFUSION MATRIX")
print("="*70)

cm = confusion_matrix(y_test, y_test_pred)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Raw counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            cbar_kws={'label': 'Count'},
            ax=axes[0],
            annot_kws={'size': 14, 'weight': 'bold'})
axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold', pad=20)
axes[0].set_ylabel('True Label', fontsize=12)
axes[0].set_xlabel('Predicted Label', fontsize=12)

# Normalized
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Oranges',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            cbar_kws={'label': 'Percentage'},
            ax=axes[1],
            annot_kws={'size': 14, 'weight': 'bold'})
axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold', pad=20)
axes[1].set_ylabel('True Label', fontsize=12)
axes[1].set_xlabel('Predicted Label', fontsize=12)

plt.tight_layout()
plt.savefig(SCRIPT_DIR / "svm_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()

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
# 6. ROC CURVE AND AUC
# ============================================

print("\n" + "="*70)
print("STEP 7: ROC CURVE & AUC SCORE")
print("="*70)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
roc_auc = auc(fpr, tpr)

print(f"\n🎯 ROC-AUC Score: {roc_auc:.4f}")

if roc_auc >= 0.90:
    interpretation = "Excellent"
elif roc_auc >= 0.80:
    interpretation = "Good"
elif roc_auc >= 0.70:
    interpretation = "Fair"
else:
    interpretation = "Poor"
print(f"   Interpretation: {interpretation}")

# Plot ROC curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='red', lw=3,
         label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
         label='Random Classifier (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('Receiver Operating Characteristic (ROC) Curve',
          fontsize=14, fontweight='bold', pad=20)
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "svm_roc_curve.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# 7. PRECISION-RECALL CURVE
# ============================================

print("\n" + "="*70)
print("STEP 8: PRECISION-RECALL CURVE")
print("="*70)

precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_test_proba)
average_precision = average_precision_score(y_test, y_test_proba)

print(f"\n🎯 Average Precision Score: {average_precision:.4f}")

plt.figure(figsize=(10, 8))
plt.plot(recall_curve, precision_curve, color='purple', lw=3,
         label=f'PR Curve (AP = {average_precision:.4f})')
plt.axhline(y=np.sum(y_test)/len(y_test), color='navy', lw=2,
            linestyle='--', label=f'Baseline')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=12, fontweight='bold')
plt.ylabel('Precision', fontsize=12, fontweight='bold')
plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold', pad=20)
plt.legend(loc="lower left", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "svm_pr_curve.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# 8. PERFORMANCE SUMMARY
# ============================================

print("\n" + "="*70)
print("STEP 9: FINAL PERFORMANCE SUMMARY")
print("="*70)

summary_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Avg Precision'],
    'Training': [f"{train_accuracy:.4f}", f"{train_precision:.4f}",
                 f"{train_recall:.4f}", f"{train_f1:.4f}", "N/A", "N/A"],
    'Test': [f"{test_accuracy:.4f}", f"{test_precision:.4f}",
             f"{test_recall:.4f}", f"{test_f1:.4f}",
             f"{roc_auc:.4f}", f"{average_precision:.4f}"]
}

summary_df = pd.DataFrame(summary_data)
print("\n", summary_df.to_string(index=False))

# Visualize metrics comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
train_scores = [train_accuracy, train_precision, train_recall, train_f1]
test_scores = [test_accuracy, test_precision, test_recall, test_f1]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, train_scores, width, label='Training',
               color='lightpink', edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, test_scores, width, label='Test',
               color='lightcoral', edgecolor='black', linewidth=1.5)

ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Performance: Training vs Test', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim([0, 1.1])
ax.grid(True, axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================
# 9. MODEL ANALYSIS
# ============================================

print("\n" + "="*70)
print("STEP 10: MODEL ANALYSIS")
print("="*70)

# Overfitting analysis
avg_diff = np.mean([
    abs(train_accuracy - test_accuracy),
    abs(train_precision - test_precision),
    abs(train_recall - test_recall),
    abs(train_f1 - test_f1)
])

print(f"\n📈 Generalization Analysis:")
print(f"   • Average Train-Test Difference: {avg_diff:.4f}")

if avg_diff < 0.02:
    print(f"   • Status: ✓ Excellent generalization (minimal overfitting)")
elif avg_diff < 0.05:
    print(f"   • Status: ✓ Good generalization (slight overfitting)")
elif avg_diff < 0.10:
    print(f"   • Status: ⚠ Moderate overfitting")
else:
    print(f"   • Status: ⚠ Significant overfitting detected")

print(f"\n🎯 Key Findings:")
print(f"   • Test Accuracy: {test_accuracy:.2%}")
print(f"   • Test F1-Score: {test_f1:.4f}")
print(f"   • ROC-AUC: {roc_auc:.4f}")
print(f"   • Model demonstrates {'strong' if test_accuracy > 0.75 else 'moderate'} performance")

print(f"\n💾 Implementation Details:")
print(f"   • Algorithm: LinearSVC (sklearn)")
print(f"   • Probability calibration: Platt scaling (sigmoid)")
print(f"   • Total training time: {training_time:.2f}s")

print(f"\n🔧 Hyperparameters Used:")
print(f"   • C (regularization): {svm_base.C}")
print(f"   • Max iterations: {svm_base.max_iter}")
print(f"   • Dual formulation: {svm_base.dual}")
print(f"   • Calibration method: sigmoid (Platt scaling)")
print(f"   • Calibration CV folds: prefit (Single Model)")

# Get support vectors info (if available)
try:
    n_support = len(svm_base.coef_[0])
    print(f"\n📊 Model Complexity:")
    print(f"   • Feature weights computed: {n_support:,}")
    print(f"   • Non-zero weights: {np.count_nonzero(svm_base.coef_):,}")
except:
    pass

print("\n" + "="*70)
print("✅ SVM COMPLETE!")
print("="*70)

# ============================================
# 10. SAVE DASHBOARD JSON
# ============================================

print("\n" + "="*70)
print("STEP 11: SAVE DASHBOARD JSON")
print("="*70)

def save_dashboard_json():
    report = classification_report(
        y_test, y_test_pred,
        target_names=['Negative', 'Positive'],
        output_dict=True,
    )

    # ROC curve — sample 12 points
    roc_indices = np.linspace(0, len(fpr) - 1, 12, dtype=int)
    roc_points  = [[round(float(fpr[i]), 3), round(float(tpr[i]), 3)] for i in roc_indices]

    # PR curve — sample 12 points
    pr_indices = np.linspace(0, len(precision_curve) - 1, 12, dtype=int)
    pr_points  = [[round(float(recall_curve[i]), 3), round(float(precision_curve[i]), 3)] for i in pr_indices]

    tn, fp, fn, tp = cm.ravel()

    dashboard = {
        'model'     : 'SVM',
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
        'training_history': None,   # SVM trains in one pass — no epoch history
        'dataset_stats': {
            'train_samples': int(X_train_tfidf.shape[0]),
            'test_samples' : int(X_test_tfidf.shape[0]),
            'n_features'   : int(X_train_tfidf.shape[1]),
        },
    }

    with open(SCRIPT_DIR / "svm_sentiment140_dashboard.json", 'w') as f:
        json.dump(dashboard, f, indent=2)
    print('✅ Dashboard JSON saved → svm_sentiment140_dashboard.json')


save_dashboard_json()

# Final cleanup
gc.collect()
