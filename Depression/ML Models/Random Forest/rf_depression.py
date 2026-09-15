# ============================================
# RANDOM FOREST
# ============================================

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
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

print("="*70)
print("RANDOM FOREST")
print("="*70)

# ============================================
# 2. CONVERT SPARSE TO DENSE (SAMPLED)
# ============================================
# RandomForestClassifier works with sparse input but OOB scoring
# requires array-compatible format. We convert to float32 array
# in chunks.

print("\n" + "="*70)
print("STEP 2: PREPARING DATA")
print("="*70)

print(f"\n📊 Dataset Overview:")
print(f"   Training samples: {X_train_tfidf.shape[0]:,}")
print(f"   Test samples:     {X_test_tfidf.shape[0]:,}")
print(f"   Features:         {X_train_tfidf.shape[1]:,}")

MAX_TRAIN_SAMPLES = 200000
np.random.seed(42)

y_train_flat = y_train.ravel()
y_test_flat  = y_test.ravel()

if X_train_tfidf.shape[0] > MAX_TRAIN_SAMPLES:
    print(f"\n⚡ Sub-sampling {MAX_TRAIN_SAMPLES:,} training examples (stratified)...")
    # Stratified sampling manually
    neg_idx = np.where(y_train_flat == 0)[0]
    pos_idx = np.where(y_train_flat == 1)[0]
    half = MAX_TRAIN_SAMPLES // 2
    sampled_neg = np.random.choice(neg_idx, half, replace=False)
    sampled_pos = np.random.choice(pos_idx, half, replace=False)
    sample_idx  = np.concatenate([sampled_neg, sampled_pos])
    np.random.shuffle(sample_idx)

    X_train_rf = X_train_tfidf[sample_idx]
    y_train_rf = y_train_flat[sample_idx]
    print(f"   Sampled: {X_train_rf.shape[0]:,} samples  "
          f"(Neg: {half:,}, Pos: {half:,})")
else:
    X_train_rf = X_train_tfidf
    y_train_rf = y_train_flat
    print("   Using full training set.")

X_test_rf  = X_test_tfidf
y_test_rf  = y_test_flat

gc.collect()

# ============================================
# 3. EARLY STOPPING VIA WARM_START + OOB ERROR
# ============================================

print("\n" + "="*70)
print("STEP 3: EARLY STOPPING SETUP")
print("="*70)

"""
   Early Stopping Strategy:
   • Method: warm_start=True + oob_score=True
   • Grow the forest incrementally (step_size trees per round)
   • Monitor OOB (Out-of-Bag) error after each round
   • Stop when OOB error does not improve for 'patience' consecutive rounds
   • Best model (lowest OOB error) is retained automatically
"""

# Hyperparameters
N_ESTIMATORS_START  = 50     # Initial number of trees (≥50 for reliable OOB)
STEP_SIZE           = 10     # Trees added per round
MAX_ESTIMATORS      = 300    # Hard cap on total trees
PATIENCE            = 3      # Rounds without improvement before stopping
MIN_IMPROVEMENT     = 1e-4   # Minimum OOB gain to count as improvement

print(f"   Start trees:     {N_ESTIMATORS_START} (minimum for reliable OOB)")
print(f"   Step size:       {STEP_SIZE} trees/round")
print(f"   Max trees:       {MAX_ESTIMATORS}")
print(f"   Patience:        {PATIENCE} rounds")
print(f"   Min improvement: {MIN_IMPROVEMENT}")

# ============================================
# 4. TRAIN WITH EARLY STOPPING
# ============================================

print("\n" + "="*70)
print("STEP 4: TRAINING RANDOM FOREST")
print("="*70)

rf_model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS_START,
    max_features='sqrt',          # Standard for classification
    max_depth=None,               # Trees grow fully — forest controls variance
    min_samples_leaf=5,           # Slight regularization to reduce overfitting
    # class_weight omitted: training set is already stratified 50/50,
    # and 'balanced' conflicts with warm_start when data is subsampled.
    oob_score=True,
    warm_start=True,              # Key: reuse existing trees, just add more
    n_jobs=-1,
    random_state=42
)

oob_errors   = []
n_trees_list = []
best_oob_err = np.inf
patience_counter = 0
best_n_estimators = N_ESTIMATORS_START

print(f"\n🚀 Training started...")
print(f"   Using warm_start: growing forest incrementally")
print("-" * 70)
print(f"{'Round':<8} {'Trees':<10} {'OOB Error':<14} {'OOB Acc':<12} {'Elapsed':<10} {'Status'}")
print("-" * 70)

total_start = time.time()

current_n = N_ESTIMATORS_START
while current_n <= MAX_ESTIMATORS:
    round_start = time.time()

    rf_model.n_estimators = current_n
    rf_model.fit(X_train_rf, y_train_rf)

    oob_err = 1.0 - rf_model.oob_score_
    oob_acc = rf_model.oob_score_
    elapsed = time.time() - total_start

    oob_errors.append(oob_err)
    n_trees_list.append(current_n)

    round_num = len(oob_errors)

    # Check improvement
    if oob_err < best_oob_err - MIN_IMPROVEMENT:
        improvement = best_oob_err - oob_err
        best_oob_err = oob_err
        best_n_estimators = current_n
        patience_counter = 0
        status = f"✅ Best (↓{improvement:.5f})"
    else:
        patience_counter += 1
        status = f"⚠ No improvement [{patience_counter}/{PATIENCE}]"

    round_time = time.time() - round_start
    print(f"{round_num:<8} {current_n:<10} {oob_err:<14.6f} {oob_acc:<12.4f} "
          f"{elapsed:<10.1f}s {status}")

    # Early stopping check
    if patience_counter >= PATIENCE:
        print(f"\n🛑 Early stopping triggered after {current_n} trees.")
        print(f"   No OOB improvement for {PATIENCE} consecutive rounds.")
        break

    current_n += STEP_SIZE

training_time = time.time() - total_start

print("-" * 70)
print(f"\n✅ Training completed in {training_time:.2f} seconds")
print(f"   Best n_estimators: {best_n_estimators}")
print(f"   Best OOB Error:    {best_oob_err:.6f}")
print(f"   Best OOB Accuracy: {1 - best_oob_err:.4f}")

# Re-train with the best number of estimators (clean model, no warm_start)
print(f"\n🔄 Re-fitting with best n_estimators={best_n_estimators} for final model...")
refit_start = time.time()
rf_final = RandomForestClassifier(
    n_estimators=best_n_estimators,
    max_features='sqrt',
    max_depth=None,
    min_samples_leaf=5,
    # class_weight omitted: same reason as above
    oob_score=True,
    n_jobs=-1,
    random_state=42
)
rf_final.fit(X_train_rf, y_train_rf)
refit_time = time.time() - refit_start
print(f"✅ Final model fitted in {refit_time:.2f} seconds")
print(f"   Final OOB Score: {rf_final.oob_score_:.4f}")

gc.collect()

# ============================================
# 5. PLOT EARLY STOPPING CURVE
# ============================================

print("\n" + "="*70)
print("STEP 5: EARLY STOPPING CURVE")
print("="*70)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(n_trees_list, oob_errors, color='steelblue', lw=2.5,
        marker='o', markersize=6, label='OOB Error')
ax.axvline(x=best_n_estimators, color='red', lw=2, linestyle='--',
           label=f'Best: {best_n_estimators} trees (OOB Err={best_oob_err:.4f})')
ax.fill_between(n_trees_list, oob_errors,
                alpha=0.15, color='steelblue')
ax.set_xlabel('Number of Trees', fontsize=12, fontweight='bold')
ax.set_ylabel('OOB Error (1 - OOB Accuracy)', fontsize=12, fontweight='bold')
ax.set_title('Random Forest Early Stopping — OOB Error vs. Trees',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "rf_early_stopping_curve.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ Early stopping curve plotted")

# ============================================
# 6. FEATURE IMPORTANCE (TOP 20)
# ============================================

print("\n" + "="*70)
print("STEP 6: FEATURE IMPORTANCE")
print("="*70)

importances = rf_final.feature_importances_
top_n = 20
top_idx = np.argsort(importances)[::-1][:top_n]
top_importances = importances[top_idx]

print(f"\n📊 Top {top_n} most important features (by Gini importance):")
for i, (idx, imp) in enumerate(zip(top_idx, top_importances), 1):
    print(f"   {i:2d}. Feature {idx:6d}: {imp:.6f}")

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(range(top_n), top_importances[::-1],
               color='steelblue', edgecolor='black', linewidth=0.8)
ax.set_yticks(range(top_n))
ax.set_yticklabels([f'Feature {idx}' for idx in top_idx[::-1]], fontsize=9)
ax.set_xlabel('Gini Importance', fontsize=12, fontweight='bold')
ax.set_title(f'Top {top_n} Feature Importances — Random Forest',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, axis='x', alpha=0.3)

for bar, val in zip(bars, top_importances[::-1]):
    ax.text(bar.get_width() + 0.0001, bar.get_y() + bar.get_height()/2,
            f'{val:.5f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(SCRIPT_DIR / "rf_feature_importance.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# 7. MAKE PREDICTIONS
# ============================================

print("\n" + "="*70)
print("STEP 7: MAKING PREDICTIONS")
print("="*70)

print("\n📊 Predicting on training set...")
pred_start = time.time()
y_train_pred  = rf_final.predict(X_train_rf)
y_train_proba = rf_final.predict_proba(X_train_rf)[:, 1]
print(f"   Training predictions: {time.time() - pred_start:.2f}s")
gc.collect()

print("📊 Predicting on test set...")
pred_start = time.time()
y_test_pred  = rf_final.predict(X_test_rf)
y_test_proba = rf_final.predict_proba(X_test_rf)[:, 1]
print(f"   Test predictions: {time.time() - pred_start:.2f}s")
gc.collect()

print("✅ Predictions completed!")

# ============================================
# 8. EVALUATION METRICS
# ============================================

print("\n" + "="*70)
print("STEP 8: MODEL EVALUATION - TRAINING SET")
print("="*70)

train_accuracy  = accuracy_score(y_train_rf, y_train_pred)
train_precision = precision_score(y_train_rf, y_train_pred)
train_recall    = recall_score(y_train_rf, y_train_pred)
train_f1        = f1_score(y_train_rf, y_train_pred)

print(f"\n📈 Training Set Performance:")
print(f"   Accuracy:  {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"   Precision: {train_precision:.4f}")
print(f"   Recall:    {train_recall:.4f}")
print(f"   F1-Score:  {train_f1:.4f}")

print("\n" + "="*70)
print("STEP 9: MODEL EVALUATION - TEST SET")
print("="*70)

test_accuracy  = accuracy_score(y_test_rf, y_test_pred)
test_precision = precision_score(y_test_rf, y_test_pred)
test_recall    = recall_score(y_test_rf, y_test_pred)
test_f1        = f1_score(y_test_rf, y_test_pred)

print(f"\n📈 Test Set Performance:")
print(f"   Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"   Precision: {test_precision:.4f}")
print(f"   Recall:    {test_recall:.4f}")
print(f"   F1-Score:  {test_f1:.4f}")

print("\n" + "="*70)
print("DETAILED CLASSIFICATION REPORT")
print("="*70)
print("\n", classification_report(y_test_rf, y_test_pred, target_names=['Negative', 'Positive'], digits=4))

# ============================================
# 9. CONFUSION MATRIX
# ============================================

print("\n" + "="*70)
print("STEP 10: CONFUSION MATRIX")
print("="*70)

cm = confusion_matrix(y_test_rf, y_test_pred)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Raw counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
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
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='YlGn',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            cbar_kws={'label': 'Percentage'},
            ax=axes[1],
            annot_kws={'size': 14, 'weight': 'bold'})
axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold', pad=20)
axes[1].set_ylabel('True Label', fontsize=12)
axes[1].set_xlabel('Predicted Label', fontsize=12)

plt.tight_layout()
plt.savefig(SCRIPT_DIR / "rf_confusion_matrix.png", dpi=150, bbox_inches='tight')
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
# 10. ROC CURVE AND AUC
# ============================================

print("\n" + "="*70)
print("STEP 11: ROC CURVE & AUC SCORE")
print("="*70)

fpr, tpr, thresholds = roc_curve(y_test_rf, y_test_proba)
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

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='forestgreen', lw=3,
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
plt.savefig(SCRIPT_DIR / "rf_roc_curve.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# 11. PRECISION-RECALL CURVE
# ============================================

print("\n" + "="*70)
print("STEP 12: PRECISION-RECALL CURVE")
print("="*70)

precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test_rf, y_test_proba)
average_precision = average_precision_score(y_test_rf, y_test_proba)

print(f"\n🎯 Average Precision Score: {average_precision:.4f}")

plt.figure(figsize=(10, 8))
plt.plot(recall_curve, precision_curve, color='darkgreen', lw=3,
         label=f'PR Curve (AP = {average_precision:.4f})')
plt.axhline(y=np.sum(y_test_rf) / len(y_test_rf), color='navy', lw=2,
            linestyle='--', label='Baseline')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=12, fontweight='bold')
plt.ylabel('Precision', fontsize=12, fontweight='bold')
plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold', pad=20)
plt.legend(loc="lower left", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "rf_pr_curve.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# 12. PERFORMANCE SUMMARY
# ============================================

print("\n" + "="*70)
print("STEP 13: PERFORMANCE SUMMARY")
print("="*70)

summary_data = {
    'Metric':   ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Avg Precision'],
    'Training': [f"{train_accuracy:.4f}", f"{train_precision:.4f}",
                 f"{train_recall:.4f}", f"{train_f1:.4f}", "N/A", "N/A"],
    'Test':     [f"{test_accuracy:.4f}", f"{test_precision:.4f}",
                 f"{test_recall:.4f}", f"{test_f1:.4f}",
                 f"{roc_auc:.4f}", f"{average_precision:.4f}"]
}

summary_df = pd.DataFrame(summary_data)
print("\n", summary_df.to_string(index=False))

# Bar chart: train vs test
metrics_list  = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
train_scores  = [train_accuracy, train_precision, train_recall, train_f1]
test_scores   = [test_accuracy, test_precision, test_recall, test_f1]

x = np.arange(len(metrics_list))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, train_scores, width, label='Training',
               color='mediumseagreen', edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, test_scores, width, label='Test',
               color='lightgreen', edgecolor='black', linewidth=1.5)

ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Performance: Training vs Test', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics_list, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim([0, 1.1])
ax.grid(True, axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================
# 13. MODEL ANALYSIS
# ============================================

print("\n" + "="*70)
print("STEP 14: MODEL ANALYSIS")
print("="*70)

avg_diff = np.mean([
    abs(train_accuracy  - test_accuracy),
    abs(train_precision - test_precision),
    abs(train_recall    - test_recall),
    abs(train_f1        - test_f1)
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
print(f"   • ROC-AUC:       {roc_auc:.4f}")
print(f"   • OOB Accuracy:  {rf_final.oob_score_:.4f}")
print(f"   • Model demonstrates {'strong' if test_accuracy > 0.75 else 'moderate'} performance")

print(f"\n💾 Implementation Details:")
print(f"   • Algorithm:          RandomForestClassifier (sklearn)")
print(f"   • Early stopping:     warm_start + OOB error monitoring")
print(f"   • Best n_estimators:  {best_n_estimators}")
print(f"   • Total training time:{training_time:.2f}s")
print(f"   • Training samples:   {X_train_rf.shape[0]:,}")

print(f"\n🔧 Hyperparameters Used:")
print(f"   • n_estimators:    {best_n_estimators} (early-stopped)")
print(f"   • max_features:    sqrt")
print(f"   • max_depth:       None (fully grown trees)")
print(f"   • min_samples_leaf:{rf_final.min_samples_leaf}")
print(f"   • class_weight:    None (data already balanced 50/50)")
print(f"   • Early stop step: {STEP_SIZE} trees/round")
print(f"   • Patience:        {PATIENCE} rounds")

print(f"\n📊 Forest Statistics:")
print(f"   • Number of trees:      {rf_final.n_estimators}")
print(f"   • Number of features:   {rf_final.n_features_in_:,}")
print(f"   • Features per split:   ~{int(np.sqrt(rf_final.n_features_in_)):,} (sqrt)")
print(f"   • OOB Score:            {rf_final.oob_score_:.4f}")

print("\n" + "="*70)
print("✅ RANDOM FOREST COMPLETE!")
print("="*70)

# ============================================
# 14. SAVE DASHBOARD JSON
# ============================================

print("\n" + "="*70)
print("STEP 15: SAVING DASHBOARD JSON")
print("="*70)

def save_dashboard_json():
    report = classification_report(
        y_test_rf, y_test_pred,
        target_names=['Negative', 'Positive'],
        output_dict=True,
    )

    # ROC curve — sample 12 points
    roc_indices = np.linspace(0, len(fpr) - 1, 12, dtype=int)
    roc_points  = [[round(float(fpr[i]), 3), round(float(tpr[i]), 3)] for i in roc_indices]

    # PR curve — sample 12 points
    pr_indices = np.linspace(0, len(precision_curve) - 1, 12, dtype=int)
    pr_points  = [[round(float(recall_curve[i]), 3), round(float(precision_curve[i]), 3)] for i in pr_indices]

    # OOB training history (serves as training curve)
    training_history = {
        'n_trees'    : [int(n) for n in n_trees_list],
        'oob_error'  : [round(float(e), 6) for e in oob_errors],
        'oob_accuracy': [round(1 - float(e), 6) for e in oob_errors],
    }

    tn, fp, fn, tp = cm.ravel()

    dashboard = {
        'model'     : 'Random Forest',
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
        'training_history': training_history,   # OOB error per round
        'model_stats': {
            'best_n_estimators' : best_n_estimators,
            'best_oob_error'    : round(float(best_oob_err), 6),
            'best_oob_accuracy' : round(float(1 - best_oob_err), 4),
            'final_oob_score'   : round(float(rf_final.oob_score_), 4),
            'train_samples'     : int(X_train_rf.shape[0]),
            'test_samples'      : int(X_test_rf.shape[0]),
            'n_features'        : int(X_train_rf.shape[1]),
        },
    }

    with open(SCRIPT_DIR / "rf_sentiment140_dashboard.json", 'w') as f:
        json.dump(dashboard, f, indent=2)
    print('✅ Dashboard JSON saved → rf_sentiment140_dashboard.json')


save_dashboard_json()

# Final cleanup
gc.collect()