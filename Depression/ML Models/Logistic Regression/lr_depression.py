# ============================================
#  LOGISTIC REGRESSION
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("LOGISTIC REGRESSION")
print("="*70)

# ============================================
# 1. LOGISTIC REGRESSION CLASS
# ============================================

class LogisticRegression:

    def __init__(self, learning_rate=0.1, n_iterations=1000, batch_size=10000, verbose=True):
        """
        Initialize model with mini-batch gradient descent

        Parameters:
        -----------
        learning_rate : float
            Learning rate for gradient descent
        n_iterations : int
            Number of iterations (epochs)
        batch_size : int
            Size of mini-batches for gradient descent
        verbose : bool
            Whether to print training progress
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.losses = []

    # Sigmoid activation with numerical stability
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    # Compute binary cross-entropy loss
    def compute_loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def fit(self, X, y):
        """
        Train using mini-batch gradient descent

        Parameters:
        -----------
        X : sparse matrix, shape (n_samples, n_features)
            Training data (TF-IDF sparse matrix)
        y : array-like, shape (n_samples,)
            Target values
        """
        # Keep X as sparse matrix
        n_samples, n_features = X.shape
        y = np.array(y, dtype=np.float32).reshape(-1, 1)

        # Initialize weights (keep as regular array, not sparse)
        self.weights = np.zeros((n_features, 1), dtype=np.float32)
        self.bias = 0.0

        print(f"\n🚀 Training started...")
        print(f"   Samples: {n_samples:,}, Features: {n_features:,}")
        print(f"   Batch size: {self.batch_size:,}")
        print(f"   Learning rate: {self.learning_rate}, Iterations: {self.n_iterations}")
        print("-" * 70)

        start_time = time.time()
        n_batches = int(np.ceil(n_samples / self.batch_size))

        # Training loop
        for epoch in range(self.n_iterations):
            epoch_loss = 0.0
            indices = np.random.permutation(n_samples)

            # Mini-batch gradient descent
            for batch_idx in range(n_batches):
                # Get batch indices
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]

                # Get batch data (X_batch stays sparse)
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                # Forward pass (sparse matrix multiplication)
                linear_model = X_batch.dot(self.weights) + self.bias
                y_pred = self.sigmoid(linear_model)

                # Compute loss
                batch_loss = self.compute_loss(y_batch, y_pred)
                epoch_loss += batch_loss * len(batch_indices)

                # Backward pass (compute gradients)
                error = y_pred - y_batch
                dw = (X_batch.T.dot(error)) / len(batch_indices)  # Sparse matrix operation
                db = np.mean(error)

                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            # Average loss for the epoch
            avg_loss = epoch_loss / n_samples
            self.losses.append(avg_loss)

            # Print progress
            if self.verbose and (epoch % 50 == 0 or epoch == self.n_iterations - 1):
                elapsed = time.time() - start_time
                print(f"Epoch {epoch:4d}/{self.n_iterations} | "
                      f"Loss: {avg_loss:.6f} | "
                      f"Time: {elapsed:.2f}s")

            # Garbage collection every 100 epochs
            if epoch % 100 == 0:
                gc.collect()

        training_time = time.time() - start_time
        print("-" * 70)
        print(f"✅ Training completed in {training_time:.2f} seconds")
        print(f"   Final loss: {self.losses[-1]:.6f}")

    # Predict probabilities in batches
    def predict_proba(self, X, batch_size=50000):
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        predictions = np.zeros(n_samples, dtype=np.float32)

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)

            X_batch = X[start_idx:end_idx]
            linear_model = X_batch.dot(self.weights) + self.bias
            predictions[start_idx:end_idx] = self.sigmoid(linear_model).flatten()

        return predictions

    # Predict class labels
    def predict(self, X, threshold=0.5, batch_size=50000):
        probabilities = self.predict_proba(X, batch_size)
        return (probabilities >= threshold).astype(int)

    # Plot training loss curve
    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses, linewidth=2, color='blue')
        plt.title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Binary Cross-Entropy Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(SCRIPT_DIR / "lr_training_history.png", dpi=150, bbox_inches='tight')
        plt.show()

# ============================================
# 2. LOAD PREPROCESSED DATA
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

# ============================================
# 3. TRAIN MODEL
# ============================================

print("\n" + "="*70)
print("STEP 2: TRAIN LOGISTIC REGRESSION MODEL")
print("="*70)

# Initialize model with parameters
lr_model = LogisticRegression(
    learning_rate=0.5,       # Higher learning rate for faster convergence
    n_iterations=1000,        # Fewer iterations needed with mini-batch
    batch_size=5000,        # Process 10k samples at a time
    verbose=True
)

# Train model
lr_model.fit(X_train_tfidf, y_train)

# Plot training loss
lr_model.plot_loss()

# Garbage collection after training
gc.collect()

# ============================================
# 4. MAKE PREDICTIONS
# ============================================

print("\n" + "="*70)
print("STEP 3: MAKING PREDICTIONS")
print("="*70)

print("\n📊 Predicting on training set...")
y_train_pred = lr_model.predict(X_train_tfidf, batch_size=5000)
y_train_proba = lr_model.predict_proba(X_train_tfidf, batch_size=5000)
gc.collect()

print("📊 Predicting on test set...")
y_test_pred = lr_model.predict(X_test_tfidf, batch_size=5000)
y_test_proba = lr_model.predict_proba(X_test_tfidf, batch_size=5000)
gc.collect()

print("✅ Predictions completed!")

# ============================================
# 5. EVALUATION METRICS
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
# 6. CONFUSION MATRIX
# ============================================

print("\n" + "="*70)
print("STEP 6: CONFUSION MATRIX")
print("="*70)

cm = confusion_matrix(y_test, y_test_pred)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Raw counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
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
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            cbar_kws={'label': 'Percentage'},
            ax=axes[1],
            annot_kws={'size': 14, 'weight': 'bold'})
axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold', pad=20)
axes[1].set_ylabel('True Label', fontsize=12)
axes[1].set_xlabel('Predicted Label', fontsize=12)

plt.tight_layout()
plt.savefig(SCRIPT_DIR / "lr_confusion_matrix.png", dpi=150, bbox_inches='tight')
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
# 7. ROC CURVE AND AUC
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
plt.plot(fpr, tpr, color='darkorange', lw=3,
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
plt.savefig(SCRIPT_DIR / "lr_roc_curve.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# 8. PRECISION-RECALL CURVE
# ============================================

print("\n" + "="*70)
print("STEP 8: PRECISION-RECALL CURVE")
print("="*70)

precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_test_proba)
average_precision = average_precision_score(y_test, y_test_proba)

print(f"\n🎯 Average Precision Score: {average_precision:.4f}")

plt.figure(figsize=(10, 8))
plt.plot(recall_curve, precision_curve, color='blue', lw=3,
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
plt.savefig(SCRIPT_DIR / "lr_pr_curve.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# 9. PERFORMANCE SUMMARY
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

# Visualize metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
train_scores = [train_accuracy, train_precision, train_recall, train_f1]
test_scores = [test_accuracy, test_precision, test_recall, test_f1]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, train_scores, width, label='Training',
               color='skyblue', edgecolor='black', linewidth=1.5)
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
# 10. SAVE DASHBOARD JSON
# ============================================

print("\n" + "="*70)
print("STEP 10: SAVING DASHBOARD JSON")
print("="*70)

def save_dashboard_json():
    report = classification_report(
        y_test, y_test_pred,
        target_names=['Negative', 'Positive'],
        output_dict=True,
        digits=4,
    )

    # ROC curve — sample 12 points
    roc_indices = np.linspace(0, len(fpr) - 1, 12, dtype=int)
    roc_points  = [[round(float(fpr[i]), 3), round(float(tpr[i]), 3)] for i in roc_indices]

    # PR curve — sample 12 points
    pr_indices = np.linspace(0, len(precision_curve) - 1, 12, dtype=int)
    pr_points  = [[round(float(recall_curve[i]), 3), round(float(precision_curve[i]), 3)] for i in pr_indices]

    # Training history (loss per epoch)
    training_history = {
        'train_loss': [round(float(l), 6) for l in lr_model.losses],
    }

    tn, fp, fn, tp = cm.ravel()

    dashboard = {
        'model'     : 'Logistic Regression',
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
            'train_samples': int(X_train_tfidf.shape[0]),
            'test_samples' : int(X_test_tfidf.shape[0]),
            'n_features'   : int(X_train_tfidf.shape[1]),
        },
    }

    import json
    with open(SCRIPT_DIR / "lr_sentiment140_dashboard.json", 'w') as f:
        json.dump(dashboard, f, indent=2)
    print('✅ Dashboard JSON saved → lr_sentiment140_dashboard.json')


save_dashboard_json()

print("\n" + "="*70)
print("✅ LOGISTIC REGRESSION COMPLETE!")
print("="*70)