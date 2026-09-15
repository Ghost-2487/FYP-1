import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.metrics import (roc_curve, precision_recall_curve, precision_score,
                             average_precision_score, classification_report, auc,
                             accuracy_score, recall_score, f1_score, confusion_matrix)
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

# Load both 
try:
    df1 = pd.read_csv(BASE_DIR / "data" / "Suicide_Detection.csv")
    df2 = pd.read_csv(BASE_DIR / "data" / "suicide_only_clean.csv")

    print(f"✅ Data loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Error: {e}. Run extract_suicide_data.py first.")
    exit()

print("===== DATASET 1: Suicide_Detection.csv =====")
print("Shape:", df1.shape)
print("Columns:", df1.columns.tolist())
print(df1.head(), "\n\n")

print("===== DATASET 2: suicide_only_clean.csv =====")
print("Shape:", df2.shape)
print("Columns:", df2.columns.tolist())
print(df2.head())


nltk.download('vader_lexicon')

df1 = df1[['text', 'class']]
df1['label'] = df1['class'].map({'suicide': 1, 'non-suicide': 0})
df1.drop(columns=['class'], inplace=True)

df2 = df2[['clean_text', 'label']]
df2.rename(columns={'clean_text': 'text'}, inplace=True)
df2['label'] = 1    # Because SuicideWatch = suicide posts

df = pd.concat([df1, df2], ignore_index=True)

print("Final merged shape:", df.shape)
print(df['label'].value_counts())
df.head()

#feature extraction
sia = SentimentIntensityAnalyzer()

keywords = [
    "suicide","kill","die","end","worthless","hopeless","depressed",
    "no reason","kill myself","end it","pain","hurt","self harm"
]

def extract_features(text):
    text = str(text)

    char_count = len(text)
    words = text.split()
    word_count = len(words)
    avg_word_len = np.mean([len(w) for w in words]) if word_count else 0
    sentence_count = len(re.split(r'[.!?]+', text))
    punct_count = len(re.findall(r'[^\w\s]', text))
    digit_count = sum(c.isdigit() for c in text)
    capital_ratio = sum(c.isupper() for c in text) / (len(text) + 1)

    # TextBlob sentiment
    blob = TextBlob(text)
    polarity = blob.polarity
    subjectivity = blob.subjectivity

    # VADER sentiment
    vader = sia.polarity_scores(text)
    neg = vader['neg']
    pos = vader['pos']
    neu = vader['neu']

    # Suicide keywords
    key_count = sum(text.lower().count(k) for k in keywords)

    return pd.Series([
        char_count, word_count, avg_word_len, sentence_count,
        punct_count, digit_count, capital_ratio,
        polarity, subjectivity, neg, pos, neu, key_count
    ])

feature_names = [
    "char_count", "word_count", "avg_word_len", "sentence_count",
    "punct_count", "digit_count", "capital_ratio",
    "polarity", "subjectivity", "neg", "pos", "neu", "keyword_count"
]

df_features = df['text'].apply(extract_features)
df_features.columns = feature_names

# Final dataset for model
final_df = pd.concat([df_features, df['label']], axis=1)

print(final_df.head())
print("Final feature shape:", final_df.shape)


# Convert to numpy arrays
X = final_df.drop(columns=['label']).values
y = final_df['label'].values

# Train/test split manually
def train_test_split_custom(X, y, test_size=0.2):
    idx = np.arange(len(X))
    np.random.shuffle(idx)

    test_len = int(test_size * len(X))
    test_idx = idx[:test_len]
    train_idx = idx[test_len:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = train_test_split_custom(X, y)
print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

def gini_impurity(y):
    classes = np.unique(y)
    impurity = 1.0
    for c in classes:
        p = np.sum(y == c) / len(y)
        impurity -= p ** 2
    return impurity


def best_split(X, y, features):
    best_feature = None
    best_thresh = None
    best_gain = 0

    current_impurity = gini_impurity(y)

    for feature in features:
        values = np.unique(X[:, feature])
        for threshold in values:
            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            left_impurity = gini_impurity(y[left_mask])
            right_impurity = gini_impurity(y[right_mask])

            p_left = len(y[left_mask]) / len(y)
            p_right = 1 - p_left

            gain = current_impurity - (p_left * left_impurity + p_right * right_impurity)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_thresh = threshold

    return best_feature, best_thresh, best_gain

class DecisionTree:
    def __init__(self, max_depth=5, min_size=5, n_features=None):
        self.max_depth = max_depth
        self.min_size = min_size
        self.n_features = n_features
        self.tree = None

    def fit(self, X, y, depth=0):
        self.tree = self._build(X, y, depth)

    def _build(self, X, y, depth):
        # Stopping conditions
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) <= self.min_size:
            return np.bincount(y).argmax()

        # random feature subset
        n_features = X.shape[1]
        features = np.random.choice(n_features, self.n_features, replace=False)

        feature, threshold, gain = best_split(X, y, features)

        if gain == 0 or feature is None:
            return np.bincount(y).argmax()

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[right_mask], y[right_mask], depth + 1)

        return (feature, threshold, left, right)

    def predict_one(self, x, node=None):
        if node is None:
            node = self.tree

        if not isinstance(node, tuple):  # leaf
            return node

        feature, threshold, left, right = node

        if x[feature] <= threshold:
            return self.predict_one(x, left)
        else:
            return self.predict_one(x, right)

    def predict(self, X):
        return np.array([self.predict_one(sample) for sample in X])

class RandomForest:
    def __init__(self, n_trees=10, max_depth=5, min_size=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_size = min_size
        self.trees = []

    def bootstrap(self, X, y):
        n_samples = len(X)
        idx = np.random.choice(n_samples, n_samples, replace=True)
        return X[idx], y[idx]

    def fit(self, X, y):
        self.trees = []
        n_features = int(np.sqrt(X.shape[1]))  # random sqrt features

        for _ in range(self.n_trees):
            X_s, y_s = self.bootstrap(X, y)
            tree = DecisionTree(max_depth=self.max_depth, min_size=self.min_size, n_features=n_features)
            tree.fit(X_s, y_s)
            self.trees.append(tree)

    def predict(self, X):
        # Collect predictions from all trees
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Majority vote
        final_pred = []
        for i in range(X.shape[0]):
            votes = tree_preds[:, i]
            final_pred.append(np.bincount(votes).argmax())
        return np.array(final_pred)


#train model
forest = RandomForest(n_trees=10, max_depth=7, min_size=5)
forest.fit(X_train, y_train)

y_pred = forest.predict(X_test)

accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)


def gini_impurity(y):
    if len(y) == 0:
        return 0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)

def best_split(X, y, features):
    best_feature = None
    best_thresh = None
    best_gain = 0

    current_impurity = gini_impurity(y)
    n_samples = len(y)

    for feature in features:
        values = X[:, feature]

        thresholds = np.percentile(values, [10, 20, 30, 40, 50, 60, 70, 80, 90])

        for threshold in thresholds:
            left_mask = values <= threshold
            right_mask = ~left_mask

            n_left = np.sum(left_mask)
            n_right = np.sum(right_mask)

            if n_left < 2 or n_right < 2:
                continue

            # Calculate weighted Gini
            left_impurity = gini_impurity(y[left_mask])
            right_impurity = gini_impurity(y[right_mask])

            weighted_impurity = (n_left / n_samples) * left_impurity + (n_right / n_samples) * right_impurity
            gain = current_impurity - weighted_impurity

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_thresh = threshold

    return best_feature, best_thresh, best_gain

class DecisionTree:
    def __init__(self, max_depth=5, min_size=10, n_features=None):
        self.max_depth = max_depth
        self.min_size = min_size
        self.n_features = n_features
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build(X, y, depth=0)

    def _build(self, X, y, depth):
        n_samples, n_features = X.shape

        if (depth >= self.max_depth or
            len(np.unique(y)) == 1 or
            n_samples <= self.min_size):
            return np.bincount(y).argmax()

        n_feat_subset = self.n_features if self.n_features else int(np.sqrt(n_features))
        n_feat_subset = min(n_feat_subset, n_features)
        features = np.random.choice(n_features, n_feat_subset, replace=False)

        feature, threshold, gain = best_split(X, y, features)

        if gain == 0 or feature is None:
            return np.bincount(y).argmax()

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[right_mask], y[right_mask], depth + 1)

        return (feature, threshold, left, right)

    def predict_one(self, x, node=None):
        if node is None:
            node = self.tree

        if not isinstance(node, tuple):
            return node

        feature, threshold, left, right = node

        if x[feature] <= threshold:
            return self.predict_one(x, left)
        else:
            return self.predict_one(x, right)

    def predict(self, X):
        return np.array([self.predict_one(sample) for sample in X])


class RandomForest:
    def __init__(self, n_trees=10, max_depth=7, min_size=10):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_size = min_size
        self.trees = []

    def bootstrap(self, X, y):
        n_samples = len(y)
        idx = np.random.choice(n_samples, n_samples, replace=True)
        return X[idx], y[idx]

    def fit(self, X, y):
        print(f"\n{'='*60}")
        print(f"TRAINING RANDOM FOREST")
        print(f"{'='*60}")
        print(f"Trees: {self.n_trees} | Max depth: {self.max_depth} | Min size: {self.min_size}")
        print(f"Training samples: {X.shape[0]} | Features: {X.shape[1]}")
        print(f"{'='*60}\n")

        self.trees = []
        n_features = int(np.sqrt(X.shape[1]))

        for i in range(self.n_trees):
            print(f"Training tree {i+1}/{self.n_trees}...", end=" ")
            X_sample, y_sample = self.bootstrap(X, y)

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_size=self.min_size,
                n_features=n_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            print("✓")

        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"{'='*60}\n")

    def predict(self, X):
        print(f"Making predictions on {X.shape[0]} samples...")
        tree_preds = np.array([tree.predict(X) for tree in self.trees])

        # Majority vote (vectorized)
        final_pred = np.array([np.bincount(tree_preds[:, i]).argmax() for i in range(X.shape[0])])
        print("Predictions complete!\n")
        return final_pred
    
    def predict_proba(self, X):
        """Returns probability of positive class (fraction of trees voting 1)."""
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Fraction of trees that voted positive class
        return tree_preds.mean(axis=0)

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

forest = RandomForest(n_trees=15, max_depth=6, min_size=20)
forest.fit(X_train, y_train)

y_pred = forest.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("="*60)
print("MODEL PERFORMANCE")
print("="*60)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print("="*60)

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
plt.savefig(SCRIPT_DIR / 'rf_suicide_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"  True Negatives:  {tn:,}")
print(f"  False Positives: {fp:,}")
print(f"  False Negatives: {fn:,}")
print(f"  True Positives:  {tp:,}")

# ============================================
# CLASS DISTRIBUTION BAR CHART
# ============================================

pred_counts = pd.Series(y_pred).value_counts().sort_index()
true_counts = pd.Series(y_test).value_counts().sort_index()

x = np.arange(2)
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, true_counts.values, width, label='Actual', alpha=0.8)
plt.bar(x + width/2, pred_counts.values, width, label='Predicted', alpha=0.8)
plt.xlabel('Class', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Class Distribution — Actual vs Predicted', fontsize=14, fontweight='bold', pad=20)
plt.xticks(x, target_names)
plt.legend(fontsize=11)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'rf_suicide_class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# ROC CURVE
# ============================================

print("\n" + "="*60)
print("ROC CURVE & AUC SCORE")
print("="*60)

y_proba = forest.predict_proba(X_test)

fpr, tpr, _ = roc_curve(y_test, y_proba)
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
plt.title('ROC Curve — Random Forest (Suicide Dataset)',
          fontsize=14, fontweight='bold', pad=20)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'rf_suicide_roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# PRECISION-RECALL CURVE
# ============================================

print("\n" + "="*60)
print("PRECISION-RECALL CURVE")
print("="*60)

precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
average_precision = average_precision_score(y_test, y_proba)

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
plt.title('Precision-Recall Curve — Random Forest (Suicide Dataset)',
          fontsize=14, fontweight='bold', pad=20)
plt.legend(loc='lower left', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'rf_suicide_pr_curve.png', dpi=150, bbox_inches='tight')
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
        'model'     : 'Random Forest',
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
        'training_history': None,   # Custom RF has no epoch-level loss tracking
        'model_stats': {
            'n_trees'       : forest.n_trees,
            'max_depth'     : forest.max_depth,
            'min_size'      : forest.min_size,
            'train_samples' : int(X_train.shape[0]),
            'test_samples'  : int(X_test.shape[0]),
            'n_features'    : int(X_train.shape[1]),
        },
    }

    with open(SCRIPT_DIR / 'rf_suicide_dashboard.json', 'w') as f:
        json.dump(dashboard, f, indent=2)
    print('✅ Dashboard JSON saved → rf_suicide_dashboard.json')

save_dashboard_json()