<div align="center">

# Mental Health Detection from Social Media Text

**Final Year Research Project — ML & DL Comparative Study | FAST-NUCES Karachi · Spring 2026**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)](https://keras.io/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=flat)](https://huggingface.co/)
[![Pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-11557C?style=flat)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.x-4C72B0?style=flat)](https://seaborn.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-F5A623?style=flat&logoColor=white)](https://opensource.org/licenses/MIT)

</div>

---

This repository contains the complete implementation of our Final Year Project — a comparative study of Machine Learning and Deep Learning approaches for mental health detection from social media text. It includes all model scripts, preprocessing pipelines, result artifacts, documentation, and an interactive React dashboard for visualizing and comparing model performance.

This project explores two critical domains of digital mental health surveillance: **Depression Sentiment Analysis** and **Suicide Ideation Detection**. Each pipeline uses techniques tailored to its dataset's scale and characteristics. To ensure fairness and reproducibility, all models within each pipeline are evaluated using standardized preprocessing pipelines and identical train/test splits.

---

## 👥 Authors & Affiliation
* **Developers:**
  * Muhammad Tahir Haider
  * Muhammad Huzaifa Saleem 
  * Syed Muhammad Sufyan
* **Project Supervisor:** Ms. Fizza Aqeel
* **Institution:** FAST School of Computing, National University of Computer and Emerging Sciences (FAST-NUCES), Karachi Campus.
* **Timeline:** Fall 2025 – Spring 2026

---

## 📌 Executive Summary
Mental health disorders represent a growing global crisis. Traditional clinical diagnostics often suffer from social stigma and delayed reporting. Social media platforms have emerged as a rich source of unfiltered, real-time expression — offering a non-invasive opportunity to detect early signs of distress through natural language processing. Leveraging user-generated text from platforms like Twitter and Reddit, this project systematically evaluates classical ML, deep learning, and transformer-based approaches to determine which paradigms are most effective for mental health detection at scale.

This project evaluates **16 model implementations** (8 models trained independently for Depression detection and 8 for Suicide Ideation detection) across three major paradigms:
1. **Classical Machine Learning:** Logistic Regression, Support Vector Machine (SVM), Naive Bayes, and Random Forest.
2. **Deep Learning:** 1D Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM), and Bidirectional LSTM (BiLSTM).
3. **Transformers:** BERT (`bert-base-uncased`).

---

## 🛠️ Architecture & Feature Engineering

```
             ┌─────────────────────────────────────────────┐
             │               Data Ingestion                │
             └──────────────────────┬──────────────────────┘
                                    ▼
            ┌───────────────────────────────────────────────┐
            │             Preprocessing Pipeline            │
            │       Lowercasing · URL/Mention Removal       │
            │        Noise Reduction · Tokenization         │
            └───────────────────────┬───────────────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           ▼                        ▼                        ▼
   ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
   │  TF-IDF Vectors  │     │  Word Embeddings │     │  BERT Tokenizer  │
   └────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
            ▼                        ▼                        ▼
   ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
   │   Classical ML   │     │  Deep Learning   │     │   Transformer    │
   │                  │     │                  │     │                  │
   │ Logistic Reg.    │     │ CNN              │     │ BERT             │
   │ Naive Bayes      │     │ LSTM             │     │ bert-base-       │
   │ SVM              │     │ BiLSTM           │     │ uncased          │
   │ Random Forest    │     │                  │     │                  │
   └────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
            └────────────────────────┼────────────────────────┘
                                     ▼
             ┌────────────────────────────────────────────────┐
             │             Evaluation Framework               │
             │  Accuracy · Precision · Recall · F1 · AUC      │
             │  Confusion Matrix · ROC Curve · PR Curve       │
             └───────────────────────┬────────────────────────┘
                                     ▼
              ┌────────────────────────────────────────────────┐
              │             Comparative Analysis               │
              └────────────────────────────────────────────────┘
```

### 1. Feature Representation Pathways
* **Classical ML Path:** Utilizes TF-IDF vectorization generating high-dimensional sparse matrices. The Depression pipeline uses unigrams + bigrams constrained to the top 100,000 features (`min_df=3`). The Suicide pipeline uses unigrams only (`max_features=20,000`), with the exception of Random Forest which bypasses TF-IDF entirely in favour of hand-crafted features.
* **Deep Learning Path:** Operates on integer sequences generated by Keras tokenizers.
  * *Depression Domain:* Employs a **hybrid embedding strategy** combining frozen 100-dimensional pre-trained **GloVe Twitter 27B** embeddings with trainable 200-dimensional domain-specific embeddings, creating a rich 300-dimensional representation.
  * *Suicide Domain:* Employs a single-branch 128-dimensional trainable embedding optimized with CuDNN kernels.
* **Transformer Path:** Utilizes Hugging Face's `BertTokenizerFast` / `AutoTokenizer` with `bert-base-uncased` to generate 768-dimensional bidirectional contextual representations. The Depression BERT uses a dedicated preprocessing path (`clean_text_for_bert()`) with contraction expansion and `[URL]`/`[USER]` special tokens, preserving the natural language structure required by BERT's tokenizer. The Suicide BERT uses a lighter cleaning path (URL/mention removal, lowercasing) without special token injection.

### 2. Custom Implementations (From Scratch)
To guarantee memory scalability on massive social media datasets (up to 1.6M entries), we developed **custom, memory-optimized Python implementations** from the ground up for select classifiers, bypassing standard library memory limits via sparse operations:

* **Logistic Regression (both pipelines):** Implemented using mini-batch gradient descent operating directly on sparse `CSR` matrices. The Depression variant uses 5,000-sample batches; the Suicide variant adds explicit L2 regularization (`lambda=0.01`).
* **Naive Bayes (both pipelines):** Implemented as a Multinomial Naive Bayes classifier utilizing sparse class-conditional feature accumulation and the log-sum-exp trick for numerical stability.
* **SVM (Suicide pipeline):** Implemented from scratch as a custom `LinearSVM` class using mini-batch subgradient descent with hinge loss and upsampling for class balance.
* **Random Forest (Suicide pipeline):** Implemented from scratch with custom decision tree nodes using Gini impurity, operating directly on the 13 hand-crafted features.

The remaining classical models use optimized sklearn implementations:
* **SVM (Depression pipeline):** sklearn's `LinearSVC` with Platt scaling probability calibration (`CalibratedClassifierCV`).
* **Random Forest (Depression pipeline):** sklearn's `RandomForestClassifier` with warm-start incremental tree growing and Out-of-Bag (OOB) error monitoring for early stopping.

---

## 📊 Datasets & Corpus Profiles

### 1. Depression Sentiment Analysis Dataset
* **Source Corpus:** [Sentiment140 (Kaggle)](https://www.kaggle.com/datasets/kazanova/sentiment140)
* **Dataset Volume:** 1,600,000 entries (balanced binary sentiment markers)
* **Experimental Split:** 90% Training / 10% Test (**Stratified**)
* **Linguistic Profile:** Short-form microtext averaging 64 characters per tweet, characterized by informal language, social media slang, and frequent contractions.
* **Core Data Pipeline (`preprocess_sentiment_data.py`):** Ingests raw text and applies lowercasing, character deduplication (e.g., *coool* → *cool*), URL/mention replacement (` URL `, ` USER `), and punctuation padding. Outputs clean text arrays and label configurations to `Depression/data/`.

### 2. Suicide Ideation Dataset
* **Source Corpus:** Combined from two sources:
  1. [Reddit Dataset r/depression and r/SuicideWatch](https://www.kaggle.com/datasets/xavrig/reddit-dataset-rdepression-and-rsuicidewatch) — Suicide watch threads isolated.
  2. [Suicide Watch Dataset (Kaggle)](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)
* **Dataset Volume:** 242,066 entries (126,029 Suicidal vs. 116,037 Non-Suicidal)
* **Experimental Split:** 80% Training / 20% Test (**Stratified**)
* **Linguistic Profile:** Long-form prose averaging 132 words per post, with more structured language and contextual narratives than the Depression dataset.
* **Core Data Pipeline (`extract_suicide_data.py`):** Loads and merges both source CSVs, strips URLs and mentions, normalizes whitespace, and drops empty records, outputting a unified clean CSV to `Suicide/data/`.

---

## 🎯 Design Rationale: Why Different Techniques Across Pipelines?

A natural question arises: if the techniques differ between pipelines, is this a valid comparative study?

The answer is yes — and the differences are deliberate, not arbitrary. The core comparison in this study is **within each pipeline** (ML vs. DL vs. Transformers on the same dataset), not across pipelines. Each pipeline is an independent benchmark evaluated on its own dataset under consistent, standardized conditions.

The cross-pipeline differences exist because the two datasets have fundamentally different characteristics that make a one-size-fits-all approach inappropriate:

- **Dataset scale:** The Depression dataset contains 1.6M tweets, making memory efficiency a hard constraint. This necessitates custom from-scratch implementations where needed, TF-IDF constrained to 100K features, and GloVe Twitter embeddings pre-trained on social media text. The Suicide dataset (~242K entries) operates at a more manageable scale, allowing standard sklearn implementations and learned embeddings trained from scratch.
- **Text characteristics:** Depression tweets are short (avg. 64 characters), informal, and rich in slang and contractions — making GloVe Twitter 27B the ideal embedding choice. Suicide posts are longer Reddit text (avg. 132 words) with more structured language, where a trainable Keras embedding learned directly from the corpus is sufficient.
- **Class balance handling:** The Suicide SVM uses upsampling to address mild class imbalance; the Depression SVM uses sklearn's `class_weight='balanced'` via `LinearSVC`. Both are valid approaches for their respective scales.

In short, the design choices within each pipeline are driven by the dataset's scale, text style, and computational constraints — not by an attempt to give either pipeline an unfair advantage. The study's comparative validity is preserved because each pipeline is internally consistent across all 8 of its models.

---

## 📈 Experimental Results

Below are the experimental results across both pipelines:

### Table 1: Depression Dataset (Sentiment140) Performance Summary
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 81.36% | 81.37% | 81.36% | 81.36% | 89.27% | 89.44% |
| **Naive Bayes** | 80.25% | 80.25% | 80.25% | 80.25% | 88.14% | 88.19% |
| **SVM** | 82.32% | 82.33% | 82.32% | 82.32% | 90.29% | 90.46% |
| **Random Forest** | 77.52% | 77.56% | 77.52% | 77.51% | 85.77% | 85.73% |
| **CNN** | 83.05% | 83.05% | 83.05% | 83.05% | 91.07% | 91.30% |
| **LSTM** | 83.50% | 83.50% | 83.50% | 83.50% | 91.50% | 91.67% |
| **BiLSTM** | 83.53% | 83.53% | 83.53% | 83.53% | 91.53% | 91.69% |
| **BERT** (Fine-tuned) | **87.65%** | **87.65%** | **87.65%** | **87.65%** | **94.87%** | **94.94%** |

### Table 2: Suicide Dataset Performance Summary
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 86.93% | 89.38% | 85.01% | 87.14% | 94.05% | 94.38% |
| **Naive Bayes** | 91.16% | 91.53% | 91.16% | 91.12% | 97.72% | 97.95% |
| **SVM** | 85.28% | 85.30% | 85.28% | 85.27% | 92.63% | 91.39% |
| **Random Forest** | 81.96% | 82.03% | 81.96% | 81.97% | 87.60% | 86.20% |
| **CNN** | 94.59% | 94.60% | 94.59% | 94.59% | 98.72% | 98.80% |
| **LSTM** | 94.24% | 94.25% | 94.24% | 94.24% | 98.43% | 98.47% |
| **BiLSTM** | 93.97% | 93.98% | 93.97% | 93.97% | 98.36% | 98.36% |
| **BERT** (Fine-tuned) | **97.81%** | **97.81%** | **97.81%** | **97.81%** | **99.64%** | **99.66%** |

---

## 🔑 Key Research Insights & Trade-offs

1. **The Performance-Efficiency Frontier:**
   Fine-tuned BERT achieves outstanding results, yielding **87.65%** and **97.81%** accuracies. However, it demands immense computational overhead, requiring **167–210 minutes** on a GPU. In contrast, classical models like Naive Bayes train in **1 to 2 minutes**, maintaining robust performance (80.25% on Depression, 91.16% on Suicide), offering a highly efficient baseline.
2. **Context vs. Local Sequential Patterns:**
   For Suicide detection, deep learning models achieve high classification accuracies (up to 94.59% for CNN). The fact that CNN slightly outperforms LSTM models suggests that local sequential patterns captured by convolutional filters are highly informative for identifying crisis signatures in long-form Reddit posts, without requiring the full sequential context that LSTMs model.
3. **Interpretability vs. Black-box Representations:**
   Classical ML models (LR, NB, SVM, Random Forest) provide transparent feature associations, showing exactly which words or features contribute to a classification decision. Deep learning and transformer architectures sacrifice clinical interpretability to achieve higher performance.

---

## 📁 Repository Structure
```
📂 FYP/
├── 📂 Dashboard/                         # React Dashboard for Results Visualization
│   ├── 📂 public/                        # Public assets
│   ├── 📂 src/                           # React codebase
│   │   ├── 📂 components/                # UI Components (Overview, Comparison, etc.)
│   │   └── 📂 data/                      # Extracted JSON metrics & dataset metadata
│   ├── 📄 package.json                   # Node.js dependencies
│   ├── 📄 package-lock.json
│   └── 📄 README.md                      # Dashboard-specific instructions
├── 📂 Depression/                        # Depression Sentiment Pipeline
│   ├── 📂 data/                          # Raw datasets, preprocessed .npy/.npz/.pkl files & GloVe embeddings (gitignored)
│   ├── 📂 ML Models/                     # Classical ML models
│   │   ├── 📂 Logistic Regression/
│   │   ├── 📂 Naive Bayes/
│   │   ├── 📂 Random Forest/
│   │   └── 📂 SVM/
│   ├── 📂 DL Models/                     # Deep Learning models
│   │   ├── 📂 BiLSTM/
│   │   ├── 📂 CNN/
│   │   └── 📂 LSTM/
│   ├── 📂 Transformer/                   # Transformer model
│   │   └── 📂 BERT/
│   └── 📄 preprocess_sentiment_data.py   # Cleaning, Tokenization, Label Encoding & TF-IDF vectorization
├── 📂 Suicide/                           # Suicide Ideation Pipeline
│   ├── 📂 data/                          # Raw datasets & preprocessed .npz/.pkl files (gitignored)
│   ├── 📂 ML Models/                     # Classical ML models
│   │   ├── 📂 Logistic Regression/
│   │   ├── 📂 Naive Bayes/
│   │   ├── 📂 Random Forest/
│   │   └── 📂 SVM/
│   ├── 📂 DL Models/                     # Deep Learning models
│   │   ├── 📂 BiLSTM/
│   │   ├── 📂 CNN/
│   │   └── 📂 LSTM/
│   ├── 📂 Transformer/                   # Transformer model
│   │   └── 📂 BERT/
│   └── 📄 extract_suicide_data.py        # Extracts & cleans suicide entries from Reddit dataset
├── 📂 Documentation/                     # Project Reports & Documentation
│   ├── 📄 Proposal.pdf
│   ├── 📄 Literature Review.pdf
│   ├── 📄 Progress Report.pdf
│   └── 📄 Final Report.pdf
├── 📄 .gitignore                         
├── 📄 LICENSE                            # MIT License
└── 📄 README.md                          # Project overview and setup guide
```

---

## 🚀 Getting Started
 
This codebase can be executed either in the cloud using **Google Colab** (Highly Recommended) or via a **Local Machine**. 

> [!NOTE]
> You can run these scripts locally on Windows, but training Deep Learning and BERT architectures is computationally intensive and requires a **CUDA-supported NVIDIA GPU**. Modern `tensorflow` has no native GPU support on Windows, and its required GPU package—`tensorflow[and-cuda]`—only functions in Linux environments. In order to utilize hardware acceleration, you would need to set up **WSL2 (Windows Subsystem for Linux)**. If run on native Windows, the environment will automatically fall back to CPU execution, and you will not be able to utilize your GPU hardware.

---

### Cloud Execution via Google Colab
 
#### Step 1: Clone & Prepare Your Data Folders Locally
Before uploading to the cloud, clone the repository to your computer and download the required datasets into their designated folders:

1. **Depression Pipeline Data:**
   * Download [Sentiment140 (Kaggle)](https://www.kaggle.com/datasets/kazanova/sentiment140) and place the raw CSV inside `Depression/data/`.
   * Download the [GloVe Twitter 27B Embeddings](https://nlp.stanford.edu/data/glove.twitter.27B.zip), unzip the archive, and place the extracted text files directly into `Depression/data/`.
2. **Suicide Pipeline Data:**
   * Download the [Reddit Dataset](https://www.kaggle.com/datasets/xavrig/reddit-dataset-rdepression-and-rsuicidewatch) and the [Suicide Watch Dataset](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch).
   * Place both raw source CSVs inside `Suicide/data/`.
 
#### Step 2: Upload Workspace to Google Drive
Upload your entire pre-configured `FYP/` root folder directory straight onto your **Google Drive**.
 
#### Step 3: Initialize Google Colab Workspace
Open a new notebook inside Google Colab. If you plan to train Deep Learning or BERT models, change your runtime type to **GPU runtime** (`Runtime > Change runtime type` and select an available GPU tier). Otherwise, a standard CPU runtime is sufficient for traditional Machine Learning workflows. Execute the following commands in an empty cell to mount your storage and enter the project environment:
```python
from google.colab import drive
drive.mount('/content/drive')

# Step into your uploaded repository workspace
%cd /content/drive/MyDrive/FYP
```

#### Step 4: Run Pipeline Core Preprocessing
Before executing individual classifiers, you must run the core data processing scripts to extract, clean, and generate the localized binary arrays and sparse matrices (`.npy`/`.npz` files) required by the models:

* **For the Depression Domain:**
  Run the preprocessing script to clean the Sentiment140 data and generate the TF-IDF matrices (`.npz`), integer sequences (`.npy`), and pickle artifacts (`.pkl`) in `Depression/data/`, consumed by all downstream Depression model scripts:
  ```python
  %run Depression/preprocess_sentiment_data.py
  ```
* **For the Suicide Domain:**
  Run the extraction script to generate `suicide_only_clean.csv` in `Suicide/data/`:
  ```python
  %run Suicide/extract_suicide_data.py
  ```

#### Step 5: Run Any Model Script
You are now completely set up! You can invoke any classical, deep learning, or transformer architecture script directly using Colab's `%run` magic command. For example, to train and evaluate the Logistic Regression for depression:
```python
%cd Depression
%run "ML Models/Logistic Regression/lr_depression.py"
```

---

### Launching the Interactive Results Dashboard Locally
 
The dashboard is fully set up with pre-populated evaluation results generated by the model training scripts — no model training required to view it. Simply clone the repo and launch:
 
```bash
# Navigate to Dashboard directory
cd Dashboard
 
# Install dependencies (first time only)
npm install
 
# Start development server
npm start
```
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.
 
> [!NOTE]
> Make sure [Node.js](https://nodejs.org) is installed before running the above commands.

---

## ⚠️ Ethical Disclaimer
This project is an academic research study exploring computational approaches to mental health detection from public social media data.
* It is **not** a clinical diagnostic tool and has not been clinically validated.
* The system is not designed to replace professional medical advice, diagnosis, or intervention.
* If you or someone you know is experiencing distress or thoughts of self-harm, please contact national mental health helplines or support communities.

---

<div align="center">

Developed as part of the Final Year Project — Spring 2026
<br/>
FAST School of Computing, NUCES Karachi

</div>
