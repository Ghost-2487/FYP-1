# ============================================
# 1. IMPORT LIBRARIES
# ============================================
import tensorflow as tf
import pandas as pd
import numpy as np
import re
import pickle
from scipy.sparse import save_npz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================
# 2. CONFIGURATION PARAMETERS
# ============================================
CONFIG = {
    'TRAIN_SIZE': 0.9,
    'MAX_NB_WORDS': 200000,
    'MAX_SEQUENCE_LENGTH': 50,
    'TFIDF_MAX_FEATURES': 100000,
    'TFIDF_NGRAM_RANGE': (1, 2),
    'RANDOM_STATE': 42,
    'DATASET_PATH': BASE_DIR / "data" / "training.1600000.processed.noemoticon.csv",
    'ENCODING': 'latin'
}

# ============================================
# 3. LOAD DATA
# ============================================
print("="*60)
print("LOADING DATASET")
print("="*60)

df = pd.read_csv(
    CONFIG['DATASET_PATH'],
    encoding=CONFIG['ENCODING'],
    header=None,
    names=['sentiment', 'id', 'date', 'query', 'user_id', 'text']
)

# Basic cleaning
df = df.drop(['id', 'date', 'query', 'user_id'], axis=1)
df = df.dropna(subset=['text', 'sentiment'])
df = df.drop_duplicates(subset=['text'], keep='first')

# Map labels
lab_to_sentiment = {0: "Negative", 4: "Positive"}
df['sentiment'] = df['sentiment'].apply(lambda x: lab_to_sentiment.get(x, "Unknown"))

# ============================================
# 4. TEXT PREPROCESSING
# ============================================
print("\n" + "="*60)
print("TEXT PREPROCESSING")
print("="*60)

# Minimalistic approach: Keep stopwords and punctuation context
def preprocess(text):
    if pd.isna(text): return ""

    # 1. Lowercase
    tweet = str(text).lower()

    # 2. Replace URLs, Usernames, and sequence patterns
    tweet = re.sub(r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)", ' URL ', tweet)
    tweet = re.sub(r'@[^\s]+', ' USER ', tweet)
    tweet = re.sub(r"(.)\1\1+", r"\1\1", tweet) # coool -> cool

    # 3. Handle special characters (keep ?, ! as they carry sentiment)
    tweet = re.sub(r"[^a-z?!' ]", " ", tweet)

    # 4. Add space around punctuation for tokenization
    tweet = re.sub(r'([?!])', r' \1 ', tweet)

    return " ".join(tweet.split())

print("Preprocessing text...")
df["text"] = df["text"].apply(preprocess)
df = df[df['text'].str.strip() != '']

# ============================================
# 5. TRAIN-TEST SPLIT
# ============================================
train_data, test_data = train_test_split(
    df,
    test_size=1-CONFIG['TRAIN_SIZE'],
    random_state=CONFIG['RANDOM_STATE'],
    stratify=df['sentiment']
)

# ============================================
# 6. TOKENIZATION
# ============================================
print("\n" + "="*60)
print("TOKENIZATION")
print("="*60)

tokenizer = Tokenizer(num_words=CONFIG['MAX_NB_WORDS'], filters='') # Keep punctuation if already spaced
tokenizer.fit_on_texts(train_data.text)

x_train = pad_sequences(
    tokenizer.texts_to_sequences(train_data.text),
    maxlen=CONFIG['MAX_SEQUENCE_LENGTH'],
    padding='post',
    truncating='post'
)

x_test = pad_sequences(
    tokenizer.texts_to_sequences(test_data.text),
    maxlen=CONFIG['MAX_SEQUENCE_LENGTH'],
    padding='post',
    truncating='post'
)

# ============================================
# 7. LABEL ENCODING
# ============================================
encoder = LabelEncoder()
y_train = encoder.fit_transform(train_data.sentiment).reshape(-1, 1)
y_test = encoder.transform(test_data.sentiment).reshape(-1, 1)

# ============================================
# 8. TF-IDF
# ============================================
vectoriser = TfidfVectorizer(
    ngram_range=CONFIG['TFIDF_NGRAM_RANGE'],
    max_features=CONFIG['TFIDF_MAX_FEATURES'],
    min_df=3
)
vectoriser.fit(train_data.text)
X_train_tfidf = vectoriser.transform(train_data.text)
X_test_tfidf = vectoriser.transform(test_data.text)

# ============================================
# 9. SAVING
# ============================================
print("\n" + "="*60)
print("SAVING PREPROCESSED DATA")
print("="*60)

with open(BASE_DIR / "data" / "tokenizer.pickle", 'wb') as h: pickle.dump(tokenizer, h)
with open(BASE_DIR / "data" / "label_encoder.pickle", 'wb') as h: pickle.dump(encoder, h)
with open(BASE_DIR / "data" / "tfidf_vectorizer.pickle", 'wb') as h: pickle.dump(vectoriser, h)

np.save(BASE_DIR / "data" / "x_train_sequences.npy", x_train)
np.save(BASE_DIR / "data" / "x_test_sequences.npy", x_test)
np.save(BASE_DIR / "data" / "y_train.npy", y_train)
np.save(BASE_DIR / "data" / "y_test.npy", y_test)

save_npz(BASE_DIR / "data" / "X_train_tfidf.npz", X_train_tfidf)
save_npz(BASE_DIR / "data" / "X_test_tfidf.npz", X_test_tfidf)

print("\nPREPROCESSING COMPLETE!")