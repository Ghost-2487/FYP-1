import re
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "data" / "reddit_depression_suicidewatch.csv")

df.head()
df.columns

print("Original size:", len(df))
print(df["label"].value_counts())

df_suicide = df[df["label"] == "SuicideWatch"]

print("SuicideWatch size:", len(df_suicide))

# ------------------------------
# 2. Text cleaning
# ------------------------------
clean_re = r"@\S+|https?:\S+|http?:\S|[^A-Za-z0-9 ]+"

def clean_text(t):
    t = str(t).lower()
    t = re.sub(clean_re, " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

df_suicide["clean_text"] = df_suicide["text"].apply(clean_text)

# Remove empty rows
df_suicide = df_suicide[df_suicide["clean_text"].str.strip() != ""]

# ------------------------------
# 3. Save clean suicide dataset
# ------------------------------
save_path = BASE_DIR / "data" / "suicide_only_clean.csv"
df_suicide.to_csv(save_path, index=False)

print("Saved →", save_path)
