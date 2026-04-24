import re
import string
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)

_stop_words = set(stopwords.words("english"))
_stemmer = PorterStemmer()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    tokens = text.split()
    tokens = [_stemmer.stem(t) for t in tokens if t not in _stop_words and len(t) > 2]
    return " ".join(tokens)


def combine_features(row: pd.Series) -> str:
    fields = ["title", "company_profile", "description", "requirements", "benefits"]
    parts = []
    for field in fields:
        val = row.get(field, "")
        if pd.notna(val) and isinstance(val, str) and val.strip():
            parts.append(val)
    return " ".join(parts)


def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.fillna("")

    df["combined_text"] = df.apply(combine_features, axis=1)
    df["clean_text"] = df["combined_text"].apply(clean_text)

    # Drop rows with empty text
    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)

    if "fraudulent" in df.columns:
        df["label"] = df["fraudulent"].astype(int)

    return df


def preprocess_single(text: str) -> str:
    return clean_text(text)
