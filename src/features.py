# src/features.py
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocess import clean_text, tokenize


def build_tfidf_vectorizer(text_series, max_features: int = 10000) -> TfidfVectorizer:
    """
    Fit a TF-IDF vectorizer on the combined question text.
    Limiting to 10k features keeps memory usage reasonable on a laptop.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.9,
    )
    vectorizer.fit(text_series)
    return vectorizer


def tfidf_transform_pairs(df, vectorizer: TfidfVectorizer):
    """Transform question1 and question2 into TF-IDF sparse matrices."""
    q1 = df["question1_clean"].fillna("").tolist()
    q2 = df["question2_clean"].fillna("").tolist()

    q1_tfidf = vectorizer.transform(q1)
    q2_tfidf = vectorizer.transform(q2)
    return q1_tfidf, q2_tfidf


def rowwise_cosine_similarity(q1_tfidf, q2_tfidf) -> np.ndarray:
    """
    Compute cosine similarity between q1[i] and q2[i] for each row.
    Uses sparse operations to avoid allocating an (N x N) matrix.
    """
    # dot product for each row -> (N, 1)
    numerator = q1_tfidf.multiply(q2_tfidf).sum(axis=1)

    # squared norms
    q1_sq = q1_tfidf.multiply(q1_tfidf).sum(axis=1)
    q2_sq = q2_tfidf.multiply(q2_tfidf).sum(axis=1)

    q1_norm = np.sqrt(np.asarray(q1_sq).ravel())
    q2_norm = np.sqrt(np.asarray(q2_sq).ravel())

    denom = q1_norm * q2_norm + 1e-9
    cos_sim = np.asarray(numerator).ravel() / denom
    return cos_sim.reshape(-1, 1)


def compute_similarity_features(q1_tfidf, q2_tfidf, df):
    """
    Build dense similarity features:
    - rowwise cosine similarity
    - absolute length difference between questions
    - common word ratio (intersection / union)
    """
    cos_sim = rowwise_cosine_similarity(q1_tfidf, q2_tfidf)

    q1_len = df["question1_clean"].str.len().values.reshape(-1, 1)
    q2_len = df["question2_clean"].str.len().values.reshape(-1, 1)
    len_diff = np.abs(q1_len - q2_len)

    common_ratio = []
    for q1, q2 in zip(df["question1_clean"], df["question2_clean"]):
        t1 = set(tokenize(q1))
        t2 = set(tokenize(q2))
        if not t1 and not t2:
            common_ratio.append(0.0)
            continue
        inter = len(t1 & t2)
        union = len(t1 | t2)
        common_ratio.append(inter / (union + 1e-9))

    common_ratio = np.array(common_ratio).reshape(-1, 1)

    dense_features = np.hstack([cos_sim, len_diff, common_ratio])
    return dense_features


def build_feature_matrix(df, vectorizer: TfidfVectorizer):
    """
    Combine sparse TF-IDF features and dense similarity features
    into a single sparse feature matrix suitable for XGBoost.
    """
    q1_tfidf, q2_tfidf = tfidf_transform_pairs(df, vectorizer)
    dense_features = compute_similarity_features(q1_tfidf, q2_tfidf, df)

    sparse_part = sp.hstack([q1_tfidf, q2_tfidf])
    dense_sparse = sp.csr_matrix(dense_features)
    X = sp.hstack([sparse_part, dense_sparse]).tocsr()
    return X
