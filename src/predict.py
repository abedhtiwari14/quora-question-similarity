# src/predict.py
import os
import sys
import joblib
import scipy.sparse as sp
import pandas as pd
from xgboost import XGBClassifier

from .preprocess import clean_text
from .features import compute_similarity_features


MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "quora_xgb_model.json")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")


def load_model_and_vectorizer():
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model, vectorizer


def build_features_for_pair(q1_raw: str, q2_raw: str, vectorizer):
    q1_clean = clean_text(q1_raw)
    q2_clean = clean_text(q2_raw)

    q1_tfidf = vectorizer.transform([q1_clean])
    q2_tfidf = vectorizer.transform([q2_clean])

    df_pair = pd.DataFrame(
        {
            "question1_clean": [q1_clean],
            "question2_clean": [q2_clean],
        }
    )
    dense_feats = compute_similarity_features(q1_tfidf, q2_tfidf, df_pair)
    dense_sparse = sp.csr_matrix(dense_feats)

    sparse_part = sp.hstack([q1_tfidf, q2_tfidf])
    X = sp.hstack([sparse_part, dense_sparse]).tocsr()
    return X


def main():
    if len(sys.argv) < 3:
        print('Usage: python -m src.predict "question 1" "question 2"')
        sys.exit(1)

    q1 = sys.argv[1]
    q2 = sys.argv[2]

    model, vectorizer = load_model_and_vectorizer()
    X = build_features_for_pair(q1, q2, vectorizer)

    prob = model.predict_proba(X)[0, 1]
    pred = int(prob >= 0.5)

    print(f"Question 1: {q1}")
    print(f"Question 2: {q2}")
    print(f"Duplicate probability: {prob:.4f}")
    print(f"Predicted label (is_duplicate): {pred}")


if __name__ == "__main__":
    main()
