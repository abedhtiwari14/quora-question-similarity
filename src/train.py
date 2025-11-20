# src/train.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier

from .preprocess import clean_text
from .features import build_tfidf_vectorizer, build_feature_matrix

DATA_PATH = "data/raw/train.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "quora_xgb_model.json")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[["question1", "question2", "is_duplicate"]].dropna()
    return df


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["question1_clean"] = df["question1"].apply(clean_text)
    df["question2_clean"] = df["question2"].apply(clean_text)
    return df


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading data...")
    df = load_data()
    df = preprocess_df(df)

    # Subsample for local training; still large enough for good performance
    df = df.sample(200_000, random_state=42)

    X_train_df, X_val_df, y_train, y_val = train_test_split(
        df,
        df["is_duplicate"].astype(int),
        test_size=0.2,
        random_state=42,
        stratify=df["is_duplicate"].astype(int),
    )

    print("Fitting TF-IDF vectorizer on training questions...")
    train_text = pd.concat(
        [X_train_df["question1_clean"], X_train_df["question2_clean"]],
        axis=0,
    )
    vectorizer = build_tfidf_vectorizer(train_text)

    print("Building feature matrices...")
    X_train = build_feature_matrix(X_train_df, vectorizer)
    X_val = build_feature_matrix(X_val_df, vectorizer)

    print("Training XGBoost classifier...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    print("Evaluating on validation set...")
    val_probs = model.predict_proba(X_val)[:, 1]
    val_pred = (val_probs >= 0.5).astype(int)

    acc = accuracy_score(y_val, val_pred)
    f1 = f1_score(y_val, val_pred)

    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation F1-score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, val_pred))

    print(f"Saving model to {MODEL_PATH}")
    model.save_model(MODEL_PATH)

    print(f"Saving TF-IDF vectorizer to {VECTORIZER_PATH}")
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("Training complete.")


if __name__ == "__main__":
    train()
