# src/preprocess.py
import re


def clean_text(text: str) -> str:
    """
    Basic text normalization:
    - lowercase
    - keep only letters and digits
    - collapse multiple spaces
    """
    if text is None:
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    """Whitespace tokenization on cleaned text."""
    if text is None:
        return []
    return clean_text(text).split()
