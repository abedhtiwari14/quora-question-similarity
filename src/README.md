# 🔍 Quora Question Pair Similarity (Duplicate Question Detection)

This project builds a machine learning model to detect whether two Quora questions express the same meaning.  
Using a hybrid NLP approach — **TF-IDF**, **semantic similarity features**, and an **XGBoost classifier** — the model achieves strong performance on the official Quora Question Pairs benchmark dataset.

---

## 📌 **Project Overview**

Question duplication is a major problem on forums like Quora, StackOverflow, Reddit, etc.  
This project predicts whether:

question1 ≈ question2 → is_duplicate = 1


The solution uses:
- Text preprocessing  
- TF-IDF vectorization  
- Row-wise cosine similarity  
- Length-based & lexical overlap features  
- XGBoost classifier  

All implemented from scratch with efficient sparse matrix handling to train cleanly on a MacBook.

---

## 📂 **Dataset**

This project uses the **official Kaggle Quora Question Pairs** dataset:

- 404,290 labeled question pairs  
- Columns:
  - `question1`, `question2`
  - `is_duplicate` (target)
  - `id`, `qid1`, `qid2`

Only `train.csv` is used for model development, split into:

- **80% training**
- **20% validation**

> Kaggle’s `test.csv` contains *no labels*, so it is not used for evaluation.

---

## 🧹 **Text Preprocessing**

Each question pair is cleaned by:
- Lowercasing
- Removing non-alphanumeric characters
- Collapsing multiple spaces
- Simple whitespace tokenization

This ensures clean, uniform text for feature generation.

---

## 🧠 **Feature Engineering**

The model uses a **hybrid feature set**:

### 1️⃣ TF-IDF Features  
- 10,000 most informative unigram + bigram features  
- Applied separately to `question1` and `question2`  
- Efficient sparse matrices

### 2️⃣ Semantic Similarity Features  
Computed row-by-row:

| Feature | Description |
|--------|-------------|
| **Cosine similarity** | How similar Q1 & Q2 vectors are |
| **Length difference** | `|len(q1) − len(q2)|` |
| **Common word ratio** | Intersection / Union of word sets |

These provide interpretable signals beyond TF-IDF.

---

## ⚙️ **Model**

The final classifier is:

**XGBoost (Gradient Boosted Decision Trees)**  
- Handles sparse high-dimensional text well  
- Robust and fast  
- Strong real-world performance on tabular+text hybrids

Training subset: **200,000 pairs** (for efficient local training)  
Vectorizer + model are saved in the `models/` directory.

---

## 📊 **Results**

Achieved on the validation split:

- **Validation Accuracy:** **0.7737**
- **Validation F1-score:** **0.6871**

**Classification Report:**

| Metric | Value |
|--------|-------|
| Precision | 0.77 |
| Recall    | 0.77 |
| F1-score  | 0.77 |
| Support   | 40,000 samples |

These results are in line with standard TF-IDF + XGBoost baselines on the Quora dataset.

---

## 🚀 **Usage**

### ▶️ **Train the model**

```bash
python -m src.train
