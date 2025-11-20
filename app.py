# app.py
import streamlit as st

from src.predict import load_model_and_vectorizer, build_features_for_pair


@st.cache_resource
def get_model_and_vectorizer():
    return load_model_and_vectorizer()


def main():
    st.set_page_config(
        page_title="Quora Question Pair Similarity",
        page_icon="🔍",
        layout="centered",
    )

    st.title("🔍 Quora Question Pair Similarity")
    st.write(
        "Enter two questions and the model will predict whether they are "
        "**duplicates** (i.e., asking the same thing) using TF-IDF features, "
        "custom similarity features, and an XGBoost classifier."
    )
    st.markdown("---")

    q1 = st.text_area(
        "✏️ Question 1",
        height=100,
        placeholder="e.g. How can I learn data science?",
    )
    q2 = st.text_area(
        "✏️ Question 2",
        height=100,
        placeholder="e.g. What is the best way to get started with data science?",
    )

    if st.button("Predict Similarity"):
        if not q1.strip() or not q2.strip():
            st.warning("Please enter both questions.")
            return

        with st.spinner("Analyzing..."):
            model, vectorizer = get_model_and_vectorizer()
            X = build_features_for_pair(q1, q2, vectorizer)
            prob = model.predict_proba(X)[0, 1]
            pred = int(prob >= 0.5)

        st.markdown("---")
        st.subheader("🔎 Result")

        if pred == 1:
            st.success(f"Prediction: ✅ Duplicate (Similar)")
        else:
            st.info(f"Prediction: ❌ Not Duplicate (Different)")

        st.write(f"Duplicate probability: `{prob:.4f}`")
        st.progress(float(prob))

        st.caption(
            "Model: TF-IDF (10k features) + custom similarity features + "
            "XGBoost trained on the Quora Question Pairs dataset."
        )


if __name__ == "__main__":
    main()
