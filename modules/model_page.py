# ============================================
# model_page.py
# ============================================

import streamlit as st
import pandas as pd
import os
from modules import train_sequential
from modules.train_apriori import train_apriori
from modules.train_fp_growth import train_fp_growth
from modules.train_sequential import train_sequential
from modules.train_lstm_sentiment import train_lstm_sentiment


def model_page():
    st.title("🧠 Model Training Page")

    # ------------------------------------------------------------
    # Step 1: Ensure Data Availability
    # ------------------------------------------------------------
    if "uploaded_df" not in st.session_state:
        st.warning("⚠️ Please upload and preprocess data first on the Dataset page.")
        st.stop()

    df = None
    if "processed_tabular_df" in st.session_state:
        df = st.session_state["processed_tabular_df"]
    elif "processed_text_df" in st.session_state:
        df = st.session_state["processed_text_df"]
    elif "uploaded_df" in st.session_state:
        df = st.session_state["uploaded_df"]
    else:
        st.warning("⚠️ Please upload and preprocess data first on the Dataset page.")
        st.stop()
    st.write("**Detected Data Type:**", st.session_state.get("data_type", "Unknown"))

    algo = st.selectbox(
        "🔍 Choose an Algorithm:",
        [
            "Apriori (Association Rule Mining)",
            "FP-Growth (Association Rule Mining)",
            "Sequential Pattern Matching",
            "LSTM (Sentiment Analysis)"
        ],
    )

    st.markdown("---")

    # ------------------------------------------------------------
    # Apriori Algorithm
    # ------------------------------------------------------------
    if algo == "Apriori (Association Rule Mining)":
        min_support = st.slider("Minimum Support:", 0.01, 0.5, 0.05)
        min_confidence = st.slider("Minimum Confidence:", 0.1, 1.0, 0.4)

        if st.button("🚀 Train Apriori Model"):
            with st.spinner("Training Apriori model..."):
                rules, path = train_apriori(df, min_support, min_confidence)
                st.session_state["trained_model"] = rules
                st.session_state["model_type"] = "Apriori"
                st.success(f"✅ Apriori model trained successfully! Saved at {path}")
                st.dataframe(rules.head())

    # ------------------------------------------------------------
    # FP-Growth Algorithm
    # ------------------------------------------------------------
    elif algo == "FP-Growth (Association Rule Mining)":
        min_support = st.slider("Minimum Support:", 0.01, 0.5, 0.05)
        min_confidence = st.slider("Minimum Confidence:", 0.1, 1.0, 0.4)

        if st.button("🚀 Train FP-Growth Model"):
            with st.spinner("Training FP-Growth model..."):
                rules, path = train_fp_growth(df, min_support, min_confidence)
                st.session_state["trained_model"] = rules
                st.session_state["model_type"] = "FP-Growth"
                st.success(f"✅ FP-Growth model trained successfully! Saved at {path}")
                st.dataframe(rules.head())

    # ------------------------------------------------------------
    # Sequential Pattern Matching
    # ------------------------------------------------------------
    elif algo == "Sequential Pattern Matching":
        if "uploaded_df" not in st.session_state:
            st.warning("⚠️ Please upload and preprocess data first on the Dataset page.")
            return

        df = st.session_state["uploaded_df"]
        session_col = st.selectbox("Select Session/User Column:", df.columns)
        time_col = st.selectbox("Select Timestamp Column:", df.columns)
        item_col = st.selectbox("Select Item Column:", df.columns)
        order = st.number_input("Order (e.g. 1 for next-item):", min_value=1, max_value=5, value=1)
        min_support = st.number_input("Minimum Support:", min_value=1, max_value=10, value=1)

        if st.button("🚀 Train Sequential Pattern Model"):
            with st.spinner("Training Sequential Pattern model..."):
                model, model_path, json_path = train_sequential(df, session_col, time_col, item_col, order, min_support)
                st.session_state["trained_model"] = model
                st.session_state["model_type"] = "Sequential Pattern Matching"
                st.session_state["json_path"] = json_path

            st.success(f"✅ Model trained and saved successfully!")
            st.write(f"📦 Pickle: `{model_path}`")
            st.write(f"📄 JSON: `{json_path}`")
            st.json(dict(list(model.items())[:5]))

    # ------------------------------------------------------------
    # LSTM (Sentiment Analysis)
    # ------------------------------------------------------------
    elif algo == "LSTM (Sentiment Analysis)":
        if "processed_text_df" not in st.session_state:
            st.warning("⚠️ Please preprocess text data first on the Dataset page.")
            return

        df = st.session_state["processed_text_df"]

        text_col = st.selectbox("Select Text Column:", df.columns)
        label_col = st.selectbox("Select Label Column:", df.columns)

        embedding_dim = st.number_input("Embedding Dimension:", 16, 256, 64)
        lstm_units = st.number_input("LSTM Units:", 16, 256, 64)
        epochs = st.number_input("Training Epochs:", 1, 50, 5)

        if st.button("🚀 Train LSTM Model"):
            with st.spinner("Training LSTM sentiment model..."):
                model, tokenizer, model_path = train_lstm_sentiment(
                    df,
                    text_col=text_col,
                    label_col=label_col,
                    embedding_dim=embedding_dim,
                    lstm_units=lstm_units,
                    epochs=epochs,
                )

                st.session_state["trained_model"] = model
                st.session_state["tokenizer"] = tokenizer
                st.session_state["model_type"] = "LSTM (Sentiment Analysis)"
                st.session_state["model_path"] = model_path

                st.success(f"✅ LSTM model trained successfully! Saved at {model_path}")

    # ------------------------------------------------------------
    st.markdown("---")
    st.info("✅ Once training is complete, go to the **Recommendation Page** for predictions.")
