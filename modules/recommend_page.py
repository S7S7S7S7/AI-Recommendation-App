# ============================================
# recommend_page.py (UPDATED for Sequential Pattern)
# ============================================

import streamlit as st
from modules.recommend_utils import recommend_from_patterns
from modules.recommend_utils import recommend_from_rules
from modules.recommend_utils import predict_sentiment

def recommend_page():
    st.header("🛒 Recommendation Page")
    st.write("Get top-N recommendations or predictions based on the selected algorithm.")

    # ------------------------------------------------------------
    # 1️⃣ Ensure model is trained
    # ------------------------------------------------------------
    if "trained_model" not in st.session_state:
        st.warning("⚠️ Please train a model first on the Model page.")
        st.stop()

    algo = st.session_state.get("model_type", None)
    model = st.session_state.get("trained_model", None)

    # ------------------------------------------------------------
    # 2️⃣ Load processed data safely (avoid truth-value ambiguity)
    # ------------------------------------------------------------
    df = None
    if "processed_tabular_df" in st.session_state and not st.session_state["processed_tabular_df"].empty:
        df = st.session_state["processed_tabular_df"]
    elif "processed_text_df" in st.session_state and not st.session_state["processed_text_df"].empty:
        df = st.session_state["processed_text_df"]
    elif "uploaded_df" in st.session_state and not st.session_state["uploaded_df"].empty:
        df = st.session_state["uploaded_df"]

    if df is None or df.empty:
        st.warning("⚠️ Please upload and preprocess data first on the Dataset page.")
        st.stop()

    # ------------------------------------------------------------
    # 3️⃣ User Input UI
    # ------------------------------------------------------------
    user_input = ""
    top_n = 5

    if algo in ["Apriori", "FP-Growth"]:
        user_input = st.text_input(
            "Enter your purchased items (comma separated):",
            placeholder="Example: Milk, Bread, Butter"
        )
        user_items = [i.strip() for i in user_input.split(",") if i.strip()]
        top_n = st.number_input("Number of recommendations:", min_value=1, max_value=20, value=5)
        
    elif algo == "Sequential Pattern Matching":
        model = st.session_state.get("trained_model", None)

        # ✅ Load model order dynamically
        import os, json
        meta_path = "models/sequential_model_meta.json"
        order = 1
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            order = meta.get("order", 1)

        # 🧾 Dynamic user instruction
        st.markdown("### 🧠 Sequential Pattern Recommendation")
        st.caption(f"💡 Enter the last **{order} item(s)** separated by commas. "
                   f"Example: {', '.join([f'Item{i+1}' for i in range(order)])}")

        # 🔹 Input field
        user_input = st.text_input(
            "Enter your purchased item(s):",
            placeholder="Example: Milk, Bread"
        )
        user_items = [i.strip() for i in user_input.split(",") if i.strip()]
        top_n = st.number_input("Number of recommendations:", min_value=1, max_value=20, value=5)


    elif algo == "LSTM (Sentiment Analysis)":
        user_input = st.text_area(
            "Enter text or sequence for sentiment prediction:",
            placeholder="Example: This product is amazing!"
        )

    # ------------------------------------------------------------
    # 4️⃣ Generate Recommendations
    # ------------------------------------------------------------
    if st.button("🔍 Get Recommendations / Prediction"):
        if not user_input:
            st.warning("⚠️ Please provide input to get results.")
            st.stop()

        recommended_items = []

        # ---------------------------------------------
        # 📊 APRIORI / FP-GROWTH
        # ---------------------------------------------
        if algo in ["Apriori", "FP-Growth"]:
            rules_df = model
            recommended_items = recommend_from_rules(user_items, rules_df)
            recommended_items = list(recommended_items)[:top_n]

        # ---------------------------------------------
        # 🔁 SEQUENTIAL PATTERN MATCHING (FIXED)
        # ---------------------------------------------
        elif algo == "Sequential Pattern Matching":
            # from modules.recommend_utils import recommend_from_patterns
            with st.spinner("Generating recommendations..."):
                recommended_items = recommend_from_patterns(user_items, model, top_n=top_n)

            # ✅ Display result
            if recommended_items and "⚠️" not in recommended_items[0]:
                st.success(f"🎯 Top {top_n} Recommended Next Items:")
                for i, rec in enumerate(recommended_items, start=1):
                    st.write(f"{i}. {rec}")
            else:
                st.warning(recommended_items[0] if recommended_items else "⚠️ No matching pattern found.")


        # ---------------------------------------------
        # 🤖 LSTM SENTIMENT ANALYSIS
        # ---------------------------------------------
        elif algo == "LSTM (Sentiment Analysis)":
            tokenizer = st.session_state.get("tokenizer", None)
            if tokenizer is None:
                st.error("❌ Tokenizer not found. Please preprocess text data first.")
                st.stop()

            sentiment, prob = predict_sentiment(user_input, model, tokenizer)
            st.success(f"✅ Sentiment Prediction: **{sentiment}** (Confidence: {prob:.2f})")
            return  # No recommendation list for LSTM

        # ---------------------------------------------
        # ✅ Final Output Display
        # ---------------------------------------------
        if recommended_items:
            st.success(f"🎯 Top {top_n} Recommendations:")
            st.write(recommended_items)
        else:
            st.info("No recommendations found for the given input.")
