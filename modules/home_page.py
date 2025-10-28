# ============================================
# home_page.py
# ============================================

import streamlit as st
import pandas as pd

def home_page():
    st.title("🤖 Empowering Insight through AI")
    st.subheader("Welcome to the AI-Driven Recommendation System")

    # --------------------------------------------
    # App Introduction
    # --------------------------------------------
    st.write("""
    This app helps you **analyze customer purchase patterns**, **predict the next likely products**,  
    and perform **sentiment analysis** on customer reviews to understand user opinions better.
    """)


    st.markdown("---")

    # --------------------------------------------
    # Interactive Algorithm Explanation
    # --------------------------------------------
    st.subheader("💡 Choose an Algorithm to Learn About It")

    algorithms = ["Apriori / FP-Growth", "Sequential Pattern Mining", "LSTM"]
    selected_algo = st.radio("Select Algorithm:", algorithms, horizontal=True)

    if selected_algo == "Apriori / FP-Growth":
        st.markdown("""
        **Purpose:** Find frequently bought together items.  
        **How it works:** Scans transactions to discover itemsets that appear together often.  
        **Output Example:**  
        - `Milk -> Bread`  
        - `Diaper -> Beer -> Chips`  
        **Best Use Case:** Small to medium datasets, frequent item analysis.
        """)
    elif selected_algo == "Sequential Pattern Mining":
        st.markdown("""
        **Purpose:** Discover common purchase sequences.  
        **How it works:** Finds patterns of items bought in order over time.  
        **Output Example:**  
        - `A -> B -> C`  
        - `Bread -> Butter -> Jam`  
        **Best Use Case:** Users with repeated purchase sequences.
        """)
    elif selected_algo == "LSTM":
        st.markdown("""
        **Purpose:**  
        - Analyze customer reviews to classify sentiment as **Positive, Neutral, or Negative**.

        **How it works:**  
        - Tokenizes and encodes review text into sequences.  
        - Uses an LSTM neural network to learn the context and order of words.  
        - Decodes the output back into human-readable sentiment labels.

        **Output Example:**  
        - Review: "I love this product!" → Sentiment: Positive  
        - Review: "Battery drains too fast" → Sentiment: Negative  
        - Review: "Average quality, okay for daily use" → Sentiment: Neutral

        **Best Use Case:**  
        - Understanding customer feedback from textual reviews.  
        - Enhancing recommendation systems with sentiment-based insights.
        """)


    st.markdown("---")

    # --------------------------------------------
    # Step-by-step User Guide
    # --------------------------------------------
    st.subheader("📌 How to Use This App")
    st.markdown("""
    1. Go to the **Dataset** page and upload your dataset (CSV, Excel, TXT, JSON).  
    2. Preprocess the data on the **Preprocessing** section if required.  
    3. Select an algorithm and set parameters on the **Model** page.  
    4. Train the model and view recommendations.  
    5. Explore insights and top patterns.
    """)

    # --------------------------------------------
    # Quick Tips / Recommendations
    # --------------------------------------------
    st.subheader("💡 Tips for Choosing an Algorithm")
    st.info("Apriori / FP-Growth: Works well with small-medium datasets and frequent itemsets.")
    st.info("Sequential Pattern Mining: Ideal for repeated purchase sequences.")
    st.info("LSTM: Use for long sequences and large datasets with complex patterns.")

    st.markdown("---")

    # --------------------------------------------
    # Sample Dataset per Algorithm (Collapsible & Downloadable)
    # --------------------------------------------
    st.subheader("🧩 Sample Transaction Data")
    import pandas as pd
    if selected_algo in ["Apriori / FP-Growth"]:
        with st.expander("📦 Sample Data for Apriori / FP-Growth"):
            sample_df = pd.DataFrame({
                "UserID": [1, 1, 2, 2, 3],
                "Item": ["Milk", "Bread", "Bread", "Butter", "Milk"]
            })
            st.table(sample_df)
            st.download_button(
                label="Download Sample CSV",
                data=sample_df.to_csv(index=False),
                file_name="apriori_sample.csv",
                mime="text/csv"
            )

    elif selected_algo == "Sequential Pattern Mining":
        with st.expander("📦 Sample Data for Sequential Pattern Mining"):
            sample_df = pd.DataFrame({
            "user_id": ["U1", "U1", "U1", "U2", "U2", "U2", "U3", "U3", "U3", "U4", "U4", "U4"],
            "item_name": [
                "Milk", "Bread", "Butter",   # U1
                "Milk", "Cereal", "Milkshake",  # U2
                "Bread", "Butter", "Jam",   # U3
                "Butter", "Jam", "Coffee"   # U4
            ],
            "timestamp": [
                "2024-01-01 09:00", "2024-01-01 09:05", "2024-01-01 09:10",
                "2024-01-02 10:00", "2024-01-02 10:05", "2024-01-02 10:10",
                "2024-01-03 11:00", "2024-01-03 11:05", "2024-01-03 11:10",
                "2024-01-04 08:00", "2024-01-04 08:05", "2024-01-04 08:10"
            ]
            })
            st.table(sample_df)
            st.download_button(
                label="Download Sample CSV",
                data=sample_df.to_csv(index=False),
                file_name="sequential_sample.csv",
                mime="text/csv"
            )

    elif selected_algo == "LSTM":
        with st.expander("📦 Sample Data for LSTM Sentiment Analysis"):
            import pandas as pd

            sample_df = pd.DataFrame({
                "UserID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "Review_Text": [
                    "I love this product, it works perfectly!",
                    "Battery drains too fast, not satisfied at all.",
                    "Average quality, okay for daily use.",
                    "The sound quality is amazing and very clear.",
                    "Totally disappointed, the item stopped working in a week.",
                    "Excellent performance and great value for money!",
                    "The design is nice but the material feels cheap.",
                    "Very comfortable and easy to use, highly recommend!",
                    "Not worth the price, expected better performance.",
                    "Decent product, neither too good nor too bad."
                ],
                "Sentiment": [
                    "Positive",
                    "Negative",
                    "Neutral",
                    "Positive",
                    "Negative",
                    "Positive",
                    "Neutral",
                    "Positive",
                    "Negative",
                    "Neutral"
                ]
            })

            print(sample_df)

            st.table(sample_df)
            st.download_button(
                label="Download Sample CSV",
                data=sample_df.to_csv(index=False),
                file_name="lstm_sentiment_sample.csv",
                mime="text/csv"
            )