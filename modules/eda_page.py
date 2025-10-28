# ============================================
# eda_page.py (Final - with Timestamp trend fix)
# ============================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def eda_page():
    st.title("📊 Exploratory Data Analysis (EDA)")

    # ---------------------------------------------
    # 1️⃣ Check Dataset
    # ---------------------------------------------
    if "uploaded_df" not in st.session_state:
        st.warning("⚠️ Please upload the dataset first!")
        return

    df = st.session_state["uploaded_df"].copy()
    st.info("Using **Raw Data** for EDA — categorical columns are human-readable.")

    # ---------------------------------------------
    # 2️⃣ Dataset Preview
    # ---------------------------------------------
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(5))
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

    # ---------------------------------------------
    # 3️⃣ Statistical Description
    # ---------------------------------------------
    if st.checkbox("📈 Show Statistical Description"):
        st.dataframe(df.describe())

    # ---------------------------------------------
    # 4️⃣ Detect Data Types
    # ---------------------------------------------
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # Auto-detect and convert date/time columns
    date_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ["date", "time", "timestamp"]):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                if df[col].notna().sum() > 0:
                    date_cols.append(col)
            except Exception:
                pass

    # ---------------------------------------------
    # 5️⃣ Correlation Heatmap
    # ---------------------------------------------
    if len(numeric_cols) > 1:
        st.subheader("🔗 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

    # ---------------------------------------------
    # 7️⃣ Line Chart (Trend Over Time - Fixed)
    # ---------------------------------------------
    st.subheader("📈 Trend Over Time")
    if date_cols:
        date_col = st.selectbox("Select Date/Time Column:", date_cols)

        # Let user choose what to plot
        possible_y_cols = numeric_cols if numeric_cols else cat_cols
        if possible_y_cols:
            y_col = st.selectbox("Select column to analyze trend:", possible_y_cols)

            # Aggregate daily counts or sums
            if y_col in numeric_cols:
                df_trend = df.groupby(df[date_col].dt.date)[y_col].sum().reset_index()
            else:
                df_trend = df.groupby(df[date_col].dt.date)[y_col].count().reset_index()

            df_trend.rename(columns={date_col: "Date", y_col: "Value"}, inplace=True)
            fig, ax = plt.subplots()
            ax.plot(df_trend["Date"], df_trend["Value"], marker="o", color="orange")
            ax.set_title(f"{y_col} Trend Over Time ({date_col})")
            ax.set_xlabel("Date")
            ax.set_ylabel("Value Count")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.info("No valid columns to plot trend.")
    else:
        st.warning("⚠️ No valid date or timestamp column found. Please ensure your date column is in datetime format.")

    # ---------------------------------------------
    # 8️⃣ Bar Chart (Top Categories)
    # ---------------------------------------------
    if cat_cols:
        st.subheader("🏆 Most Frequent Categorical Values")
        selected_cat = st.selectbox("Select column for bar chart:", cat_cols)
        top_n = st.slider("Select top N categories:", 5, 20, 10)
        top_items = df[selected_cat].value_counts().head(top_n)

        fig, ax = plt.subplots()
        sns.barplot(x=top_items.values, y=top_items.index, ax=ax, palette="mako")
        ax.set_title(f"Top {top_n} {selected_cat} Categories")
        st.pyplot(fig)

    # ---------------------------------------------
    # 9️⃣ Pie Chart
    # ---------------------------------------------
    if cat_cols:
        st.subheader("🥧 Category Percentage Share")
        selected_pie_col = st.selectbox("Select column for Pie Chart:", cat_cols)
        top_values = df[selected_pie_col].value_counts().head(10)

        fig, ax = plt.subplots()
        ax.pie(top_values.values, labels=top_values.index, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        ax.set_title(f"Top 10 {selected_pie_col} - Percentage Share")
        st.pyplot(fig)

    st.success("✅ EDA Visualizations Completed Successfully!")
