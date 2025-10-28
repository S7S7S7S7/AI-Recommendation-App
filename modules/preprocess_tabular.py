# ============================================
# preprocess_tabular.py
# ============================================

import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from mlxtend.preprocessing import TransactionEncoder

# -----------------------------
# Handle Missing Values
# -----------------------------
def fill_missing(df, method="Mean"):
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            if method == "Mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif method == "Median":
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df


# -----------------------------
# Remove Outliers
# -----------------------------
def remove_outliers(df):
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    return df


# -----------------------------
# Encode Categorical Columns
# -----------------------------
def encode_data(df):
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    le_dict = st.session_state.get("label_encoders", {})

    for col in cat_cols:
        if col in le_dict:
            le = le_dict[col]
            df[col] = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            le_dict[col] = le

    st.session_state["label_encoders"] = le_dict
    return df, le_dict


# -----------------------------
# Transaction Encoder for Association Rules
# -----------------------------
def transaction_encode(df):
    """Automatically detect and convert transaction-style dataset."""
    # Detect if dataset has list-like column
    list_cols = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, (list, set, tuple))).any()]

    if list_cols:
        st.info(f"🧺 Detected transaction data in column: {list_cols[0]}")
        transactions = df[list_cols[0]].tolist()
    else:
        # Assume transactional rows, group items by Transaction_ID or Customer_ID
        possible_keys = [col for col in df.columns if "id" in col.lower() or "transaction" in col.lower()]
        if possible_keys:
            key_col = possible_keys[0]
            item_col = [col for col in df.columns if col != key_col][-1]
            transactions = df.groupby(key_col)[item_col].apply(list).tolist()
            st.info(f"🧺 Grouped transactional data by '{key_col}'")
        else:
            st.warning("⚠️ Could not detect transaction format. Provide explicit transaction and item columns.")
            return df

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # Save mapping for inverse transform or display
    st.session_state["transaction_encoder"] = te
    st.success("✅ Transaction encoding completed successfully.")
    return df_encoded


# -----------------------------
# Correlation Heatmap
# -----------------------------
def display_correlation_heatmap(df):
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) == 0:
        st.warning("No numeric columns available for correlation heatmap.")
        return
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)


# -----------------------------
# Streamlit UI Wrapper
# -----------------------------
def preprocess_tabular_ui(df):
    st.write("### 🛠️ Tabular / Transactional Data Preprocessing")

    # Detect if dataset contains transactional structure
    if any(df[col].apply(lambda x: isinstance(x, (list, set, tuple))).any() for col in df.columns):
        # Type 2 format: list of items in one column
        if st.checkbox("Apply Transaction Encoder for List-based Data 🧺"):
            df = transaction_encode(df)
    
    elif st.checkbox("Detect & Convert Transactional Rows (Apriori/FPGrowth) 🧾"):
        st.warning("⚠️ This option is **only for Apriori or FP-Growth algorithms**. Please uncheck it for other models.")
        df = transaction_encode(df)

    # -----------------------------
    # Handle Missing / Outliers
    # -----------------------------
    if st.checkbox("Show Missing Values ⚠️"):
        missing_df = pd.DataFrame({
            "Column": df.columns,
            "Missing Values": df.isnull().sum(),
            "Missing %": (df.isnull().sum() / len(df) * 100)
        })
        st.dataframe(missing_df[missing_df["Missing Values"] > 0])

    if st.checkbox("Fill Missing Values 🧮"):
        method = st.radio("Method:", ["Mean", "Median", "Mode"], horizontal=True)
        df = fill_missing(df, method)
        st.success("✅ Missing values filled")

    if st.checkbox("Remove Outliers 📊"):
        df = remove_outliers(df)
        st.success("✅ Outliers removed")

    # -----------------------------
    # Label Encoding (optional now)
    # -----------------------------
    st.markdown("---")
    st.subheader("🔢 Label Encoding Control")

    if st.checkbox("Apply Label Encoding (for non-sequential models)"):
        df, le_dict = encode_data(df)
        st.success("✅ Encoding complete. Mappings saved for future use.")
    else:
        st.info("ℹ️ Skipped label encoding — keep original names for Sequential Pattern models.")

    # -----------------------------
    # Correlation Heatmap (Optional)
    # -----------------------------
    if st.checkbox("Show Correlation Heatmap 🔗"):
        display_correlation_heatmap(df)

    # Always save processed df to session_state
    st.session_state["processed_tabular_df"] = df
    st.session_state["data_type"] = "Tabular Data"

    return df

