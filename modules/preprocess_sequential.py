# ============================================
# preprocessing_sequential_ui.py
# ============================================

import re
import string
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Download NLTK resources (quietly)
# -----------------------------
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# -----------------------------
# Global objects
# -----------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ============================================================
# 🧹 Core Helper Functions
# ============================================================

def clean_text(text: str) -> str:
    """Clean text by removing punctuation, stopwords, and lemmatizing."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    return text


def preprocess_text_data(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Cleans text column and adds 'clean_text'."""
    st.info("🧹 Cleaning text data...")
    df[text_column] = df[text_column].astype(str)
    df["clean_text"] = df[text_column].apply(clean_text)
    st.success("✅ Text cleaned successfully!")
    return df


def tokenize_and_pad(df: pd.DataFrame, text_column: str, vocab_size: int = 5000, max_length: int = 100):
    """Tokenize text and pad sequences."""
    st.info("🔡 Tokenizing and padding text...")
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(df[text_column])

    sequences = tokenizer.texts_to_sequences(df[text_column])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")

    st.success(f"✅ Tokenization complete! Data shape: {padded_sequences.shape}")
    return padded_sequences, tokenizer


def encode_labels(df: pd.DataFrame, label_column: str):
    """Encode target labels."""
    encoder = LabelEncoder()
    df[label_column] = encoder.fit_transform(df[label_column])
    st.success("✅ Label encoding complete!")
    return df, encoder


# ============================================================
# 📄 Streamlit UI Wrapper (like preprocess_tabular_ui)
# ============================================================

def preprocess_sequential_ui(df: pd.DataFrame):
    """Streamlit UI for text/sequential data preprocessing."""
    st.write("### 🧠 Text / Sequential Data Preprocessing")

    # -----------------------------
    # 1️⃣ Select text column
    # -----------------------------
    text_col = st.selectbox("Select the text column:", df.columns)
    df[text_col] = df[text_col].astype(str)

    # -----------------------------
    # 2️⃣ Text Cleaning
    # -----------------------------
    if st.checkbox("🧹 Clean Text Data"):
        df = preprocess_text_data(df, text_col)
        st.dataframe(df.head(5))

    # -----------------------------
    # 3️⃣ Tokenization & Padding
    # -----------------------------
    if st.checkbox("🔡 Tokenize & Pad Sequences"):
        vocab_size = st.slider("Vocabulary Size:", 1000, 20000, 5000, step=1000)
        max_length = st.slider("Max Sequence Length:", 50, 500, 100, step=10)

        X, tokenizer = tokenize_and_pad(df, "clean_text" if "clean_text" in df.columns else text_col,
                                        vocab_size=vocab_size, max_length=max_length)
        st.write("Sample encoded sequence:", X[0][:30])
        st.write("✅ Tokenization Done.")

        st.session_state["padded_sequences"] = X
        st.session_state["tokenizer"] = tokenizer

    # -----------------------------
    # 4️⃣ Optional Label Encoding
    # -----------------------------
    if st.checkbox("🔢 Encode Label Column (Optional)"):
        label_col = st.selectbox("Select label column:", ["None"] + list(df.columns))
        if label_col != "None":
            df, encoder = encode_labels(df, label_col)
            st.session_state["encoder"] = encoder
            st.success(f"✅ Label encoding applied on '{label_col}'")

    # -----------------------------
    # 5️⃣ Save for downstream modules
    # -----------------------------
    st.session_state["processed_text_df"] = df
    st.session_state["data_type"] = "Sequential/Text Data"

    st.success("✅ Sequential/Text preprocessing completed successfully.")
    st.dataframe(df.head(10))

    return df
