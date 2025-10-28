# modules/dataset_upload.py
import streamlit as st
import pandas as pd

# --- Import preprocessing modules ---
from modules.preprocess_sequential import preprocess_sequential_ui
from modules.preprocess_tabular import preprocess_tabular_ui

def upload_dataset():
    """
    Handles dataset uploading for Tabular and Sequential/Text data.
    Returns the uploaded dataframe and stores it in session_state.
    """

    st.write("📤 Upload your dataset to explore and visualize it.")

    # --- Tab-like navigation ---
    tabs = ["Tabular Data", "Sequential/Text Data"]
    active_tab = st.session_state.get("active_tab", "Tabular Data")
    selected_tab = st.radio(
        "Select Data Type:",
        tabs,
        index=tabs.index(active_tab),
        horizontal=True
    )
    st.session_state["active_tab"] = selected_tab

    df = None  # initialize dataframe

    # ------------------------------------------------------------------
    # 🧩 TABULAR DATA SECTION
    # ------------------------------------------------------------------
    if selected_tab == "Tabular Data":
        st.subheader("📁 Upload Tabular Data (CSV / Excel)")
        tab_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            key="tabular_uploader"
        )

        if tab_file:
            if tab_file.name.endswith(".csv"):
                df = pd.read_csv(tab_file)
            else:
                df = pd.read_excel(tab_file)

            st.success("✅ Tabular dataset loaded!")
            st.dataframe(df.head())

            # Save to session state
            st.session_state["uploaded_df"] = df
            st.session_state["data_type"] = "Tabular Data"

            # --- Preprocess tabular data ---
            processed_df = preprocess_tabular_ui(df)
            return processed_df

    # ------------------------------------------------------------------
    # 📜 SEQUENTIAL / TEXT DATA SECTION
    # ------------------------------------------------------------------
    elif selected_tab == "Sequential/Text Data":
        st.subheader("📄 Upload Text / Sequential Data (TXT / LOG / JSON / CSV)")

        text_file = st.file_uploader(
            "Upload your file",
            type=["txt", "log", "json", "csv"],
            key="text_uploader"
        )

        if text_file:
            # --- Load file ---
            if text_file.name.endswith(".csv"):
                df = pd.read_csv(text_file)
            elif text_file.name.endswith(".json"):
                df = pd.read_json(text_file, lines=True)
            else:
                text_data = text_file.read().decode("utf-8", errors="ignore")
                lines = [line.strip() for line in text_data.splitlines() if line.strip()]
                df = pd.DataFrame({"text": lines})

            st.success("✅ Sequential/Text dataset loaded!")
            st.dataframe(df.head())

            # Save raw upload
            st.session_state["uploaded_df"] = df
            st.session_state["data_type"] = "Sequential/Text Data"

            # Call preprocessing UI
            from modules.preprocess_sequential import preprocess_sequential_ui
            processed_df = preprocess_sequential_ui(df)

            # ✅ Ensure consistent downstream data
            st.session_state["processed_text_df"] = processed_df
            if "padded_sequences" in st.session_state:
                st.session_state["data_ready_for_model"] = True
                st.success("✅ Text data ready for model training!")
            else:
                st.warning("⚠️ Complete tokenization before training the LSTM.")

            return processed_df
