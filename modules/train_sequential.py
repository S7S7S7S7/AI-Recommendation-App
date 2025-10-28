# =============================================
# modules/train_sequential.py  (Final Unified Pipeline)
# =============================================

import os
import json
import pickle
import pandas as pd
from collections import defaultdict, Counter


# --------------------------------------------------
# 🔹 Helper: Build Ordered Sequences
# --------------------------------------------------
def build_sequences(df: pd.DataFrame, session_col: str, time_col: str, item_col: str):
    """
    Convert raw transaction log into ordered sequences per session/user.
    Example output: [['Milk', 'Bread', 'Butter'], ['Tea', 'Sugar']]
    """
    if not all(col in df.columns for col in [session_col, time_col, item_col]):
        raise ValueError("❌ Please ensure all required columns exist in the dataframe.")

    df_sorted = df[[session_col, time_col, item_col]].dropna()
    df_sorted[time_col] = pd.to_datetime(df_sorted[time_col], errors="coerce")

    grouped = (
        df_sorted.sort_values([session_col, time_col])
        .groupby(session_col)[item_col]
        .apply(list)
    )

    sequences = grouped.tolist()
    if not sequences:
        raise ValueError("⚠️ No valid sequences found. Please check your columns or data.")
    return sequences


# --------------------------------------------------
# 🔹 Helper: Build Transitions (n-gram model)
# --------------------------------------------------
def build_transitions(sequences, order=1, min_support=1):
    """
    Build an n-gram transition dictionary mapping prefix → next item count.
    """
    transitions = defaultdict(Counter)
    for seq in sequences:
        if len(seq) <= order:
            continue
        for i in range(len(seq) - order):
            prefix = tuple(seq[i:i + order])
            next_item = seq[i + order]
            transitions[prefix][next_item] += 1

    # Filter by min_support
    filtered = defaultdict(Counter)
    for prefix, nexts in transitions.items():
        for nxt, cnt in nexts.items():
            if cnt >= min_support:
                filtered[prefix][nxt] = cnt
    return filtered


# --------------------------------------------------
# 🔹 TRAIN FUNCTION
# --------------------------------------------------
def train_sequential(df, session_col, time_col, item_col, order=1, min_support=1, save_dir="models"):
    """
    Train a sequential pattern model using n-gram transitions with encoding.
    Saves: pickle model, item mapping, readable JSON, and metadata.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Step 1: Build ordered sequences
    sequences = build_sequences(df, session_col, time_col, item_col)

    # Step 2: Build encoding maps (item → id, id → item)
    unique_items = pd.Series([i for seq in sequences for i in seq]).unique()
    item_to_id = {str(item): str(idx) for idx, item in enumerate(unique_items)}
    id_to_item = {str(idx): str(item) for item, idx in item_to_id.items()}

    # Save mapping
    mapping_path = os.path.join(save_dir, "item_mapping.pkl")
    with open(mapping_path, "wb") as f:
        pickle.dump({"item_to_id": item_to_id, "id_to_item": id_to_item}, f)

    # Step 3: Encode sequences into ID form
    encoded_sequences = [[item_to_id[str(i)] for i in seq if str(i) in item_to_id] for seq in sequences]

    # Step 4: Build transitions on encoded IDs
    encoded_transitions = build_transitions(encoded_sequences, order, min_support)

    # Step 5: Create human-readable transitions
    readable_model = {}
    for prefix, nexts in encoded_transitions.items():
        prefix_names = tuple(id_to_item[str(p)] for p in prefix)
        next_names = {id_to_item[str(nxt)]: cnt for nxt, cnt in nexts.items()}
        readable_model[str(prefix_names)] = next_names

    # Step 6: Save models
    model_path = os.path.join(save_dir, "sequential_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(encoded_transitions, f)

    readable_json_path = os.path.join(save_dir, "sequential_model_readable.json")
    with open(readable_json_path, "w", encoding="utf-8") as f:
        json.dump(readable_model, f, indent=2, ensure_ascii=False)

    # Step 7: Save metadata
    meta = {
        "order": order,
        "min_support": min_support,
        "num_sequences": len(sequences),
        "unique_items": len(unique_items)
    }
    meta_path = os.path.join(save_dir, "sequential_model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"✅ Sequential model trained successfully!")
    print(f"📦 Saved files:\n - Model: {model_path}\n - Mapping: {mapping_path}\n - Readable: {readable_json_path}")
    return readable_model, model_path, readable_json_path


# --------------------------------------------------
# 🔹 LOAD FUNCTION
# --------------------------------------------------
def load_sequential_model(path="models/sequential_model.pkl"):
    """Load the trained sequential pattern model (encoded transitions)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Model not found at {path}. Train the model first.")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model
