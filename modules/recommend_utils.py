# ============================================
# recommend_utils.py
# ============================================

from collections import defaultdict
from tensorflow.keras.preprocessing.sequence import pad_sequences


# -----------------------------
# Apriori / FP-Growth Recommendation
# -----------------------------
def recommend_from_rules(user_items, rules_df):
    recommendations = set()
    for _, row in rules_df.iterrows():
        if set(user_items).issubset(row["antecedents"]):
            recommendations.update(row["consequents"])
    return list(recommendations - set(user_items))


# ============================================
# 🔁 Sequential Pattern Matching (Human-readable version)
# ============================================
def recommend_from_patterns(user_items, model, top_n=5):
    """
    Recommend next items directly from the human-readable Sequential Pattern model.
    Works with original item names (no encoding or mapping required).
    """
    from ast import literal_eval

    if not user_items:
        return ["⚠️ Please enter at least one item."]

    # Convert input into tuple format
    prefix = tuple(user_items[-1:])
    prefix_str = str(prefix)

    # 🧩 Try to find a matching key in model (tuple or stringified tuple)
    found_key = None
    for k in model.keys():
        if k == prefix or k == prefix_str:
            found_key = k
            break
        try:
            tup = literal_eval(k)
            if tup == prefix:
                found_key = k
                break
        except Exception:
            continue

    if not found_key:
        return [f"⚠️ No matching pattern found for this sequence ({prefix})."]

    # Retrieve next possible items
    next_items = model[found_key]
    if not next_items:
        return [f"⚠️ No next items found for {prefix}."]

    # Sort and return top-N recommendations
    sorted_items = sorted(next_items.items(), key=lambda x: x[1], reverse=True)
    recommendations = [item for item, _ in sorted_items[:top_n]]

    return recommendations


# ============================================
# 🤖 LSTM Sentiment Analysis
# ============================================
def predict_sentiment(
    text,
    model=None,
    tokenizer=None,
    model_path="models/lstm_model.h5",
    tokenizer_path="models/lstm_tokenizer.pkl",
    encoder_path="models/lstm_encoder.pkl",
    max_length=100
):
    """
    Predict sentiment (or category) using trained LSTM model.
    Works with either preloaded session models or saved files.
    """
    import numpy as np
    import pickle
    from tensorflow.keras.models import load_model

    # -----------------------------
    # Load model + tokenizer if not provided
    # -----------------------------
    if model is None:
        try:
            model = load_model(model_path)
        except Exception as e:
            return f"❌ Model load error: {e}", 0.0

    if tokenizer is None:
        try:
            with open(tokenizer_path, "rb") as f:
                tokenizer = pickle.load(f)
        except Exception as e:
            return f"❌ Tokenizer load error: {e}", 0.0

    try:
        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)
    except Exception:
        encoder = None

    # -----------------------------
    # Tokenize and pad
    # -----------------------------
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_length, padding="post", truncating="post")

    # -----------------------------
    # Predict
    # -----------------------------
    preds = model.predict(padded)
    score = float(preds[0][0]) if preds.shape[-1] == 1 else float(np.max(preds))

    # -----------------------------
    # Decode label
    # -----------------------------
    if encoder is not None and preds.shape[-1] > 1:
        label_idx = int(np.argmax(preds))
        label = encoder.inverse_transform([label_idx])[0]
    else:
        label = "Positive 😀" if score >= 0.5 else "Negative 😞"

    return label, round(score, 3)
