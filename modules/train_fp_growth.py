import pandas as pd
import pickle
import os
from mlxtend.frequent_patterns import fpgrowth, association_rules


def train_fp_growth(df_encoded, min_support=0.02, min_confidence=0.5, min_lift=1.0, save_path="models/fpgrowth_model.pkl"):
    """
    Trains an FP-Growth model on one-hot encoded data, generates rules,
    and saves both frequent itemsets and rules to disk.
    """

    # Step 1: Ensure binary (0/1) values
    df_encoded = df_encoded.copy()
    df_encoded[df_encoded > 1] = 1

    # Step 2: Find frequent itemsets using FP-Growth
    frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        raise ValueError("No frequent itemsets found. Try lowering min_support.")

    # Step 3: Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules[(rules["confidence"] >= min_confidence) & (rules["lift"] >= min_lift)]

    # Step 4: Convert frozensets to lists for readability
    rules["antecedents"] = rules["antecedents"].apply(list)
    rules["consequents"] = rules["consequents"].apply(list)

    # Step 5: Ensure model folder exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Step 6: Save model
    with open(save_path, "wb") as f:
        pickle.dump({"frequent_itemsets": frequent_itemsets, "rules": rules}, f)

    return rules, save_path


def load_fp_growth_model(path="models/fpgrowth_model.pkl"):
    """Loads a previously saved FP-Growth model."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model["frequent_itemsets"], model["rules"]
