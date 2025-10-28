# ============================================
# train_apriori.py
# ============================================

import pandas as pd
import pickle
import os
from mlxtend.frequent_patterns import apriori, association_rules


def train_apriori(df_encoded, min_support=0.02, min_lift=1.0, min_confidence=0.5, save_path="models/apriori_model.pkl"):
    """
    Trains an Apriori model on one-hot encoded data, generates rules,
    and saves both frequent itemsets and rules to disk.
    """
    # Step 1: Find frequent itemsets
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        raise ValueError("No frequent itemsets found. Try lowering min_support.")

    # Step 2: Generate association rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
    rules = rules[(rules["confidence"] >= min_confidence) & (rules["lift"] >= min_lift)]

    # Step 3: Convert frozensets to lists for easier readability
    rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x))
    rules["consequents"] = rules["consequents"].apply(lambda x: list(x))

    # Step 4: Ensure folder exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Step 5: Save model
    with open(save_path, "wb") as f:
        pickle.dump({"frequent_itemsets": frequent_itemsets, "rules": rules}, f)

    return rules, save_path


def load_apriori_model(path="models/apriori_model.pkl"):
    """Loads a previously saved Apriori model."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model["frequent_itemsets"], model["rules"]
