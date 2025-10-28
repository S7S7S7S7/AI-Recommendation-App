# ============================================
# train_lstm_sentiment.py
# ============================================

import os
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def train_lstm_sentiment(df, text_col, label_col,
                         embedding_dim=64, lstm_units=64, epochs=5,
                         vocab_size=5000, max_length=100,
                         save_dir="models"):
    """
    Train an LSTM model for sentiment / text sequence classification.
    """
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from sklearn.preprocessing import LabelEncoder

    # -----------------------------
    # Encode labels
    # -----------------------------
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[label_col])

    # -----------------------------
    # Tokenize and pad text
    # -----------------------------
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(df[text_col])
    sequences = tokenizer.texts_to_sequences(df[text_col])
    X = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

    # -----------------------------
    # Build model
    # -----------------------------
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer=Adam(1e-3), metrics=["accuracy"])

    # -----------------------------
    # Train model
    # -----------------------------
    es = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    model.fit(X, y, validation_split=0.2, epochs=epochs, batch_size=32, callbacks=[es], verbose=1)

    # -----------------------------
    # Save model + tokenizer + encoder
    # -----------------------------
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "lstm_model.h5")
    model.save(model_path)

    with open(os.path.join(save_dir, "lstm_tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)

    with open(os.path.join(save_dir, "lstm_encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)

    return model, tokenizer, model_path
