# train_dl.py
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dropout, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from config import *
from common import *

def train_dl():
    print("📥 Chargement et préparation des données DL...")
    X_text, y = load_data(sample_size=SAMPLE_SIZE)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_dl = pd.get_dummies(y_enc).values

    tok = Tokenizer(num_words=MAX_FEATURES, oov_token='<OOV>')
    tok.fit_on_texts(X_text)
    seqs = tok.texts_to_sequences(X_text)
    X_seq = pad_sequences(seqs, maxlen=MAX_LEN)

    X_temp_dl, X_test_dl, y_temp_dl, y_test_dl = train_test_split(X_seq, y_dl, test_size=TEST_PCT, random_state=RANDOM_STATE, stratify=y_dl)
    X_train_dl, X_val_dl, y_train_dl, y_val_dl = train_test_split(X_temp_dl, y_temp_dl, test_size=VAL_RELATIVE, random_state=RANDOM_STATE, stratify=y_temp_dl)

    if not os.path.exists(W2V_FILE):
        glove2word2vec(GLOVE_FILE, W2V_FILE)
    wv = KeyedVectors.load_word2vec_format(W2V_FILE)

    vocab_size = min(MAX_FEATURES, len(tok.word_index) + 1)
    emb_matrix = np.zeros((vocab_size, EMBED_DIM))
    for word, idx in tok.word_index.items():
        if idx < vocab_size and word in wv:
            emb_matrix[idx] = wv[word]

    inp = Input(shape=(MAX_LEN,))
    x = Embedding(vocab_size, EMBED_DIM, weights=[emb_matrix], trainable=True)(inp)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64))(x)
    x = Dense(32, activation='relu')(x)
    out = Dense(y_dl.shape[1], activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ck = ModelCheckpoint(DL_MODEL_FILE, monitor='val_loss', save_best_only=True)

    print("\n🎯 Entraînement du modèle LSTM bidirectionnel…")
    model.fit(X_train_dl, y_train_dl, validation_data=(X_val_dl, y_val_dl),
              batch_size=64, epochs=20, callbacks=[es, ck], verbose=1)

    test_metrics = model.evaluate(X_test_dl, y_test_dl, verbose=0)
    print(f"\n📊 Évaluation DL sur test set : Accuracy = {test_metrics[1]:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(tok, TOKENIZER_FILE)
    joblib.dump(le, LABEL_ENCODER_FILE)
    print("✅ Modèle DL et tokenizer sauvegardés.")

if __name__ == "__main__":
    train_dl()
