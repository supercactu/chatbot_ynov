import re, os
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dropout, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.naive_bayes import MultinomialNB
from common import *
from config import *

# 1) Prepare data
X_text, y = load_data()
X_text, y = balance_data(X_text, y, max_samples=MAX_SAMPLES_PER_CLASS)
X_temp, X_test, y_temp, y_test = train_test_split(X_text, y, test_size=0.1, random_state=RANDOM_STATE, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2222, random_state=RANDOM_STATE, stratify=y_temp)

# 2) Feature extraction
vec = TfidfVectorizer(max_features=MAX_FEATURES)
X_train_tf = vec.fit_transform(X_train)
X_val_tf = vec.transform(X_val)
X_test_tf = vec.transform(X_test)

# Modèles à comparer
models = {
    'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE),
    'NaiveBayes': MultinomialNB()
}

# Comparaison
best_model = None
best_f1 = 0

print("🔍 Model Comparison (Validation Set):\n")
for name, model in models.items():
    model.fit(X_train_tf, y_train)
    y_val_pred = model.predict(X_val_tf)

    acc = accuracy_score(y_val, y_val_pred)
    rec = recall_score(y_val, y_val_pred, average='macro')
    prec = precision_score(y_val, y_val_pred, average='macro')
    f1 = f1_score(y_val, y_val_pred, average='macro')

    print(f"📌 {name}")
    print(f" - Accuracy : {acc:.4f}")
    print(f" - Recall   : {rec:.4f}")
    print(f" - Precision: {prec:.4f}")
    print(f" - F1-score : {f1:.4f}\n")
    
    if f1 > best_f1:
        best_f1 = f1
        best_model = model

# Résultat
print(f"✅ Best model selected: {best_model.__class__.__name__} with macro F1-score = {best_f1:.4f}")

# Évaluation sur le test set
y_test_pred = best_model.predict(X_test_tf)
print("\n📊 Test set classification report (Best ML Model):")
print(classification_report(y_test, y_test_pred, digits=3))

# Save artifacts
joblib.dump(vec, VECTORIZER_FILE)
joblib.dump(best_model, ML_MODEL_FILE)

# Preparer embeddings pour DL
if not os.path.exists(W2V_FILE):
    glove2word2vec(GLOVE_FILE, W2V_FILE)
wv = KeyedVectors.load_word2vec_format(W2V_FILE)

# DL text to sequence
tok = Tokenizer(num_words=MAX_FEATURES, oov_token='<OOV>')
tok.fit_on_texts(X_text)
seqs = tok.texts_to_sequences(X_text)
X_dl = pad_sequences(seqs, maxlen=MAX_LEN)
y_dl = pd.get_dummies(y).values

# DL splits
X_temp_dl, X_test_dl, y_temp_dl, y_test_dl = train_test_split(X_dl, y_dl, test_size=0.1, random_state=RANDOM_STATE, stratify=y_dl)
X_train_dl, X_val_dl, y_train_dl, y_val_dl = train_test_split(X_temp_dl, y_temp_dl, test_size=0.2222, random_state=RANDOM_STATE, stratify=y_temp_dl)

# Matrice embedding
vocab_size = min(MAX_FEATURES, len(tok.word_index) + 1)
emb_matrix = np.zeros((vocab_size, EMBED_DIM))
for w, i in tok.word_index.items():
    if i < MAX_FEATURES and w in wv:
        emb_matrix[i] = wv[w]

# Model bidirectional LSTM
inp = Input(shape=(MAX_LEN,), name='inputs')
embed = Embedding(vocab_size, EMBED_DIM, weights=[emb_matrix], trainable=True, name='embed')(inp)
x = Bidirectional(LSTM(128, return_sequences=True), name='bilstm1')(embed)
x = Dropout(0.3)(x)
x = Bidirectional(LSTM(64), name='bilstm2')(x)
x = Dense(32, activation='relu', name='dense1')(x)
out = Dense(y_dl.shape[1], activation='softmax', name='predictions')(x)
model = Model(inputs=inp, outputs=out, name='bidir_lstm')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks & training
es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
ck = ModelCheckpoint(DL_MODEL_FILE, monitor='val_loss', save_best_only=True)
model.fit(X_train_dl, y_train_dl, epochs=20, batch_size=64, validation_data=(X_val_dl, y_val_dl), callbacks=[es, ck], verbose=1)

# Evaluate DL
dl_preds = model.predict(X_test_dl, verbose=0)
dl_preds_labels = np.argmax(dl_preds, axis=1)
dl_true_labels = np.argmax(y_test_dl, axis=1)
classes = y.unique()
print("\n📊 Test set classification report (Deep Learning Model):")
print(classification_report(dl_true_labels, dl_preds_labels, target_names=classes, digits=3))
dl_acc = accuracy_score(dl_true_labels, dl_preds_labels)
dl_prec = precision_score(dl_true_labels, dl_preds_labels, average='macro')
dl_rec = recall_score(dl_true_labels, dl_preds_labels, average='macro')
dl_f1 = f1_score(dl_true_labels, dl_preds_labels, average='macro')
print(f"DL test metrics:")
print(f" - Accuracy : {dl_acc:.4f}")
print(f" - Precision: {dl_prec:.4f}")
print(f" - Recall   : {dl_rec:.4f}")
print(f" - F1-score : {dl_f1:.4f}")

# Save tokenizer and classes
joblib.dump(tok, TOKENIZER_FILE)
joblib.dump(y.unique(), CLASSES_FILE)
print("All artifacts saved.")