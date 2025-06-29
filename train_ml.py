# train_ml.py
import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.utils import resample
from config import *
from common import *

def train_ml():
    print("📥 Chargement et préparation des données ML...")
    X_text, y = load_data(sample_size=SAMPLE_SIZE)
    X_text, y = balance_data(X_text, y)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=TEST_PCT, random_state=RANDOM_STATE, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VAL_RELATIVE, random_state=RANDOM_STATE, stratify=y_train)
    to_dense = FunctionTransformer(lambda X: X.toarray(), accept_sparse=True)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=MAX_FEATURES, min_df=2, ngram_range=(1, 1))),
        ('to_dense', to_dense),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE))
    ])

    param_grid = [
        {
            'clf': [LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)],
            'clf__C': [1, 5]
        },
        {
            'to_dense': ['passthrough'],
            'clf': [MultinomialNB()],
            'clf__alpha': [0.5, 1.0]
        },
        {
            'clf': [RandomForestClassifier(class_weight='balanced_subsample', random_state=RANDOM_STATE, n_jobs=-1)],
            'clf__n_estimators': [300],
            'clf__max_depth': [None]
        }
    ]

    scorer = make_scorer(f1_score, average='macro')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(pipeline, param_grid, scoring=scorer, cv=cv, n_jobs=-1, verbose=2)

    print("🔧 Entraînement du GridSearchCV ML...")
    grid.fit(X_train, y_train)

    print("\n📊 Validation ML :")
    y_pred_val = grid.predict(X_val)
    print(classification_report(le.inverse_transform(y_val), le.inverse_transform(y_pred_val), digits=4, zero_division=0))

    print("\n🏆 Évaluation sur test set ML :")
    print(classification_report(le.inverse_transform(y_test), le.inverse_transform(grid.predict(X_test)), digits=4, zero_division=0))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(grid.best_estimator_, ML_MODEL_FILE)
    joblib.dump(le, LABEL_ENCODER_FILE)
    joblib.dump(grid.best_estimator_.named_steps['tfidf'], VECTORIZER_FILE)
    print("\n✅ Modèle ML entraîné et sauvegardé.")

if __name__ == "__main__":
    train_ml()
