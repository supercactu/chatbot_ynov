# common.py
import os
import re
import numpy as np
import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from config import *

# Initialisation NLTK (à exécuter une seule fois)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Ressources linguistiques
SW = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Fonctions de prétraitement
def preprocess(text, remove_stopwords=True, lemmatize=True, return_tokens=False):
    text = re.sub(r"[^\w\s]", "", str(text).lower())
    text = re.sub(r"\d+", "", text)
    tokens = word_tokenize(text)
    
    if remove_stopwords:
        tokens = [w for w in tokens if w not in SW]
    
    if lemmatize:
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
    
    return tokens if return_tokens else " ".join(tokens)

# Chargement et préparation des données ---
def load_data(sample_size=None):
    df = pd.read_excel(DATA_FILE, usecols=['category', 'headline', 'short_description'])
    df.dropna(inplace=True)
    df['text'] = df['headline'].astype(str) + ' ' + df['short_description'].astype(str)
    df['clean'] = df['text'].apply(preprocess)
    if sample_size:
        df = df.sample(n=sample_size, random_state=RANDOM_STATE)
    return df['clean'], df['category']


def balance_data(X, y, max_samples=MAX_SAMPLES_PER_CLASS):
    df = pd.DataFrame({'text': X, 'label': y})
    balanced = []
    for label in df['label'].unique():
        subset = df[df['label'] == label]
        if len(subset) > max_samples:
            subset = resample(subset,
                              replace=False,
                              n_samples=max_samples,
                              random_state=RANDOM_STATE)
        balanced.append(subset)
    df_bal = pd.concat(balanced).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return df_bal['text'], df_bal['label']