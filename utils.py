# utils.py
import re, joblib, numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BartForConditionalGeneration, BartTokenizer, pipeline
import wikipedia
from bs4 import BeautifulSoup
from common import *
from config import *

# Load artifacts
vectorizer = joblib.load(VECTORIZER_FILE)
ml_model   = joblib.load(ML_MODEL_FILE)
tokenizer  = joblib.load(TOKENIZER_FILE)
dl_model   = load_model(DL_MODEL_FILE)
classes_file = joblib.load(CLASSES_FILE)

wikipedia.BeautifulSoup = lambda markup: BeautifulSoup(markup, features="lxml")

# ML classification
def classify_ml(text):
    clean = preprocess(text)
    X     = vectorizer.transform([clean])
    return ml_model.predict(X)[0]

# DL classification
def classify_dl(text):
    clean = preprocess(text)
    seq = tokenizer.texts_to_sequences([clean])
    pad = pad_sequences(seq, maxlen=MAX_LEN)
    preds = dl_model.predict(pad)
    return classes_file[np.argmax(preds)]

# Extraire mots clés
def extract_keywords(text, top_n=10, weight_threshold=0.1):
    clean = preprocess(text)
    X = vectorizer.transform([clean])
    predicted_class = ml_model.predict(X)[0]
    class_idx = np.where(ml_model.classes_ == predicted_class)[0][0]
    coefs = ml_model.coef_[class_idx]
    feature_names = vectorizer.get_feature_names_out()

    word_coefs = [(word, coefs[i]) for i, word in enumerate(feature_names)]
    word_coefs = sorted(word_coefs, key=lambda x: abs(x[1]), reverse=True)

    text_words = set(word_tokenize(clean))

    keywords = [
        word for word, coef in word_coefs 
        if word in text_words and abs(coef) >= weight_threshold
    ][:top_n]

    return keywords

# Summarization
def get_summarizer():
    tok = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    mod = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    return tok, mod
_s_tok, _s_mod = get_summarizer()

def summarize_text(text):
    if len(text.split()) < 20:
        return "Text too short."
    inputs = _s_tok(text, return_tensors='pt', truncation=True, max_length=1024)
    out = _s_mod.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4)
    return _s_tok.decode(out[0], skip_special_tokens=True)

# QA pipeline
bert_qa = pipeline('question-answering', model='bert-large-uncased-whole-word-masking-finetuned-squad')

def qa(text):
    return bert_qa({'question': text, 'context': text})['answer']

# Wikipedia search
def wiki_search(query):
    try:
        # Nettoyage de la requête
        query = str(query).strip()
        if not query:
            return "Requête vide"

        print(f"[INFO] Recherche Wikipedia pour: '{query}'")

        # Configuration Wikipedia
        wikipedia.set_lang("en")

        # Recherche étape par étape
        page = None
        page_title = None
        page_content = None
        search_method = "unknown"

        # 1. Recherche directe sans auto-suggestion
        try:
            print(f"[INFO] Tentative de recherche directe...")
            page = wikipedia.page(query, auto_suggest=False)
            page_title = page.title
            page_content = page.summary
            search_method = "direct"
            print(f"[INFO] Page trouvée: '{page_title}'")

        except wikipedia.exceptions.DisambiguationError as e:
            print(f"[INFO] Désambiguïsation nécessaire. Options: {e.options[:5]}")
            # Sélectionner l'option la plus pertinente (heuristique simple)
            for option in e.options:
                # Prioriser les options contenant le terme exact (case-insensitive)
                if query.lower() in option.lower():
                    try:
                        page = wikipedia.page(option, auto_suggest=False)
                        page_title = page.title
                        page_content = page.summary
                        search_method = "disambiguation"
                        print(f"[INFO] Page sélectionnée: '{page_title}'")
                        break
                    except:
                        continue
            if not page:
                # Si aucune option pertinente, prendre la première
                try:
                    page = wikipedia.page(e.options[0], auto_suggest=False)
                    page_title = page.title
                    page_content = page.summary
                    search_method = "disambiguation_fallback"
                    print(f"[INFO] Page fallback: '{page_title}'")
                except:
                    pass

        except wikipedia.exceptions.PageError:
            print(f"[INFO] Page non trouvée, recherche de suggestions...")
            suggestions = wikipedia.search(query, results=5)
            print(f"[INFO] Suggestions: {suggestions}")
            if suggestions:
                for suggestion in suggestions:
                    try:
                        page = wikipedia.page(suggestion, auto_suggest=False)
                        page_title = page.title
                        page_content = page.summary
                        search_method = "suggestion"
                        print(f"[INFO] Page sélectionnée: '{page_title}'")
                        break
                    except:
                        continue

        # Vérification finale
        if not page_title or not page_content:
            return f"Aucun résultat trouvé pour '{query}'"

        # Limiter le contenu pour le résumé
        content_to_summarize = page_content[:1000]

        # Tentative de reformulation
        try:
            if len(content_to_summarize.split()) >= 15:
                summary = summarize_text(content_to_summarize)
                if summary and summary != "Text too short." and len(summary) > 20:
                    return f"**{page_title}** (méthode: {search_method})\n\n{summary}"
        except Exception as e:
            print(f"[ERROR] Erreur lors du résumé: {e}")

        # Fallback: retourner le contenu brut
        clean_content = page_content[:300]
        if len(page_content) > 300:
            clean_content += "..."
        return f"**{page_title}** (méthode: {search_method})\n\n{clean_content}"

    except Exception as e:
        print(f"[ERROR] Erreur générale: {str(e)}")
        return f"Erreur lors de la recherche: {str(e)}"