# config.py
import os

# Racine du projet
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dossiers de données et de modèles
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Chemins des fichiers
DATA_FILE = os.path.join(DATA_DIR, 'huffpost.xlsx')
GLOVE_FILE = os.path.join(DATA_DIR, 'glove.6B.100d.txt')
W2V_FILE = os.path.join(DATA_DIR, 'glove.6B.100d.word2vec.txt')

VECTORIZER_FILE = os.path.join(MODEL_DIR, 'vectorizer.pkl')
ML_MODEL_FILE = os.path.join(MODEL_DIR, 'ml_model.pkl')
DL_MODEL_FILE = os.path.join(MODEL_DIR, 'dl_model.keras')
TOKENIZER_FILE = os.path.join(MODEL_DIR, 'tokenizer.pkl')
CLASSES_FILE = os.path.join(MODEL_DIR, 'classes.pkl')

# Paramètres
MAX_FEATURES =20000
MAX_SAMPLES_PER_CLASS = 7000 
MAX_LEN = 50
EMBED_DIM = 100
RANDOM_STATE = 42
SAMPLE_SIZE = 20000