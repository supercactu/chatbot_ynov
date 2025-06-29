# app.py
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, render_template
import nltk
from utils import classify_ml, classify_dl, extract_keywords, summarize_text, qa, wiki_search

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    txt = request.form['user_input']
    task = request.form['task']
    
    if task == 'classify':
        return f"ML: {classify_ml(txt)}<br>DL: {classify_dl(txt)}"
    
    elif task == 'keywords':
        kws = extract_keywords(txt, top_n=10, weight_threshold=0.1)
        if kws:
            return "Mots‑clés ML : " + ", ".join(kws)
        else:
            return "Aucun mot‑clé trouvé pour ce texte."
    
    elif task == 'summarize':
        return summarize_text(txt)
    
    elif task == 'qa':
        return qa(txt)
    
    elif task == 'wiki':
        res = wiki_search(txt)
        return res or "No Wikipedia info found."
    
    else:
        return "Invalid task", 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')