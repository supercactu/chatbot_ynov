# Projet de Classification et Analyse de Texte

Ce projet propose un pipeline complet pour la classification de textes, l'extraction de mots-clés, le résumé automatique, la recherche d'informations sur Wikipedia et la réponse à des questions.  
Il combine des approches de Machine Learning (ML) classiques et de Deep Learning (DL) avec des modèles pré-entraînés de NLP (BART, BERT).

---

## Structure du projet

- `config.py` : Configuration des chemins, constantes et paramètres.
- `common.py` : Fonctions de prétraitement du texte (nettoyage, tokenization, lemmatisation).
- `utils.py` : Fonctions principales pour la classification, extraction de mots-clés, résumé, QA, recherche Wikipedia.
- `app.py` : Application web Flask pour interagir avec les fonctionnalités via une interface utilisateur.
- `data/` : Contient les fichiers de données (`huffpost.xlsx`) et d'embeddings (`glove.6B.100d.txt`).
- `models/` : Contient les modèles sauvegardés (vectorizer, modèles ML et DL, tokenizer).

---

## Installation

1. Cloner ce dépôt :

```bash
git clone https://github.com/ton-utilisateur/ton-projet.git
cd ton-projet
