# Projet de Classification et Analyse de Texte

Ce projet propose un pipeline complet pour la classification de textes, l'extraction de mots-clés, le résumé automatique, la recherche d'informations sur Wikipedia et la réponse à des questions.  
Il combine des approches de Machine Learning (ML) classiques et de Deep Learning (DL) avec des modèles pré-entraînés de NLP (BART, BERT).

---

## Lien du dépôt GitHub

Le projet est disponible ici : [https://github.com/supercactu/chatbot_ynov](https://github.com/supercactu/chatbot_ynov)

---

## Remarque importante sur le dossier `data`

Le dossier `data/` contient des fichiers très volumineux (notamment les données brutes et les embeddings).  
**Ces fichiers ne sont pas inclus dans ce dépôt Git** pour éviter de surcharger le dépôt.  
Si tu as besoin de ces données, il faudra les obtenir séparément ou les générer localement.

---

## Structure du projet

- `config.py` : Configuration des chemins, constantes et paramètres.
- `common.py` : Fonctions de prétraitement du texte (nettoyage, tokenization, lemmatisation).
- `utils.py` : Fonctions principales pour la classification, extraction de mots-clés, résumé, QA, recherche Wikipedia.
- `app.py` : Application web Flask pour interagir avec les fonctionnalités via une interface utilisateur.
- `data/` : Contient les fichiers de données (`huffpost.xlsx`) et d'embeddings (`glove.6B.100d.txt`). **Ce dossier est ignoré par Git à cause de sa taille.**
- `models/` : Contient les modèles sauvegardés (vectorizer, modèles ML et DL, tokenizer).

---

## Installation

1. Cloner ce dépôt :

```bash
git clone https://github.com/supercactu/chatbot_ynov.git
cd chatbot_ynov
