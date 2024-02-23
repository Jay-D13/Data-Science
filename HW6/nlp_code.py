import pandas as pd
import nltk
import re
import numpy as np
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

def preprocess_text(text) -> str:
    """
    Prétraite un texte en effectuant des tâches courantes de TALN telles que la suppression HTML,
    la suppression de la ponctuation, la tokenisation, la conversion en minuscules,
    la suppression des stopwords, la lemmatisation et le nettoyage des données.

    Paramètres:
    text (str): Le texte d'entrée à prétraiter.

    Renvois:
    str: Le texte traité après l'application de toutes les tâches de TALN.
    """
    # Initialiser un lemmatiseur WordNet pour obtenir les formes de base des mots et créer un ensemble de stopwords en anglais
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    # Supprimer HTML
    text = re.sub(r'<.*?>', ' ', text)

    # Suppression de la ponctuation : Éliminer les signes de ponctuation
    text = re.sub(r'[^\w\s]', ' ', text)

    # Tokenisation : Diviser le texte en mots
    text = word_tokenize(text)

    # Minuscules : Convertir les mots en minuscules
    text = [word.lower() for word in text]

    # Suppression des stopwords : Supprimer les stopwords courants
    text = [word for word in text if word not in stop_words]

    # Lemmatisation : Appliquer la lemmatisation pour réduire les mots à leur forme de base
    text = [lemmatizer.lemmatize(word) for word in text]

    # Nettoyage des données : Supprimer les jetons vides et effectuer tout nettoyage supplémentaire
    text = [word for word in text if word != '']

    # Joindre les jetons nettoyés pour former un texte traité
    text = ' '.join(text)

    return text


def review_lengths(df) -> pd.Series:
    """
    Calculer le nombre de mots pour chaque élément dans une colonne d'un DataFrame pandas.

    Paramètres:
    - df (pd.Series): Une série pandas contenant les critiques nettoyées.

    Renvois:
    - pd.Series: Une série Pandas contenant le décompte de mots pour chaque élément de la série d'entrée.
    """
    # Diviser le texte en mots et calculer le nombre de mots.

    return df.apply(lambda x: len(x.split()))


def word_frequency(df : pd.Series) -> pd.Series:
    """
    Calculer la fréquence des mots pour une colonne d'un DataFrame pandas.

    Paramètres:
    - df (pd.Series): Une série pandas contenant les critiques nettoyées.

    Renvois:
    pd.Series:
        Une série Pandas contenant les fréquences des mots pour chaque mot de la série d'entrée.
    """
    # Obtenir les fréquences de mots uniques et les renvoyer sous forme de pd.Series ordonnée par ordre décroissant.
    # Cela vous aidera à représenter graphiquement les 20 mots les plus fréquents et les 20 mots les moins fréquents.
    df = df.apply(lambda x: x.split())
    word_counts = {}
    for review in df:
        for word in review:
            if word not in word_counts:
                word_counts[word] = 0
            word_counts[word] += 1
            
    word_counts = pd.Series(word_counts)

    return word_counts.sort_values(ascending=False)

    # I know this can be found on internet: `return df.str.split(expand=True).stack().value_counts()` but I didn't fully understand it at the time of writing my code.


def encode_sentiment(df, sentiment_column='sentiment') -> pd.DataFrame:
    """
    Encoder la colonne de sentiment d'un DataFrame en valeurs numériques.

    Paramètres:
    - df (pd.DataFrame): Le DataFrame contenant la colonne de sentiment.
    - sentiment_column (str): Le nom de la colonne de sentiment. Par défaut, c'est 'sentiment'.

    Renvois:
    - pd.DataFrame: Un nouveau DataFrame avec la colonne de sentiment encodée en valeurs numériques.
    """
    df = df.copy()
    # Encoder nos étiquettes cibles en étiquettes numériques.
    encoder = LabelEncoder()
    encoder.fit(['negative', 'positive']) # 0 = negative, 1 = positive
    df[sentiment_column] = encoder.transform(df[sentiment_column])

    return df


def explain_instance(tfidf_vectorizer, naive_bayes_classifier, X_test, idx, num_features=10):
    """
    Expliquer une instance de texte en utilisant LIME (Local Interpretable Model-agnostic Explanations).

    Paramètres:
    - tfidf_vectorizer (sklearn.feature_extraction.text.TfidfVectorizer): Le vectoriseur TF-IDF entraîné.
    - naive_bayes_classifier (sklearn.naive_bayes.MultinomialNB): Le classificateur Naive Bayes entraîné.
    - X_test (pandas.core.series.Series): La liste des instances de texte à expliquer.
    - idx (int): L'index de l'instance à expliquer.
    - num_features (int, optionnel): Le nombre de caractéristiques (mots) à inclure dans l'explication. Par défaut, c'est 10.

    Renvois:
    - lime.lime_text.LimeTextExplainer: L'objet d'explication contenant des informations sur l'explication de l'instance.
    - float: La probabilité que l'instance soit classée comme 'positive' arrondie à 4 chiffres après la virgule.
    """

    # Créer un pipeline avec le vectoriseur et le modèle entraînés
    pipeline = make_pipeline(tfidf_vectorizer, naive_bayes_classifier)
    
    # Spécifier les noms de classe
    noms = ["negative","positive"] # 0 = negative, 1 = positive
    
    # Créer un LimeTextExplainer
    lte = LimeTextExplainer(class_names=noms)
    
    # Expliquer l'instance à l'index spécifié
    instance = X_test[idx]
    explainer = lte.explain_instance(instance, pipeline.predict_proba, num_features=num_features)

    # Calculer la probabilité que l'instance soit classée comme 'positive'. Arrondir le résultat à 4 chiffres après la virgule.
    proba = pipeline.predict_proba([instance])[0][1]
    proba = round(proba,4)

    
    return explainer, proba




