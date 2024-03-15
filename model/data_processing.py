from typing import List

import pandas as pd
from pandas.io.formats.style_render import DataFrame

import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import spacy
import pymorphy2

nltk.download('punkt')
nltk.download('stopwords')
nlkt_stopwords = set(stopwords.words('russian'))

sp = spacy.load('ru_core_news_sm')
spacy_stopwords = sp.Defaults.stop_words

stemmer = SnowballStemmer(language="russian")

morph_analyzer = pymorphy2.MorphAnalyzer()

def get_dataset(path: str) -> DataFrame:
    return pd.read_pickle(path + ".pkl")

def save_dataset(dataset: DataFrame, path: str):
    dataset.to_pickle(path + ".pkl")

def save_dataset_xlsx(dataset: DataFrame, path: str):
    dataset.to_excel(path + ".xlsx")

def get_words_without_symbols(review: str) -> List[str]:
    text_words = word_tokenize(review)
    review = [word.lower() for word in text_words if word.isalpha()]

    return review

def get_words_without_stop_words(review: List[str]) -> List[str]:
    review = [word for word in review if word not in nlkt_stopwords]
    review = [word for word in review if word not in spacy_stopwords]

    return review

def get_lemmed_words(review: List[str]) -> List[str]:
    review = [morph_analyzer.parse(word)[0].normal_form for word in review]

    return review

def get_stemmed_words(review: List[str]) -> List[str]:
    review = [stemmer.stem(word) for word in review]

    return review

def get_cleaned_words(review: str) -> List[str]:
    review = get_words_without_symbols(review)
    review = get_words_without_stop_words(review)

    review = get_lemmed_words(review)
    review = get_stemmed_words(review)

    review = [word for word in review if len(word) > 3]

    return review

def get_cleaned_reviews(reviews):
    cleaned_reviews = []

    for review in reviews:
        words = get_cleaned_words(review)
        cleaned_reviews.append(words)

    return cleaned_reviews

if __name__ == "__main__":
    dataset = get_dataset("dataset")
    reviews = dataset["review"]

    cleaned_reviews = get_cleaned_reviews(reviews)
    data = np.column_stack((cleaned_reviews, dataset["label"]))

    dataset = pd.DataFrame(data, columns = ["review", "label"])

    save_dataset(dataset, "cleaned_dataset")
    save_dataset_xlsx(dataset, "cleaned_dataset")