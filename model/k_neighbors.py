import pandas as pd
import pickle
import numpy as np

from pandas.io.formats.style_render import DataFrame

import scipy
from scipy import _lib

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, average_precision_score
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
%matplotlib inline

def get_dataset(path: str) -> DataFrame:
    return pd.read_pickle(path + ".pkl")

dataset = get_dataset("cleaned_dataset")
reviews = dataset["review"]

newComments = []
for comment in reviews.tolist():
    newComments.append(str(comment))

def classification(max_feautures, min_df, max_df, n_neighbors):
    vectorizer = TfidfVectorizer(max_features=max_feautures, min_df = min_df, max_df = max_df)
    reviews_vectorized = vectorizer.fit_transform(newComments).toarray()

    x_train, x_test, y_train, y_test = train_test_split(reviews_vectorized, dataset["label"], test_size=0.3, shuffle = True)

    classifier = KNeighborsClassifier(n_neighbors = n_neighbors)

    classifier.fit(x_train, y_train) 
    y_pred = classifier.predict(x_test)

    print("Отчёт по классификации: ")
    print(classification_report(y_test, y_pred))
    print("Точность: " + str(accuracy_score(y_test, y_pred)))

    return accuracy_score(y_test, y_pred)

def test():
    accuracy_list = []
    value_list = []

    for value in range(1000, 11000, 500):
        accuracy = classification(value, 40, 0.82, 83)
        accuracy_list.append(accuracy)
        value_list.append(value)

    data = np.column_stack((accuracy_list, value_list))
    df = pd.DataFrame(data, columns = ["Точность", "max_feautures"])
    print(df)

def test_n():
    vectorizer = TfidfVectorizer(max_features=5500, min_df = 40, max_df = 0.82)
    reviews_vectorized = vectorizer.fit_transform(reviews_string).toarray()

    x_train, x_test, y_train, y_test = train_test_split(reviews_vectorized, dataset["label"], test_size=0.3, shuffle = True)

    error_rates = []
    for i in np.arange(1, 101):
        new_model = KNeighborsClassifier(n_neighbors = i)
        new_model.fit(x_train, y_train)
        new_predictions = new_model.predict(x_test)
        error_rates.append(np.mean(new_predictions != y_test))
    plt.plot(error_rates)

#classification(5500, 40, 0.82, 82)
#test_n()
#test()