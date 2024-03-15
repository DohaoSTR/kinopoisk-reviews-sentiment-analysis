import pandas as pd
import pickle

from pandas.io.formats.style_render import DataFrame

import scipy.sparse as sp
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

from get_data_from_files import get_dataset

def classification(max_feautures, min_df, max_df, kernel, degree):
    vectorizer = TfidfVectorizer(max_features=max_feautures, 
                                 min_df = min_df, 
                                 max_df = max_df)
    vectorized_reviews = vectorizer.fit_transform(list_reviews).toarray()

    x_train, x_test, y_train, y_test = train_test_split(vectorized_reviews, 
                                                        dataset["label"], 
                                                        test_size=0.3, 
                                                        shuffle = True)

    classifier = SVC(kernel = kernel, degree = degree)

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
        accuracy = classification(5500, 40, 0.82, "poly", 2)
        accuracy_list.append(accuracy)
        value_list.append(value)

    data = np.column_stack((accuracy_list, value_list))
    df = pd.DataFrame(data, columns = ["Точность", "min_df"])
    print(df)

def get(max_feautures, min_df, max_df, kernel, degree, comment):
    reviews_string[0] = comment
    vectorizer = TfidfVectorizer(max_features=max_feautures, min_df = min_df, max_df = max_df)
    reviews_vectorized = vectorizer.fit_transform(reviews_string).toarray()

    x_train, x_test, y_train, y_test = train_test_split(reviews_vectorized, dataset["label"], test_size=0.3, shuffle = True)

    classifier = SVC(kernel = kernel, degree = degree)
    classifier.fit(x_train, y_train) 

    predicted = classifier.predict(x_test)


if __name__ == "__main__":
    dataset = get_dataset("cleaned_dataset")
    reviews = dataset["review"]

    list_reviews = []
    for review in reviews.tolist():
        list_reviews.append(str(review))
        
    classification(5500, 40, 0.82, "poly", 2)
    test()
    get(5500, 40, 0.82, "poly", 2, comment)