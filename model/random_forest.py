import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

from pandas.io.formats.style_render import DataFrame

def get_dataset(path: str) -> DataFrame:
    return pd.read_pickle(path + ".pkl")

dataset = get_dataset("cleaned_dataset")
reviews = dataset["review"]

reviews_string = []
for comment in reviews.tolist():
    reviews_string.append(str(comment))

def classification(max_feautures, min_df, max_df, n_estimators):
    vectorizer = TfidfVectorizer(max_features=max_feautures, min_df = min_df, max_df = max_df)
    reviews_vectorized = vectorizer.fit_transform(reviews_string).toarray()

    x_train, x_test, y_train, y_test = train_test_split(reviews_vectorized, dataset["label"], test_size=0.3, shuffle = True)

    classifier = RandomForestClassifier(n_estimators=n_estimators)

    classifier.fit(x_train, y_train) 
    y_pred = classifier.predict(x_test)

    print("Отчёт по классификации: ")
    print(classification_report(y_test, y_pred))
    print("Точность: " + str(accuracy_score(y_test, y_pred)))

    return accuracy_score(y_test, y_pred);

def test():
    accuracy_list = []
    value_list = []

    for value in range(100, 1000, 50):
      accuracy = classification(5500, 40, 0.82, value)
      accuracy_list.append(accuracy)
      value_list.append(value)

    data = np.column_stack((accuracy_list, value_list))
    df = pd.DataFrame(data, columns = ["Точность", "value"])
    print(df)

#classification(5500, 40, 0.82, 1000)
#test()