from typing import List

import pandas as pd
import pickle

from pandas.io.formats.style_render import DataFrame

def get_dataset(path: str) -> DataFrame:
    return pd.read_pickle(path + ".pkl")

def save_dataset(dataset: DataFrame, path: str):
    dataset.to_pickle(path + ".pkl")

dataset = get_dataset("cleaned_dataset")
reviews = dataset["review"]

# получаем словарь выборки
def get_dictionary(reviews) -> List[str]:
    dictionary = {}

    for review in reviews:
        for word in review:
            if word in dictionary:
                dictionary[word] += 1
            else:
                dictionary[word] = 1

    dictionary = list(dictionary.items())

    for element in set(dictionary):
        if element[1] == 1:
            dictionary.remove(element)

    dictionary = sorted(set(dictionary), key=itemgetter(1), reverse = True)
    return dictionary

dictionary = get_dictionary(reviews)

words = []
counts = []
for element in dictionary:
    words.append(element[0])
    counts.append(element[1])

dictionary = set(dict(zip(words, counts)))

file = open('dictionary.bin', 'wb')
pickle.dump(dictionary, file)

print(dictionary)