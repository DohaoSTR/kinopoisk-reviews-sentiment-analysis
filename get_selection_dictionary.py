from operator import itemgetter
from typing import List

import pickle

from get_data_from_files import get_dataset

def get_selection_dictionary(reviews) -> List[str]:
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

if __name__ == "__main__":
    dataset = get_dataset("cleaned_dataset")
    reviews = dataset["review"]

    dictionary = get_selection_dictionary(reviews)

    words = []
    counts = []
    for element in dictionary:
        words.append(element[0])
        counts.append(element[1])

    dictionary = set(dict(zip(words, counts)))

    file = open('dictionary.bin', 'wb')
    pickle.dump(dictionary, file)

    print(dictionary)