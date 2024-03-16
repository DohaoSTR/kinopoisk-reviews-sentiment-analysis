import os.path

import numpy as np
import pandas as pd

from enum import Enum
from typing import List, Union

class ReviewSentiment(Enum):
    UNDEFINED = 'undefined'
    BAD = 'bad'
    GOOD = 'good'
    NEUTRAL = 'neutral'

def get_text_data(path: str, review_sentiment: ReviewSentiment = ReviewSentiment.UNDEFINED) -> Union[List[str], List[str]]:
    data_x = []

    for folder_name in os.listdir(path):
        with open(os.path.join(path, folder_name), 'r', encoding='utf-8') as file:
            try:
                review = file.read()
                data_x.append(review)
            except:
                break

    data_y = [review_sentiment.value] * len(data_x)
        
    return data_x, data_y

def get_data_frame() -> pd.DataFrame:
    good_review_folder_name = "good/"
    neutral_review_folder_name = "neutral/"
    bad_review_folder_name = "bad/"

    good_x, good_y = get_text_data(good_review_folder_name, ReviewSentiment.GOOD)
    neutral_x, neutral_y = get_text_data(neutral_review_folder_name, ReviewSentiment.NEUTRAL)
    bad_x, bad_y = get_text_data(bad_review_folder_name, ReviewSentiment.BAD)

    data_x = np.concatenate((good_x, neutral_x, bad_x))
    data_y = np.concatenate((good_y, neutral_y, bad_y))

    data = np.column_stack((data_x, data_y))

    return pd.DataFrame(data, columns = ["review", "label"])

def get_folder_size(path: str) -> int:
    max_size = 0

    for folder_name in os.listdir(path):
        size = os.path.getsize(path + folder_name)

        if (size > max_size):
            max_size = size

    return max_size
      
def get_folder_sizes() -> None:
    print("Максимальная размерность нейтрального комментария - " + str(get_folder_size("neutral/")) + " бит")
    print("Максимальная размерность хорошего комментария - " + str(get_folder_size("good/")) + " бит")
    print("Максимальная размерность плохого комментария - " + str(get_folder_size("bad/")) + " бит")

def save_dataset(dataset: pd.DataFrame, path: str) -> None:
    dataset.to_pickle(path + ".pkl")

def get_dataset(path: str) -> pd.DataFrame:
    return pd.read_pickle(path + ".pkl")

if __name__ == "__main__":
    dataset = get_data_frame()
    save_dataset(dataset, "dataset")