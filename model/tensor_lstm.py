from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

import keras

from keras.preprocessing.text import Tokenizer
from pandas.io.formats.style_render import DataFrame
from sklearn.model_selection import train_test_split

import pandas as pd
import pickle
import numpy as np

from get_data_from_files import get_dataset

if __name__ == "__main__":
    dataset = get_dataset("cleaned_dataset")
    reviews = dataset["review"]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(reviews.tolist())
    textSequences = tokenizer.texts_to_sequences(reviews.tolist())

    labels = dataset["label"]
    labels[labels == "bad"] = 0
    labels[labels == "neutral"] = 1
    labels[labels == "good"] = 2

    x_train, x_test, y_train, y_test = train_test_split(textSequences, labels, test_size=0.3, shuffle = True)

    num_words = 1000

    print(u'Преобразуем описания заявок в векторы чисел...')
    tokenizer = Tokenizer(num_words=num_words)
    x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
    x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
    print('Размерность X_train:', x_train.shape)
    print('Размерность X_test:', x_test.shape)

    print(u'Преобразуем категории в матрицу двоичных чисел '
          u'(для использования categorical_crossentropy)')
    y_train = keras.utils.to_categorical(y_train, 3)
    y_test = keras.utils.to_categorical(y_test, 3)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)

    max_words = 0
    for desc in reviews.tolist():
        words = len(desc)
        if words > max_words:
            max_words = words
    print('Максимальное количество слов в самом длинном описании заявки: {} слов'.format(max_words))

    maxSequenceLength = max_words

    print(u'Собираем модель...')
    model = Sequential()
    model.add(Embedding(1000, maxSequenceLength))
    model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print (model.summary())

    batch_size = 32
    epochs = 3

    print(u'Тренируем модель...')
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test,
                          batch_size=batch_size, verbose=1)
    print()
    print(u'Оценка теста: {}'.format(score[0]))
    print(u'Оценка точности модели: {}'.format(score[1]))