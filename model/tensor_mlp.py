from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout

import keras

from keras.preprocessing.text import Tokenizer
from pandas.io.formats.style_render import DataFrame
from sklearn.model_selection import train_test_split

import pandas as pd
import pickle
import numpy as np

def get_dataset(path: str) -> DataFrame:
  return pd.read_pickle(path + ".pkl")

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

num_words = 10000

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

epochs = 10

print(u'Собираем модель...')
model = Sequential()
model.add(Dense(512, input_shape=(num_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

history = model.fit(x_train, y_train,
                    batch_size=32,
                    epochs=epochs,
                    verbose=1,)

score = model.evaluate(x_test, y_test,
                       batch_size=32, verbose=1)
print()
print(u'Оценка теста: {}'.format(score[0]))
print(u'Оценка точности модели: {}'.format(score[1]))