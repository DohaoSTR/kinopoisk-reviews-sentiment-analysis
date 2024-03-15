from data_processing import get_cleaned_reviews

from model_training import *

input_review = input("Введите комментарий для оценки: ")
input_cleaned_review = get_cleaned_reviews([input_review])

input_vec = vectorizer.transform([str(input_cleaned_review)]).toarray()

predicted = classifier.predict(input_vec)

if (predicted[0] == "bad"):
    print("Эмоциональная оценка коментария - негативный.")

if (predicted[0] == "good"):
    print("Эмоциональная оценка коментария - позитивный.")
    
if (predicted[0] == "neutral"):
    print("Эмоциональная оценка коментария - нейтральный.")