from get_data_from_files import get_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

dataset = get_dataset("cleaned_dataset")
reviews = dataset["review"]

list_reviews = []
for review in reviews.tolist():
    list_reviews.append(str(review))

vectorizer = TfidfVectorizer(max_features=5500, 
                             min_df = 40, 
                             max_df = 0.82)
reviews_vectorized = vectorizer.fit_transform(list_reviews).toarray()

x_train, x_test, y_train, y_test = train_test_split(reviews_vectorized, 
                                                    dataset["label"], 
                                                    test_size=0.3)

classifier = SVC(kernel = "poly", degree = 2)

classifier.fit(x_train, y_train) 