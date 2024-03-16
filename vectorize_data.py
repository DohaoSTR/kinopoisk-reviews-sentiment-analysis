from sklearn.feature_extraction.text import TfidfVectorizer

from get_data_from_files import get_dataset

dataset = get_dataset("cleaned_dataset")
reviews = dataset["review"]

list_reviews = []
for review in reviews.tolist():
    list_reviews.append(str(review))

tfidf_converter = TfidfVectorizer(max_features=5500, 
                                  min_df=40, 
                                  max_df=0.82)
vectorized_data = tfidf_converter.fit_transform(list_reviews).toarray()