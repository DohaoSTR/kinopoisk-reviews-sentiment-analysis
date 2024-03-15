dataset = get_dataset("cleaned_dataset")
reviews = dataset["review"]

reviewsToVectorizer = []
for review in reviews.tolist():
  reviewsToVectorizer.append(str(review))

vectorizer = TfidfVectorizer(max_features=5500, min_df = 40, max_df = 0.82)
reviews_vectorized = vectorizer.fit_transform(reviewsToVectorizer).toarray()

x_train, x_test, y_train, y_test = train_test_split(reviews_vectorized, dataset["label"], test_size=0.3)

classifier = SVC(kernel = "poly", degree = 2)

classifier.fit(x_train, y_train) 