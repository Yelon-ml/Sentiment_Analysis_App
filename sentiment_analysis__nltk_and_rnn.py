import tensorflow as tf
import pandas as pd
import re
from string import punctuation
import nltk
import numpy as np
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from collections import Counter
from functools import reduce
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split


tweets = pd.read_csv(r"Tweets.csv", usecols=['tweet_id', 'airline_sentiment', 'text'])
tweets = tweets[tweets.airline_sentiment != 'neutral']
tweets.reset_index(inplace=True)

reviews, labels = [], []
for i in range(len(tweets)):
    reviews.append(str(tweets.text[i]))
    labels.append(1 if tweets.airline_sentiment[i]=="positive" else 0)

reviews = [re.sub("@", " ", review) for review in reviews]
reviews = [review.lower() for review in reviews]
reviews = [re.sub(r'[\W]', " ", review) for review in reviews]
reviews = [re.sub('virginamerica', "", review) for review in reviews]
reviews = [re.sub('united', "", review) for review in reviews]
reviews = [re.sub('americanair', "", review) for review in reviews]
#reviews = [re.sub("  ", "", review) for review in reviews]
for i in range(len(reviews)):
    reviews[i] = ' '.join(word for word in reviews[i].split() if word not in stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
reviews = [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in reviews]


len_of_review = [len(review.split()) for review in reviews]
max_len_of_review = max(len_of_review)
min_len_of_review = min(len_of_review)
avg_len_of_review = np.float(sum(len_of_review) / len(reviews))

print("Average length of review: {},\nMax length: {},\nMinimum length {}".format(avg_len_of_review, max_len_of_review, min_len_of_review))

reviews_and_labels = pd.DataFrame({'Reviews': reviews, 'Labels': labels, "Length_of_Review": len_of_review})
reviews_and_labels = reviews_and_labels[reviews_and_labels.Length_of_Review != 0]

reviews_and_labels.head()

reviews = list(reviews_and_labels.Reviews)
labels = list(reviews_and_labels.Labels)
len(reviews)
len(labels)

list_of_words = []
for review in reviews:
    list_of_words.append([word for word in review.split()])

list_of_words = reduce(lambda x,y: x+y, list_of_words)
list_of_words = np.unique(list_of_words)

dictio = {word: idx for (idx, word) in enumerate(list_of_words)}
dictio_revers = dict((b, a) for (a, b) in dictio.items())

encoded_reviews = []
for review in reviews:
    encoded_reviews.append([dictio[word] for word in review.split()])

seq_len = max_len_of_review
reviews_with_padding = []

for review in reviews:
    length = len([word for word in review.split()])
    if length < seq_len:
        n = seq_len - length
        zero_padding = list(np.zeros(n))
        zero_padding = zero_padding + [dictio[word] for word in review.split()]
        reviews_with_padding.append(zero_padding)
    else:
        reviews_with_padding.append(review)



reviews_with_padding = np.array(reviews_with_padding)
reviews_with_padding.shape
len(labels)

X_train, X_test, y_train, y_test = train_test_split(reviews_with_padding, np.array(labels), test_size=0.1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)
X_train.shape
y_train.shape
