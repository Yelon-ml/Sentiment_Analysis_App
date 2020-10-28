import tensorflow as tf
import keras
import pandas as pd



tweets = pd.read_csv(r"files\Tweets.csv", usecols=['tweet_id', 'airline_sentiment', 'text'])
tweets = tweets[tweets.airline_sentiment != 'neutral']
tweets.reset_index(inplace=True)

reviews = [str(sentence) for sentence in tweets.text]
labels = [1 if label=='positive' else 0 for label in tweets.airline_sentiment]

reviews[:6]

class text_preprocessing:

    def __init__(self, reviews, labels):
        self.reviews = reviews
        self.labels = labels

    def text_cleaning(self, arg):

        import re
        import nltk
        import numpy as np
        from string import punctuation
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        self.reviews = [re.sub("@\S+", "", review) for review in arg]
        print("Removed terms that start with @.")
        self.reviews = [re.sub("(?:(http|s?ftp):\/\/\S+)?", "", review) for review in self.reviews]
        print("Removed terms that start with http://.")
        self.reviews = [re.sub("#\S+", "", review) for review in self.reviews]
        print("Removed terms that start with #.")
        self.reviews = [re.sub("&\S+", "", review) for review in self.reviews]
        print("Removed terms that start with &.")
        self.reviews = [review.lower() for review in self.reviews]
        print("Converted all capital letters to lower ones.")
        self.reviews = [re.sub(r'[\W]', " ", review) for review in self.reviews]
        print("Removed all terms that are not known as true words")
        self.reviews = [re.sub('virginamerica', "", review) for review in self.reviews]
        self.reviews = [re.sub('united', "", review) for review in self.reviews]
        self.reviews = [re.sub('americanair', "", review) for review in self.reviews]
        print("Removed terms that consider air lines.")
        self.reviews = [' '.join(word for word in self.reviews[i].split() if word not in stopwords.words('english')) for i in range(len(self.reviews))]
        print("Removed most common english words.")
        self.reviews = [' '.join([WordNetLemmatizer().lemmatize(word) for word in review.split()]) for review in self.reviews]
        print("Converted all terms into their root form.")

    def text_length(self, arg, show=False, count=False, replace=False, show_final_dim=False):
        self.len_of_review = [len(review.split()) for review in arg]
        self.max_len_of_review = max(self.len_of_review)
        self.min_len_of_review = min(self.len_of_review)
        self.avg_len_of_review = np.float(sum(self.len_of_review) / len(self.reviews))
        if show==True:
            print("\nAverage length of review: {},\nMax length: {},\nMinimum length {}".format(avg_len_of_review, max_len_of_review, min_len_of_review))
        if count==True:
            count_zeros = sum([1 if zero==0 else 0 for zero in self.len_of_review])
            print("\nThere is {} reviews of length zero.".format(count_zeros))
        if replace==True:
            self.reviews_and_labels = pd.DataFrame({'Reviews': self.reviews, 'Labels': self.labels, "Length_of_Review": self.len_of_review})
            self.reviews_and_labels = self.reviews_and_labels[self.reviews_and_labels.Length_of_Review != 0]
            self.reviews = reviews_and_labels.Reviews.to_list()
            self.labels = reviews_and_labels.Labels.to_list()
            if show_final_dim==True:
                print("\nReviews dimensionality: {}\nLabels dimensionality: {}.".format(np.array(self.reviews).shape, np.array(self.labels).shape))

    def dictionary(self, arg):
        from functools import reduce
        self.list_of_words = [[word for word in review.split()] for review in arg]
        self.list_of_words = reduce(lambda x,y: x+y, self.list_of_words)
        self.list_of_words = np.unique(self.list_of_words)
        self.dictio = {word: idx for (idx, word) in enumerate(self.list_of_words)}
        self.dictio_revers = dict((b, a) for (a, b) in self.dictio.items())
        self.encoded_reviews = [[dictio[word] for word in review.split()] for review in arg]

    def length_distribution(self, arg):
        import seaborn as sns
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle("Distribution of sententce's length.")
        sns.distplot(self.len_of_review, axlabel='Length of sentence', kde=False, hist=True, ax=ax1)
        sns.distplot(self.len_of_review, axlabel='Length of sentence', kde=True, hist=True, ax=ax2)
        plt.show()

    def text_padding(self, arg):
        self.seq_len = int(input("Set the length of reviews.\nAll sentences are going to be trim/pad to set value\n: "))
        self.reviews_with_padding = np.array(np.zeros(len(arg)*self.seq_len))
        self.reviews_with_padding = self.reviews_with_padding.reshape(len(arg), self.seq_len)

        for idx, review in enumerate(arg):
            length = len([word for word in review.split()])
            if length < self.seq_len:
                n = self.seq_len - length
                zero_padding = np.zeros(n).tolist()
                zero_padding = zero_padding + [dictio[word] for word in review.split()]
                self.reviews_with_padding[idx] = zero_padding
            else:
                self.reviews_with_padding[idx]= [dictio[word] for word in review.split()[:self.seq_len]]

        self.reviews_with_padding = np.array(self.reviews_with_padding)
        print("All sentences have been trimmed/padded to the set length {}.".format(self.seq_len))

    def train_val_test_split(self, x, y, test_size, val_size):
        from sklearn.model_selection import train_test_split
        x = np.array(x)
        y = np.array(y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=val_size)
        print("Train samples dimensionality: features - {}, labels - {}.".format(self.X_train.shape, self.y_train.shape))
        print("Validation samples dimensionality: features - {}, labels - {}.".format(self.X_val.shape, self.y_val.shape))
        print("Test samples dimensionality: features - {}, labels - {}.".format(self.X_test.shape, self.y_test.shape))



data = text_preprocessing(reviews, labels)
data.text_cleaning(data.reviews)
data.text_length(data.reviews, show=True, count=True, replace=True, show_final_dim=True)
data.dictionary(data.reviews)
data.length_distribution(data.reviews)
data.text_padding(data.reviews)
data.reviews_with_padding.shape
data.train_val_test_split(data.reviews_with_padding, data.labels, test_size=0.05, val_size=0.15)
data.reviews_with_padding[:10]



from sklearn.metrics import accuracy_score





import datetime

def gru_training():
    print("Creating model...")
    inputs = keras.Input(shape=(26,), name='input_layer')
    print(inputs.shape)
    embedding = keras.layers.Embedding(len(list_of_words), 2000)(inputs)
    print(embedding.shape)
    gru_1 = keras.layers.GRU(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)(embedding)
    gru_2 = keras.layers.GRU(128, return_sequences=False, dropout=0.2, recurrent_dropout=0.1)(gru_1)
    drop_1 = keras.layers.Dropout(0.4)(gru_2)
    dense_1 = keras.layers.Dense(64, activation='elu')(drop_1)
    drop_2 = keras.layers.Dropout(0.4)(dense_1)
    dense_2 = keras.layers.Dense(32, activation='elu')(drop_2)
    drop_3 = keras.layers.Dropout(0.4)(dense_2)
    dense_3 = keras.layers.Dense(16, activation='elu')(drop_3)
    drop_4 = keras.layers.Dropout(0.4)(dense_3)
    outputs = keras.layers.Dense(1, activation='sigmoid')(drop_4)

    model_with_gru = keras.Model(inputs=inputs, outputs=outputs, name="sentiment_by_gru")
    return model_with_gru

tf.keras.backend.clear_session()

gru_training()

model_with_gru

model_with_gru.compile(optimizer='RMSProp', loss='binary_crossentropy', metrics=['acc'])
print("\nStart training...")
start_time = datetime.datetime.now()
model_with_gru.fit(X_train, y_train, batch_size=256, epochs=5, validation_data=(X_val, y_val), verbose=2)
time = datetime.datetime.now() - start_time
print("\nTraining took {} seconds.".format(time))
return model_with_gru




model_with_gru.evaluate(X_test, y_test)

def inser_own_input():

    own_input = input("please write a sentence: ")
    own_input_splitted = own_input.split()

    own_input = [re.sub("@", "", word) for word in own_input_splitted]
    own_input = [word.lower() for word in own_input]

    own_input = [re.sub(r'[\W]', "", word) for word in own_input]
    own_input = [re.sub('virginamerica', "", word) for word in own_input]
    own_input = [re.sub('united', "", word) for word in own_input]
    own_input = [re.sub('americanair', "", word) for word in own_input]
    own_input

    own_input = ' '.join(word for word in own_input if word not in stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    own_input = ''.join([lemmatizer.lemmatize(word) for word in own_input])
    #own_input = list(own_input)

    input_with_padding = np.array(np.zeros(seq_len))
    input_with_padding = input_with_padding.reshape(1, 26)
    input_length = len([word for word in own_input.split()])
    if input_length < seq_len:
        n = seq_len - input_length
        zero_padding = list(np.zeros(n))
        zero_padding = zero_padding + [dictio[word] for word in own_input.split()]
        input_with_padding[0] = zero_padding
    print(model.predict(input_with_padding))

    if model.predict_classes(input_with_padding) == 1:
        print("positive")
    else:
        print("negative")

inser_own_input()
