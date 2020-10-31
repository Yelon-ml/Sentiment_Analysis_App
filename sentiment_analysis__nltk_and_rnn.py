
import pandas as pd
import re
import nltk
import numpy as np
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import keras



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
            print("\nAverage length of review: {},\nMax length: {},\nMinimum length {}".format(self.avg_len_of_review, self.max_len_of_review, self.min_len_of_review))
        if count==True:
            count_zeros = sum([1 if zero==0 else 0 for zero in self.len_of_review])
            print("\nThere is {} reviews of length zero.".format(count_zeros))
        if replace==True:
            self.reviews_and_labels = pd.DataFrame({'Reviews': self.reviews, 'Labels': self.labels, "Length_of_Review": self.len_of_review})
            self.reviews_and_labels = self.reviews_and_labels[self.reviews_and_labels.Length_of_Review != 0]
            self.reviews = self.reviews_and_labels.Reviews.to_list()
            self.labels = self.reviews_and_labels.Labels.to_list()
            if show_final_dim==True:
                print("\nReviews dimensionality: {}\nLabels dimensionality: {}.".format(np.array(self.reviews).shape, np.array(self.labels).shape))

    def dictionary(self, arg):
        from functools import reduce
        self.list_of_words = [[word for word in review.split()] for review in arg]
        self.list_of_words = reduce(lambda x,y: x+y, self.list_of_words)
        self.list_of_words = np.unique(self.list_of_words)
        self.dictio = {word: idx for (idx, word) in enumerate(self.list_of_words)}
        self.dictio_revers = dict((b, a) for (a, b) in self.dictio.items())
        self.encoded_reviews = [[self.dictio[word] for word in review.split()] for review in arg]

    def length_distribution(self, arg):
        import seaborn as sns
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle("Distribution of sententce's length.")
        sns.distplot(self.len_of_review, axlabel='Length of sentence', kde=False, hist=True, ax=ax1)
        sns.distplot(self.len_of_review, axlabel='Length of sentence', kde=True, hist=True, ax=ax2)
        plt.show()

    def text_padding(self, arg, seq_len=False):
        if seq_len==False:
            self.seq_len = int(input("Set the length of reviews.\nAll sentences are going to be trim/pad to set value\n: "))
        else:
            self.seq_len = seq_len
        self.reviews_with_padding = np.array(np.zeros(len(arg)*self.seq_len))
        self.reviews_with_padding = self.reviews_with_padding.reshape(len(arg), self.seq_len)

        for idx, review in enumerate(arg):
            length = len([word for word in review.split()])
            if length < self.seq_len:
                n = self.seq_len - length
                zero_padding = np.zeros(n).tolist()
                zero_padding = zero_padding + [self.dictio[word] for word in review.split()]
                self.reviews_with_padding[idx] = zero_padding
            else:
                self.reviews_with_padding[idx]= [self.dictio[word] for word in review.split()[:self.seq_len]]

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

class network:

    def __init__(self, x_train, y_train, list_of_words):

        self.x = x_train
        self.y = y_train
        self.list = list_of_words

        tf.keras.backend.clear_session()

        self.inputs = keras.Input(shape=(self.x.shape[1],), name='input_layer')
        self.embedding = keras.layers.Embedding(len(self.list), 2000)(self.inputs)
        self.gru_1 = keras.layers.GRU(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.1)(self.embedding)
        self.gru_2 = keras.layers.GRU(128, return_sequences=False, dropout=0.3, recurrent_dropout=0.1)(self.gru_1)
        self.drop_1 = keras.layers.Dropout(0.5)(self.gru_2)
        self.dense_1 = keras.layers.Dense(64, activation='elu')(self.drop_1)
        self.batch_1 = keras.layers.BatchNormalization()(self.dense_1)
        self.drop_2 = keras.layers.Dropout(0.5)(self.batch_1)
        self.dense_2 = keras.layers.Dense(32, activation='elu')(self.drop_2)
        self.batch_2 = keras.layers.BatchNormalization()(self.dense_2)
        self.drop_3 = keras.layers.Dropout(0.5)(self.batch_2)
        self.dense_3 = keras.layers.Dense(16, activation='elu')(self.drop_3)
        self.batch_3 = keras.layers.BatchNormalization()(self.dense_3)
        self.drop_4 = keras.layers.Dropout(0.4)(self.batch_3)
        self.outputs = keras.layers.Dense(1, activation='sigmoid')(self.drop_4)

        self.model = keras.Model(inputs=self.inputs, outputs=self.outputs, name="sentiment_analysis")

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    def train(self, x_val, y_val, batch_size, epochs):

        import datetime

        checkpoint_filepath = r"files\checkpoints"
        logs_filepath = r"files\tensorboard_logs"

        self.callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_acc", patience=6, min_delta=0),
        keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_acc', save_best_only=True, mode='max'),
        keras.callbacks.TensorBoard(log_dir=logs_filepath, write_graph=True, write_images=True, update_freq='epoch')
        ]

        print("Start training...")
        start_time = datetime.datetime.now()
        self.model.fit(self.x, self.y, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val), verbose=2, callbacks=self.callbacks)
        time = datetime.datetime.now() - start_time
        print("\nTraining took {}.".format(time))

    def evaluate(self, x_test, y_test):
        self.result = self.model.evaluate(x_test, y_test)
        print("Test loss: {}, test accuracy: {}.".format(self.result[0], self.result[1]))

    def save_model(self):
        self.model.save('best_model')

    def plot(self):
        keras.utils.plot_model(net.model, "model.png",  show_shapes=True)

net = network(data.X_train, data.y_train, data.list_of_words)
net.train(data.X_val, data.y_val, 256, 30)
net.evaluate(data.X_test, data.y_test)
net.save_model()
net.plot()


class custom_input:

    def __init__(self):
        self.input = [input("write a sentence")]

    def preprocess_input(self, arg):
        text_preprocessing.text_cleaning(self, arg)
        self.clean_input = self.reviews

    def input_padding(self, arg, seq_len):
        self.dictio = data.dictio
        text_preprocessing.text_padding(self, arg, seq_len)

    def predict_sentiment(self, arg):
        self.prediction = net.model.predict(arg)
        if self.prediction > 0.5:
            print("\n\nSentiment: Positive")
        else:
            print("\n\nSentiment: Negative")


def execute():
    own_input = custom_input()
    own_input.preprocess_input(own_input.input)
    own_input.input_padding(own_input.clean_input, data.seq_len)
    own_input.predict_sentiment(own_input.reviews_with_padding)

execute()

import pickle

def save_obj(obj, name):
    with open('files/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

save_obj(data.dictio, 'dictio')
