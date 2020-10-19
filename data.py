import tensorflow as tf
from tensorflow import keras
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(path=r'imdb.npz', seed=42)
word_index = keras.datasets.imdb.get_word_index(path=r"imbd_word_index.json")

class Sentiment:

    def __init__(self, x_train, y_train, x_test, y_test, word_index):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.word_index = word_index

    def reverse_indexing(self, dictio):
        self.reverse_word_index = dict((b, a) for (a, b) in dictio.items())

    def dictionary_check(self, sample, dictio):
        self.sample_max = max(list(map(max, sample)))
        self.dict_max = max(list(dictio.keys()))
        if self.dict_max >= self.sample_max:
            print('\n\nDictionary length fits to the sample')
        else:
            print("\n\nMax index in sample: {},\nMax index in dictionary: {}".format(self.sample_max, self.dict_max))
            n = self.sample_max - self.dict_max
            for i in range(n):
                self.reverse_word_index[self.dict_max+i+1]=stopwords.words('english')[0]
            print("\n\n{} missing values in dictionary have been replaced by '{}'.".format(n, stopwords.words('english')[0]))

    def sentence_generator(self, arg, arg2):
        self.sentence = []
        for word in arg:
                    self.sentence.append(''.join(self.reverse_word_index[word]))
        if arg2==True:
            return(self.sentence)

    def train_test_dict(self, arg, arg2, arg3='choose'):
        if arg3=='train':
            self.x_train_dict = []
            for sentence in arg:
                self.x_train_dict.append(' '.join(self.sentence_generator(sentence, True)))
            lengths = [len(i) for i in self.x_train_dict]
            avg =(float(sum(lengths)) / len(lengths))
            print("\n\nThe {} dictionary contains {} sentences, with an average length of a sentence equal to {} characters.\nThe longest sentence has {} characters.".format(arg3, len(self.x_train_dict), np.round(avg, 0), max(lengths)))
            if arg2==True:
                return(self.x_train_dict)
        elif arg3=='test':
            self.x_test_dict = []
            for sentence in arg:
                self.x_test_dict.append(' '.join(self.sentence_generator(sentence, True)))
            lengths = [len(i) for i in self.x_test_dict]
            avg =(float(sum(lengths)) / len(lengths))
            print("\n\nThe {} dictionary contains {} sentences, with an average length of a sentence equal to {} characters.\nThe longest sentence has {} characters.".format(arg3, len(self.x_test_dict), np.round(avg, 0), max(lengths)))
            if arg2==True:
                return(self.x_test_dict)
        else:
            print("Wrong value for arg3")

    def stopping_words(self, arg, arg2, arg3='choose'):
        if arg3=='train':
            self.x_train_removed_stop_words = []
            eng_stop_words = stopwords.words('english')
            for sentence in arg:
                self.x_train_removed_stop_words.append(' '.join([w for w in str(sentence).split() if w not in eng_stop_words]))
            lengths = [len(i) for i in self.x_train_removed_stop_words]
            avg =(float(sum(lengths)) / len(lengths))
            print("\n\nAfter removing most common english words, the average length of a sentense is {},\nwhile the longest one has {} characters.".format(np.round(avg, 0), max(lengths)))
            if arg2==True:
                return(self.x_train_removed_stop_words)
        elif arg3=='test':
            self.x_test_removed_stop_words = []
            eng_stop_words = stopwords.words('english')
            for sentence in arg:
                self.x_test_removed_stop_words.append(' '.join([w for w in str(sentence).split() if w not in eng_stop_words]))
            lengths = [len(i) for i in self.x_test_removed_stop_words]
            avg =(float(sum(lengths)) / len(lengths))
            print("\n\nAfter removing most common english words, the average length of a sentense is {},\nwhile the longest one has {} characters.".format(np.round(avg, 0), max(lengths)))
            if arg2==True:
                return(self.x_test_removed_stop_words)

    def n_gram_vect(self):
        print("\nData preparing...")
        self.vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
        self.vectorizer.fit(self.x_train_removed_stop_words)
        self.X_train = self.vectorizer.transform(self.x_train_removed_stop_words)
        self.X_test = self.vectorizer.transform(self.x_test_removed_stop_words)

    def LogReg(self):
        print("\nTraining...")
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size = 0.2)
        self.logreg = LogisticRegression()
        self.logreg.fit(self.X_train, self.y_train)
        print("\n\nThe accuracy score is equal to {}".format(accuracy_score(self.y_val, self.logreg.predict(self.X_val))))
        return(self.logreg)

    def my_insert(self):
        sample = input("\n\nWrite a sentence: ")
        sample = str(sample)
        sample = [sample]
        self.my_sample = self.vectorizer.transform(sample)
        prediction = self.logreg.predict(self.my_sample)
        if prediction==1:
            print("\nSample is positive")
        else:
            print("\nSample is negative")
        question = input("\n\nDo you want to check new sample? y/n")
        if question=='y':
            self.my_insert()
        else:
            print("\n\nClose the app.")



model = Sentiment(x_train, y_train, x_test, y_test, word_index)
model.reverse_indexing(model.word_index)
model.dictionary_check(model.x_train, model.reverse_word_index)
#data_set.sentence_generator(data_set.x_train[100], False)
model.train_test_dict(model.x_train, False, 'train')
model.train_test_dict(model.x_test, False, 'test')
model.stopping_words(model.x_train_dict, False, 'train')
model.stopping_words(model.x_test_dict, False, 'test')
model.n_gram_vect()
model.LogReg()
model.my_insert()
