import tensorflow as tf
from tensorflow import keras
import nltk
import numpy as np
from nltk.corpus import stopwords


(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(path=r'imdb.npz', seed=42)
word_index = keras.datasets.imdb.get_word_index(path=r"imbd_word_index.json")

class Dataset:

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
            print('Dictionary length fits to the sample')
        else:
            print("Max index in sample: {},\nMax index in dictionary: {}".format(self.sample_max, self.dict_max))
            n = self.sample_max - self.dict_max
            for i in range(n):
                self.reverse_word_index[self.dict_max+i+1]=stopwords.words('english')[0]
            print("{} missing values in dictionary have been replaced by '{}'.".format(n, stopwords.words('english')[0]))

    def sentence_generator(self, arg, arg2):
        self.sentence = []
        for word in arg:
                    self.sentence.append(''.join(self.reverse_word_index[word]))
        if arg2==True:
            return(self.sentence)

    def train_dict(self, arg, arg2):
        self.x_train_dict = []
        for sentence in arg:
            self.x_train_dict.append(' '.join(self.sentence_generator(sentence, True)))
        lengths = [len(i) for i in self.x_train_dict]
        avg =(float(sum(lengths)) / len(lengths))
        print("\nThe dictionary contains {} sentences, with an average length of a sentence equal to {} characters.\nThe longest sentence has {} characters.".format(len(self.x_train_dict), np.round(avg, 0), max(lengths)))
        if arg2==True:
            return(self.x_train_dict)

    def stopping_words(self, arg, arg2):
        self.x_train_removed_stop_words = []
        eng_stop_words = stopwords.words('english')
        for sentence in arg:
            self.x_train_removed_stop_words.append(' '.join([w for w in str(sentence).split() if w not in eng_stop_words]))
        lengths = [len(i) for i in self.x_train_removed_stop_words]
        avg =(float(sum(lengths)) / len(lengths))
        print("\nAfter removing most common english words, the average length of a sentense is {},\nwhile the longest one has {} characters.".format(np.round(avg, 0), max(lengths)))
        if arg2==True:
            return(self.x_train_removed_stop_words)





data_set = Dataset(x_train, y_train, x_test, y_test, word_index)
data_set.reverse_indexing(data_set.word_index)
data_set.dictionary_check(data_set.x_train, data_set.reverse_word_index)
data_set.sentence_generator(data_set.x_train[100], False)
data_set.train_dict(data_set.x_train, False)
data_set.stopping_words(data_set.x_train_dict, False)
