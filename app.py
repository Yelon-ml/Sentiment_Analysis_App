from tkinter import *
from PIL import ImageTk, Image
import re
import numpy as np
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import keras


class App(Frame):

    def __init__(self, master, *args, **kwargs):

        model = tf.keras.models.load_model('best_model')


        self.root = master
        self.root.geometry("720x546")

        self.background_img = Image.open(r"pics\background.jpg")
        self.background_img = ImageTk.PhotoImage(self.background_img)
        self.background_label = Label(self.root, image=self.background_img)
        self.background_label.photo = self.background_img
        self.background_label.grid(row=0, column=0, columnspan=3, rowspan=10)

        self.text_field = Text(self.root, height=6, width=45)
        self.text_field.insert(END, "write a sentence\nand run analyzer")
        self.text_field.grid(row=0, column=1)



        def analyze():

            arg = [self.text_field.get("1.0", "end-1c")]

            with open('files/' + 'dictio' + '.pkl', 'rb') as f:
                self.dictio = pickle.load(f)

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

            self.seq_len = 16

            self.reviews_with_padding = np.array(np.zeros(len(self.reviews)*self.seq_len))
            self.reviews_with_padding = self.reviews_with_padding.reshape(len(self.reviews), self.seq_len)

            for review in self.reviews:
                i=0
                missing_words = []
                for word in review.split():
                    if word not in self.dictio:
                        missing_words.append(word)
                        self.dictio[word] = self.dictio[list(self.dictio.keys())[50+i]]
                        del self.dictio[list(self.dictio.keys())[50+i]]
                        i += 1

                if i != 0:
                    missing_words_string = ' '.join([word for word in missing_words])
                    if len(missing_words) == 1:
                        verb = "does"
                        noun = "The word"
                    else:
                        verb = "do"
                        noun = "Words"

                    error_msg = "Warning!\n\n" + noun + " " + str(missing_words_string.split()) + " " + verb + " not exist in our dicionary.\nThey have been additionally added, but the prediction might be less accurate.\n\nClick to close the window."
                    self.error_window = Toplevel(self.root)
                    self.error_window.title("Error Message")
                    self.error_window.geometry("730x162")

                    def destroy():
                        self.error_window.destroy()

                    self.new_button = Button(self.error_window, text=error_msg, bg='#ad3440', font=('Arial', 16), command=destroy)
                    self.new_button.grid(row=0,column=0)



            for idx, review in enumerate(self.reviews):
                length = len([word for word in review.split()])
                if length < self.seq_len:
                    n = self.seq_len - length
                    zero_padding = np.zeros(n).tolist()
                    zero_padding = zero_padding + [self.dictio[word] for word in review.split()]
                    self.reviews_with_padding[idx] = zero_padding
                else:
                    self.reviews_with_padding[idx]= [self.dictio[word] for word in review.split()[:self.seq_len]]

            self.reviews_with_padding = np.array(self.reviews_with_padding)

            self.prediction = model.predict(self.reviews_with_padding)

            self.sentiment_label = Label(self.root, height=1, width=20, text="", bd=8)
            self.sentiment_label.grid(row=2, column=1)

            if self.prediction > 0.5:
                self.sentiment_label.config(text="positive", background="#24d677", font=("Courier", 18), fg="#000000", width=15)
            else:
                self.sentiment_label.config(text="negative", background="#ad3440", font=("Courier", 18), fg="#ffffff", width=15)

        self.anal_img = Image.open(r"pics\textanalysis.jpg")
        self.anal_img = self.anal_img.resize((150, 150))
        self.anal_img = ImageTk.PhotoImage(self.anal_img)

        self.sentiment_button = Button(self.root, height=150, width=150, command=analyze, image=self.anal_img, background="#875231")
        self.sentiment_button.grid(row=1, column=1)

        self.quit_img = Image.open(r"pics\QuitButton1.jpg")
        self.quit_img = self.quit_img.resize((110, 65))
        self.quit_img = ImageTk.PhotoImage(self.quit_img)

        self.quit_button = Button(self.root, height=65, width=110, command=quit, image=self.quit_img, bd=4)
        self.quit_button.photo = self.quit_img
        self.quit_button.grid(row=9, column=1)



if __name__ == "__main__":
    root = Tk()
    Analyzer = App(root)
    root.title("Sentiment Analyzer")
    root.mainloop()
