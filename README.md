# Sentiment Analysis App

In this project I have written a recurrent neural network, that was training to the classification task. Training data set contains tweets that recomment or not chosen airlines (they are feedbacks from the journey).

Data set has been preprocessed by deleting special characters, common english words, punctuation, terms around airline names; converting capital letters to lower ones; converting all terms to their root form (lemmatization technique).
Model is able to return distribution of sentences's length - then one can choose optimal length and all reviews are padded/trimmed to be of set length. I choose seq_len = 16.
Model has test accuracy: 0.89 and was too large to include it in this repo (almost 300 MB)

The net uses:
- 2 GRU cells - initially I used LSTM ones, but GRU work much faster without significant loss of accuracy
- 5 dense layers, with elu activation (except the last one with binary sigmoid activation)
- some dropouts set between 0.3 and 0.5
- batch normalization between dense layers
- adam optimizer (RMSProp worked similarly well) and binary_crossentropy loss
- callbacks: early stopping, model checkpoints and tenorboard's logs

In case of input, that contains any strange words (that is not included in initial dictionary), the user is informed by raising error window.

Below some screens:

![App screen1](https://github.com/Yelon-ml/Sentiment_Analysis_App/blob/main/pics/pos.PNG)
![App screen2](https://github.com/Yelon-ml/Sentiment_Analysis_App/blob/main/pics/neg.PNG)
![App screen3](https://github.com/Yelon-ml/Sentiment_Analysis_App/blob/main/pics/error.PNG)
