import re

import gensim
import keras
from keras import layers
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import matplotlib.pyplot as plt
from nltk.tokenize.treebank import TreebankWordDetokenizer
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

train = pd.read_csv('./kaggle/input/tweet-sentiment-extraction/train.csv')
train.head(15)
len(train)
train['sentiment'].unique()
train.groupby('sentiment').nunique()

train = train[['selected_text', 'sentiment']]
train.head()

train["selected_text"].isnull().sum()

train["selected_text"].fillna("No content", inplace=True)


def depure_data(data):
    # Removing URLs with a regular expression
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    data = url_pattern.sub(r'', data)

    # Remove Emails
    data = re.sub('\S*@\S*\s?', '', data)

    # Remove new line characters
    data = re.sub('\s+', ' ', data)

    # Remove distracting single quotes
    data = re.sub("\'", "", data)

    return data


temp = []
# Splitting pd.Series to list
data_to_list = train['selected_text'].values.tolist()
for i in range(len(data_to_list)):
    temp.append(depure_data(data_to_list[i]))
list(temp[:5])


def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


data_words = list(sent_to_words(temp))
print(data_words[:10])
len(data_words)


def detokenize(text):
    return TreebankWordDetokenizer().detokenize(text)


data = []
for i in range(len(data_words)):
    data.append(detokenize(data_words[i]))
print(data[:5])

data = np.array(data)

labels = np.array(train['sentiment'])
y = []
for i in range(len(labels)):
    if labels[i] == 'neutral':
        y.append(0)
    if labels[i] == 'negative':
        y.append(1)
    if labels[i] == 'positive':
        y.append(2)
y = np.array(y)
labels = keras.utils.to_categorical(y, 3, dtype="float32")
del y
len(labels)

max_words = 5000
max_len = 200
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)
tweets = pad_sequences(sequences, maxlen=max_len)
print(tweets)
print(labels)

X_train, X_test, y_train, y_test = train_test_split(tweets, labels, random_state=0)
print(len(X_train), len(X_test), len(y_train), len(y_test))

model0 = Sequential()
model0.add(layers.Embedding(max_words, 15))
model0.add(layers.SimpleRNN(15))
model0.add(layers.Dense(3, activation='softmax'))
model0.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint0 = ModelCheckpoint("best_model0.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto', period=1, save_weights_only=False)
history0 = model0.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), callbacks=[checkpoint0])

model1 = Sequential()
model1.add(layers.Embedding(max_words, 20))
model1.add(layers.LSTM(15, dropout=0.5))
model1.add(layers.Dense(3, activation='softmax'))
model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint1 = ModelCheckpoint("best_model1.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto', period=1, save_weights_only=False)
history1 = model1.fit(X_train, y_train, epochs=70, validation_data=(X_test, y_test), callbacks=[checkpoint1])

model2 = Sequential()
model2.add(layers.Embedding(max_words, 40, input_length=max_len))
model2.add(layers.Bidirectional(layers.LSTM(20, dropout=0.6)))
model2.add(layers.Dense(3, activation='softmax'))
model2.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint2 = ModelCheckpoint("best_model2.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto', period=1, save_weights_only=False)
history2 = model2.fit(X_train, y_train, epochs=70, validation_data=(X_test, y_test), callbacks=[checkpoint2])

model3 = Sequential()
model3.add(layers.Embedding(max_words, 40, input_length=max_len))
model3.add(layers.Conv1D(20, 6, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=2e-3, l2=2e-3), bias_regularizer=regularizers.l2(2e-3)))
model3.add(layers.MaxPooling1D(5))
model3.add(layers.Conv1D(20, 6, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=2e-3, l2=2e-3), bias_regularizer=regularizers.l2(2e-3)))
model3.add(layers.GlobalMaxPooling1D())
model3.add(layers.Dense(3, activation='softmax'))
model3.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
checkpoint3 = ModelCheckpoint("best_model3.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto', period=1, save_weights_only=False)
history3 = model3.fit(X_train, y_train, epochs=70, validation_data=(X_test, y_test), callbacks=[checkpoint3])

best_model = keras.models.load_model("best_model2.hdf5")

test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=2)
print('Model accuracy: ', test_accuracy)

predictions = best_model.predict(X_test)

matrix = confusion_matrix(y_test.argmax(axis=1), np.around(predictions, decimals=0).argmax(axis=1))
matrix

sentiment = ['Neutral','Negative','Positive']

sequence = tokenizer.texts_to_sequences(["""
Research Statement
Zhangchunsheng
Research background
I was educated at Sichuan University, majoring in Mechanical Design, Manufacture, and Automatization from 2008 to 2012. But after my graduation, I became more and more interested in computer science and artificial intelligence, and I quit my first job as an automotive engineer at FAW(First Automobile Works) and made up my mind to find a job in computer science and artificial intelligence.

In 2016, I got my first programming job in Beijing. In 2018, I joined Bytedance Co., Ltd. and have been working for almost five years since then. I'm familiar with C/C++, GO, Python, PHP, Node.js, and many other programming languages. I can get familiar with any programming language or software framework in a week.

During Bytedance, I mainly constructed a CI/CD platform called Bits. Now, the Bits platform has fully supported the mobile app CI/CD process in ByteDance. The whole mobile app CI/CD process, from the developer's first line of code to the delivery of the app to AppStore, is completed through this platform. In constructing platform functions, I mainly develop and maintain the platform's core process and mid-term planning. I currently host our group's agile process stand-up meeting, regularly assigning tasks, checking progress, and reporting timely follow-ups and results.

I am very interested in scientific research but have little research experience. However, I think I have a characteristic no one can compare to. I have high execution ability and will stick to the goal once I set it. I believe scientific research should also need this ability.
Research interests
I don't have a profound understanding of cutting-edge computer science and artificial intelligence, but I think natural language processing, computer vision, and multimodal machine learning are challenging and essential.

Although our computer vision is relatively mature, it is still far from biological intelligence. Although computer vision can distinguish people, dogs, and cats, this differs from what humans do in the real world. The computer trains the model through a large amount of data and pictures, then learns to distinguish these. But humans are different. After a child sees a cat, he knows it is a cat and will never misjudge it. If the computer has seen a biological dog and a toy dog that looks alike, it is likely to misjudge them since their images are similar. A child will never mistake a biological dog and a toy dog.

The same is true for natural language processing. Although the current ChatGPT is very popular, it does not reason and induct like a human. My colleagues and I have run some tests on ChatGPT. We asked it to establish a regular expression from some English sentences. It is straightforward for people to recognize the pattern. The difficulty lies in generating the regular expression for humans. But ChatGPT still couldn't induct the correct rules after several rounds of questioning. ChatGPT is more like an intelligent search engine. It is just like a child who has learned a lot of knowledge. If you ask him what he has learned, he will quickly come up with the answer, but there seem to be some deficiencies in reasoning and inducting.

For multimodal machine learning, I think this is closer to reality. Because it is difficult to understand the meaning of words, sounds, or images alone, even for humans. A classic example is "Story of Stone Grotto Poet Eating Lions." All the Chinese characters in this short article are read in Chinese "shi." You don't know what he is talking about if you don't read the text. So I think what can be applied to people's daily lives should probably be multimodal machine learning.
Research plan
I think when we use a lot of data to train the model, we are already far from the mechanism of our actual brain. A child doesn't need hundreds or thousands of photos to identify a cat. I wonder if we can use a small number of samples for model training. Just imagine, if we want to make an artificial intelligence that can reason independently, it will face the same complex world as humans. We can't fill it with all possible data and pictures from the moment it is created and let it carry out model training. There are always some scenes that it has never met before. The correct approach should be to let it learn these things through a small amount of training. This is what we should do, and it is also how to make AI closer to humans.

"""])
test = pad_sequences(sequence, maxlen=max_len)
sentiment[np.around(best_model.predict(test), decimals=0).argmax(axis=1)[0]]
