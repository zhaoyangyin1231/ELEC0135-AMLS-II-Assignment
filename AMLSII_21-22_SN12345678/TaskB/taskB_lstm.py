# subtask B in both English and Arabic

# import
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras.layers import Bidirectional
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import BatchNormalization
from keras.utils import np_utils
import gensim
import re
import numpy as np
import os

df = pd.read_csv('DataCleanB_Eng.csv')  # English Data
# df = pd.read_csv('DataCleanB_Arabic.csv')  # Arabi Data

# split data
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
print("TRAIN size:", len(df_train))
print("TEST size:", len(df_test))


# wotd2vec
def word_to_vector(df):
    doc_text = [_text.split() for _text in df.Text]
    doc_topic = [_text.split() for _text in df.Topic]
    docs = doc_text + doc_topic
    # docs = [_text.split() for _text in df.Text] 
    w2v_model = gensim.models.word2vec.Word2Vec(size=300, window=7, min_count=10, workers=8)
    w2v_model.build_vocab(docs)
    words = w2v_model.wv.vocab.keys()
    vocab_size = len(words)
    w2v_model.train(docs, total_examples=len(docs), epochs=8)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df.Text)
    vocab_size = len(tokenizer.word_index) + 1

    return tokenizer, vocab_size, w2v_model


# prepare train test data
def data_prepare(tokenizer):
    x_train = pad_sequences(tokenizer.texts_to_sequences(df_train.tidy), maxlen=300)
    x_test = pad_sequences(tokenizer.texts_to_sequences(df_test.tidy), maxlen=300)
    labels = df_train.Label.unique().tolist()
    # print(x_train)
    # print(x_test)
    encoder = LabelEncoder()
    encoder.fit(df_train.Label.tolist())

    y_train = encoder.transform(df_train.Label.tolist())
    y_test = encoder.transform(df_test.Label.tolist())

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    # print(y_train.shape)
    # print(x_train.shape)

    return x_train, x_test, y_train, y_test


def embedding_layer(vocab_size, w2v_model):
    # embedding layer
    embedding_matrix = np.zeros((vocab_size, 300))
    # print(embedding_matrix.shape)
    for word, i in tokenizer.word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]
    # print(embedding_matrix.shape)

    embedding_layer = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=300, trainable=False)
    # embedding_layer = Embedding(vocab_size, 300, input_length=300)

    return embedding_layer


def build_lstm_model(embedding_layer):
    model = Sequential()
    model.add(embedding_layer)
    # model.add(embedding_layer)
    model.add(Dropout(0.5))
    model.add(LSTM(1024, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.summary()

    return model


def train_model(model, x_train, x_test, y_train, y_test):
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])
    callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
                 EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]

    y_train = np_utils.to_categorical(y_train, num_classes=3)
    y_test = np_utils.to_categorical(y_test, num_classes=3)
    print(y_train.shape)
    print(x_train.shape)

    y_train = np.array(y_train)
    X_train = np.array(x_train)
    y_test = np.array(y_test)
    X_test = np.array(x_test)
    print(y_train.shape)
    print(x_train.shape)

    history = model.fit(x_train, y_train,
                        batch_size=128,
                        epochs=10,
                        validation_split=0.1,
                        verbose=1,
                        callbacks=callbacks)