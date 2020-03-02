# Copyright (C) 2020 IDIR Lab - UT Arlington
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License v3 as published by
#     the Free Software Foundation.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Contact Information:
#     See: https://idir.uta.edu/cli.html
#
#     Chengkai Li
#     Box 19015
#     Arlington, TX 76019
#

import os
import math
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

from sklearn import preprocessing

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Define parameters for text vectorization
max_len = 500
top_words = 5000
max_words = 10000
path_to_data = '../data/two_class'
glove_dir = '../data/glove'
embedding_dim = 300
embedding_file_name = 'glove.txt'
dataset_loc = path_to_data + 'kfold_25ncs.json'
data = pd.read_json(dataset_loc, encoding="utf-8")

# Read in labels and values (texts) for training and testing data
labels = data.label
texts = data.text

# Vectorize data
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=max_len)
labels = np.asarray(labels)

# Shuffle data and labels
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

# Convert labels to categorical values
labels_categorical = to_categorical(labels)
X_train = data
Y_train = labels_categorical

# Parse the GloVe word-embedding and normalize embedding matrix
# create dictionary to map word -> embedding vector
# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
embeddings_index = {}
f = open(os.path.join(glove_dir, embedding_file_name))
i = 0
for line in f:
    values = line.split()
    word = values[0]  # get word
    coefs = np.asarray(values[1:], dtype='float32')  # get embedding for word
    embeddings_index[word] = coefs
f.close()
print("Found %s word vectors." % len(embeddings_index))

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

embedding_matrix = preprocessing.scale(embedding_matrix)


# Define the model
def create_model(max_words, embedding_dim, max_len, embedding_matrix):
    hidden_dim = 300
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=max_len))
    model.add(Bidirectional(LSTM(hidden_dim)))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    # Load embeddings
    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False

    # Compile
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])

    return model


# Define functions for computing performance metrics
def compute_average_precision(labels, scores, cutoff=None):
    # https://github.com/apepa/clef2019-factchecking-task1/blob/master/scorer/task1.py#L52
    combined = sorted([(scores[i], labels[i]) for i in range(len(scores))], reverse=True)
    combined = combined if cutoff is None else combined[:(cutoff if cutoff < len(combined) else len(combined))]
    labels = [x[1] for x in combined]
    precisions = []
    num_correct = 0
    num_positive = sum(labels)
    for i, x in enumerate(combined):
        if cutoff is not None and i >= cutoff:
            break
        if x[1] == 1:
            num_correct += 1
            precisions.append(num_correct / (i + 1))
    if precisions:
        avg_prec = sum(precisions) / num_positive
    else:
        avg_prec = 0.0
    return avg_prec


def compute_dcg_term(i, labels, ver=1):
    # Difference between version 0 and 1: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Discounted_Cumulative_Gain
    return labels[i - 1] / math.log2(i + 1) if ver == 0 else ((1 << labels[i - 1]) - 1) / math.log2(i + 1)


def compute_ndcg(labels, scores, cutoff=None):
    # Precondition: for each index i, scores[i] corresponds with labels[i]
    ver = 0
    combined = sorted([(scores[i], labels[i]) for i in range(len(scores))], reverse=True)
    combined = combined if cutoff is None else combined[:(cutoff if cutoff < len(combined) else len(combined))]
    labels = [x[1] for x in combined]
    dcg = sum([compute_dcg_term(i, labels, ver=ver) for i in range(1, len(labels) + 1, 1)])
    ideal_labels = sorted(labels, reverse=True)
    idcg = sum([compute_dcg_term(i, ideal_labels, ver=ver) for i in range(1, len(labels) + 1, 1)])
    return dcg / idcg


# K-Folds model training and evaluation
n_folds = 4
shuffle = True
random_state = 1
predicted_y_list = []
true_y_list = []
cfs_probabilities = []

# Train BiLSTM
for train_index, test_index in StratifiedKFold(
        n_splits=n_folds, shuffle=shuffle, random_state=random_state).split(X_train, labels):
    x_train, x_test = X_train[train_index], X_train[test_index]
    y_train, y_test = Y_train[train_index], Y_train[test_index]

    # instantiate model
    print("x_train dimensions: ", x_train.shape)
    print("y_train dimensions: ", y_train.shape)
    k_fold_model = create_model(max_words, embedding_dim, max_len, embedding_matrix)

    # train and evaluate model for ['loss', 'accuracy'] metrics
    print("Training ....")
    history = k_fold_model.fit(x_train, y_train, epochs=15)

    # print model classification report
    y_hat = k_fold_model.predict(x_test, verbose=0)
    cfs_probs = y_hat[:, 1]
    y_hat_classes = tf.argmax(y_hat, axis=1).numpy()
    y_test_classes = tf.argmax(y_test, axis=1).numpy()

    print(classification_report(y_test_classes, y_hat_classes, ))
    print("Average precision: ", compute_average_precision(y_test_classes, y_hat_classes))
    print("ndcg: ", compute_ndcg(y_test_classes, cfs_probs))

    # store predicted and true values (from test)
    # for larger classification matrix
    predicted_y_list.extend(y_hat_classes)
    true_y_list.extend(y_test_classes)
    cfs_probabilities.extend(cfs_probs)

print("Classification report for all models")
print(classification_report(true_y_list, predicted_y_list, digits=4))

precisions = compute_average_precision(true_y_list, predicted_y_list)
print(precisions)

ndcg = compute_ndcg(true_y_list, cfs_probabilities)
print(ndcg)

# Training model on full dataset and saving
full_model = create_model(max_words, embedding_dim, max_len, embedding_matrix)
history = full_model.fit(X_train, Y_train, epochs=14)
full_model.save(os.path.join("./saved_models/", 'Full_BiLSTM.h5'))