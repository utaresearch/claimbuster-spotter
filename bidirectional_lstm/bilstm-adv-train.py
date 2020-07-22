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

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import os
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
tf.compat.v1.enable_eager_execution()
path_to_data = '../data/csv/'

#Define parameters for text vectorization
max_len = 500
top_words = 5000
max_words = 10000

# Obtain and process the data
dataset_loc = path_to_data + 'cfs_ncs_2_5xNCS.csv'
data = pd.read_csv(dataset_loc)
print("Preview training data\n")
print(data.head())

# Read in labels and values (texts) for training and testing data
labels = data.label
texts = data.text

# Inspect labels and values
print("Inspecting labels and values...")
print(texts[42])
print(labels[42])
print("Inspecting labels and values:DONE")

# Vectorize data
print("Vectorizing data...")
tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print("Found %s unique tokens" % len(word_index))
data = pad_sequences(sequences, maxlen=max_len)
print("Shape of training data tensor: ", data.shape)
labels = np.asarray(labels)
print("Shape of training label tensor: ", labels.shape)
""" shuffle data and labels"""
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
print("Labels shape: ", labels.shape)
print("Vectorizing data: DONE")

# Convert labels to categorical values
print("Converting labels to categorical values...")
labels_categorical = to_categorical(labels)
X_train = data
Y_train = labels_categorical
print("Converting labels to categorical values: DONE")

# Parsing the GloVe word-embedding file
print("Parsing the GloVe word-embedding file...")
glove_dir = '../../../../glove'

embeddings_index = {}                                       
f = open(os.path.join(glove_dir, 'glove.6B.300d.txt'))
i = 0
for line in f:
    values = line.split()
    word = values[0]                                         
    coefs = np.asarray(values[1:], dtype='float32')          
    embeddings_index[word] = coefs
f.close()
print("Found %s word vectors." % len(embeddings_index))

embedding_dim = 300

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

print("Shape of embedding matrix: ")
print(embedding_matrix.shape)
print("Parsing the GloVe word-embedding file: DONE")


#Normalizing the word-embedding
print("Normalizing the word embedding...")
from sklearn import preprocessing
embedding_matrix = preprocessing.scale(embedding_matrix)
print("Normalizing the word embedding: DONE")


# Define model builder
print("Defining model builder...")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import StratifiedKFold

def make_embedding(max_words, embedding_dim, max_len, hidden_dim, embedding_matrix):
    model = Sequential()
    model.add(Embedding(max_words,embedding_dim, input_length=max_len))
    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
    #print("Embedding Layer: \n")
    #print(model.summary())
    return model

def create_model(hidden_dim):
    model = Sequential()
    model.add(Bidirectional(LSTM(hidden_dim, input_shape=(500, 300))))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    return model

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

import math

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

print("Defining model builder: DONE")

# Define Loss and Gradient functions
print("Defining loss, gradient functions, and optimizer")
loss_object = tf.keras.losses.BinaryCrossentropy()

def loss(model, x, y):
    y_hat = model(x)
    return loss_object(y_true=y, y_pred=y_hat), y_hat

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value, y_hat = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables), y_hat

def adv_grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        loss_value = loss(model, inputs, targets)
    gradient = tape.gradient(loss_value, inputs)
    return tf.sign(gradient)

optimizer = tf.keras.optimizers.RMSprop()
print("Defining loss, gradient functions, and optimizer: DONE")


# Fold datasets and train the model
print("Folding the datasets and training the model...")
n_folds = 4
shuffle = True
random_state = 1
predicted_y_list = []
true_y_list = []
cfs_probabilities = []

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

embedding_model = make_embedding(max_words, embedding_dim, max_len,300, embedding_matrix)

for train_index, test_index in StratifiedKFold(
    n_splits=n_folds, shuffle=shuffle, random_state=random_state).split(X_train, labels):
    
    x_train, x_test = X_train[train_index], X_train[test_index]
    y_train, y_test = Y_train[train_index], Y_train[test_index]
    print("x_train dimensions: ", x_train.shape)
    print("y_train dimensions: ", y_train.shape) 
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    model = create_model(300)
    
    #### Train and evaluate model ####
    epsilon = 2.0
    training_loss_results = []
    training_accuracy_results = []
    
    epochs = 15

    for epoch in range(epochs):
        """Use these in training_loss_results, training_accuracy_results"""
        epoch_mean_loss = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # Training loop using batches of size == 1
        for step, (x_training, y_training) in enumerate(train_dataset):

            # Record the forward-pass using GradientTape for
            # autodifferentiation
            with tf.GradientTape() as tape:

                x = embedding_model(x_training)

                regular_loss, _, y_hat = grad(model, x, y_training) 

                adversarial_gradient = adv_grad(model, x, y_training)
                adversarial_x = x + epsilon*adversarial_gradient

                adversarial_loss, _, _ = grad(model, adversarial_x, y_training)

                total_loss = regular_loss + adversarial_loss
                #print(total_loss)

            # Retrieve the gradients w.r.t.loss.
            grads = tape.gradient(total_loss, model.trainable_weights)

            # Run one step of gradient descent.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            """ 
            epoch_mean_loss and epoch_accuracy update the loss and
            accuracy metrics with each new batch
            """

            #print("Total loss: ", total_loss)
            #print("y_train: ", y_train)
            #print("y_hat: ", y_hat)

            epoch_mean_loss(total_loss)
            epoch_accuracy(tf.argmax(y_training, axis=1),y_hat)


            # Log every 500 batches (i.e. 500 records because batch_size = 1)
            if step % 10 == 0:
                print('Training loss (for one batch) at step %s: %s' % (step, float(total_loss)))
                print('Seen so far: %s samples' % ((step + 1))) #<= this should just be (step+1)


        training_loss_results.append(epoch_mean_loss.result())
        training_accuracy_results.append(epoch_accuracy.result())

    
    testing_array = x_test
    word_emb_testing = embedding_model(testing_array)
    test_results = model(word_emb_testing)
    
    cfs_probs = test_results[:,1]
    
    y_hat_classes = tf.argmax(test_results, axis=1).numpy()
    print(y_hat_classes.shape)
    
    y_true_classes = tf.argmax(y_test, axis=1).numpy()
    print(y_true_classes.shape)
    
    cfs_probabilities.extend(cfs_probs)
    predicted_y_list.extend(y_hat_classes)
    true_y_list.extend(y_true_classes)
    
    
    print(classification_report(y_true_classes, y_hat_classes, digits=4))
print("Folding the datasets and training the model: DONE")

# Compute average precision and ndcg
print("Computing average precision and ndcg...")
precisions = compute_average_precision(true_y_list, predicted_y_list)
ndcg = compute_ndcg(true_y_list, predicted_y_list)

print("Average precision: %s" % str(precisions))
print("Average ndcg: %s" % str(ndcg))
print("Computing average precision and ndcg: DONE")
