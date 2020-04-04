#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:43:06 2017

@author: amine

Improved on Sat Apr 04 2020 by Tran Hai Anh

"""

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from math import log

# importing the dataset
dataset = pd.read_csv('dataset/kddcup.data_10_percent_corrected')

# change Multi-class to binary-class
dataset['normal.'] = dataset['normal.'].replace(['back.', 'buffer_overflow.', 'ftp_write.', 'guess_passwd.', 'imap.', 'ipsweep.', 'land.', 'loadmodule.', 'multihop.', 'neptune.', 'nmap.', 'perl.', 'phf.', 'pod.', 'portsweep.', 'rootkit.', 'satan.', 'smurf.', 'spy.', 'teardrop.', 'warezclient.', 'warezmaster.'], 'attack')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 41].values

# encoding categorical data of 3 fields: protocol_type, service, and flag
labelencoder_x_1 = LabelEncoder()
labelencoder_x_2 = LabelEncoder()
labelencoder_x_3 = LabelEncoder()
x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])
x[:, 3] = labelencoder_x_3.fit_transform(x[:, 3])

# onehotencoder_1 = OneHotEncoder(categorical_features=[1])
# x = onehotencoder_1.fit_transform(x).toarray()
# onehotencoder_2 = OneHotEncoder(categorical_features=[4])
# x = onehotencoder_2.fit_transform(x).toarray()
# onehotencoder_3 = OneHotEncoder(categorical_features=[70])
# x = onehotencoder_3.fit_transform(x).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# splitting the dataset into the training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# feature scaling
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


# Initialising the ANN
def create_classifier():
    classifier = Sequential()

    # Adding the input layer and the first hidden layer
    classifier.add(Dense(output_dim=60, init='uniform', activation='relu', input_dim=41))

    # Adding a second hidden layer
    classifier.add(Dense(output_dim=60, init = 'uniform', activation='relu'))

    # Adding a third hidden layer
    classifier.add(Dense(output_dim=60, init='uniform', activation='relu'))

    # Adding the output layer
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

    # Compiling the ANN
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return classifier


# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
# checkpoint_path = "checkpoints/cp-0004.ckpt"

# Create a callback that saves the model's weights every 5 epochs
cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    period=2)

# Create a new model called classifier
classifier = create_classifier()

# # Loads the weights
# classifier.load_weights(checkpoint_path)

# Save the weights using the `checkpoint_path` format
classifier.save_weights(checkpoint_path.format(epoch=0))

# Fitting the ANN to the Training set
classifier.fit(x_train, y_train, batch_size=10, callbacks=[cp_callback], nb_epoch=4)

# Evaluate the model
loss, acc = classifier.evaluate(x_test,  y_test, verbose=2)
print("Accuracy: {:5.2f}%".format(100*acc))

# # Predicting the Test set results
# y_pred = classifier.predict(x_test)
# y_pred = (y_pred > 0.5)
#
# # Making the Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
#
# # the performance of the classification model
# print("the Accuracy is: " + str((cm[0, 0]+cm[1, 1])/(cm[0, 0]+cm[0, 1]+cm[1, 0]+cm[1, 1])))
# recall = cm[1, 1]/(cm[0, 1]+cm[1, 1])
# print("Recall is : " + str(recall))
# print("False Positive rate: " + str(cm[1, 0]/(cm[0, 0]+cm[1, 0])))
# precision = cm[1, 1]/(cm[1, 0]+cm[1, 1])
# print("Precision is: " + str(precision))
# print("F-measure is: " + str(2*((precision*recall)/(precision+recall))))
# print("Entropy is: " + str(-precision*log(precision)))




