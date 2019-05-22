#Libraries initiated
import numpy as np
import pickle
import matplotlib.pyplot as plt  
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import csv
import logging
from time import time
import pandas as pd
from keras.layers import Flatten
from keras import optimizers
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
print("Imported libraries")
###########################################################################################################
#Input
train_x = np.load("input_train.npy")
train_y = np.load("output_train.npy")
test_x = np.load("input_test.npy")
test_y = np.load("output_test.npy")

# LSTMs are sensitive to the scale of the input data, specifically when the
# sigmoid (default) or tanh activation functions are used. It can be a good
# practice to rescale the data to the range of 0-to-1, also called normalizing.

# normalize the dataset
x_min = train_x.min(axis=(0, 1), keepdims=True)
x_max = train_x.max(axis=(0, 1), keepdims=True)
train_x = (train_x - x_min)/(x_max - x_min)

x_min = test_x.min(axis=(0, 1), keepdims=True)
x_max = test_x.max(axis=(0, 1), keepdims=True)
test_x = (test_x - x_min)/(x_max - x_min)

# binary encode
train_y = OneHotEncoder(sparse=False,categories='auto').fit_transform(train_y)
test_y = OneHotEncoder(sparse=False,categories='auto').fit_transform(test_y)
print("Train_y one hot:", train_y.shape)
print("Train_y one hot:", test_y.shape)
print("Input processing done.")
###########################################################################################################
#RNN
n_batch = train_x.shape[0]
n_epoch = 1
n_neurons = 28
model = Sequential()

# Recurrent layer
model.add(LSTM(n_neurons, batch_input_shape=(n_batch, train_x.shape[1], train_x.shape[2]), return_sequences=False, 
               dropout=0.1, recurrent_dropout=0.1))

# Fully connected layer
model.add(Dense(n_neurons, activation='relu'))

# Dropout for regularization
model.add(Dropout(0.5))

# Output layer
model.add(Dense(train_y.shape[1], activation='softmax'))

# Compile the model
model.compile(
    optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Create callbacks
callbacks = [EarlyStopping(monitor='val_loss', patience=5),
             ModelCheckpoint('../models/model.h5', save_best_only = True, 
                             save_weights_only=False)]

history = model.fit(train_x, train_y, 
                    batch_size= n_batch, epochs=n_epoch,
                    callbacks=callbacks, shuffle = True)
print("Model Created")
###########################################################################################################
# make predictions
trainPredict = model.predict(train_x)
testPredict = model.predict(test_x)
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(train_y[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(test_y[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
print("Model pridected")

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
###########################################################################################################
