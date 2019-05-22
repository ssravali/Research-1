#Libraries initiated
from sklearn.neighbors import KNeighborsClassifier  
import numpy as np
import pickle
import matplotlib.pyplot as plt  
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import decomposition
import csv
import logging
from time import time
import scipy.io as sio

#Data loading
for i in range(1,11):
    if i == 1:
        s = 'EMG' + str(i) + '.mat'
        mat_contents = sio.loadmat(s)
        EMG = np.array(mat_contents['EMG'])
        inc = 2800 - int(EMG.shape[1])
        x_EMG = np.pad(EMG,((0,0),(0,inc),(0,0)),'constant')
        s = 'ACC' + str(i) + '.mat'
        mat_contents = sio.loadmat(s)
        ACC = np.array(mat_contents['ACC'])
        x_ACC = np.pad(ACC,((0,0),(0,inc),(0,0)),'constant')
        s = 'Y' + str(i) + '.mat'
        mat_contents = sio.loadmat(s)
        Y = np.array(mat_contents['Y'])
    else:
        s = 'EMG' + str(i) + '.mat'
        mat_contents = sio.loadmat(s)
        EMG = np.array(mat_contents['EMG'])
        inc = 2800 - int(EMG.shape[1])
        EMG = np.pad(EMG,((0,0),(0,inc),(0,0)),'constant')
        x_EMG = np.concatenate((x_EMG,EMG),axis = 0)
        s = 'ACC' + str(i) + '.mat'
        mat_contents = sio.loadmat(s)
        ACC = np.array(mat_contents['ACC'])
        ACC = np.pad(ACC,((0,0),(0,inc),(0,0)),'constant')
        x_ACC = np.concatenate((x_ACC,ACC),axis = 0)
        s = 'Y' + str(i) + '.mat'
        mat_contents = sio.loadmat(s)
        Y = np.concatenate((Y,np.array(mat_contents['Y'])),axis = 0)
        
print ('The shape of output is: ',Y.shape)
print ('The shape of input EMG is: ',x_EMG.shape)
print ('The shape of input ACC is: ',x_ACC.shape)
classes = np.unique(Y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)
