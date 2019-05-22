#Libraries initiated
from sklearn.neighbors import KNeighborsClassifier  
import numpy as np
import pickle
from random import gauss
from random import seed
from pandas import Series
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


def corr2_coeff(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(0)
    B_mB = B - B.mean(0)
    
    # Sum of squares across rows
    ssA = (A_mA**2).sum(0);
    ssB = (B_mB**2).sum(0);
    
    # Finally get corr coeff
    return (((A_mA*B_mB).sum(0)).mean(0)/(np.sqrt(ssA*ssB)).mean(0))

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

[y_m,y_n] = Y.shape
[x1_m,x1_n,x1_ch] = x_EMG.shape
[x2_m,x2_n,x2_ch] = x_ACC.shape
x_EMG_cf = np.zeros((x1_ch,x1_ch,nClasses))
x_ACC_cf = np.zeros((x2_ch,x2_ch,nClasses))

for i in range(nClasses):
    r,r1 = np.where(Y == (i+1))#r1 is a redundent variable
    for j in range(x1_ch):
        for k in range(x1_ch):
            x_EMG_cf[j,k,i] = corr2_coeff(x_EMG[r,:,j],x_EMG[r,:,k])
        x_EMG_cf[j,j,i] = 0
    for j in range(x2_ch):
        for k in range(x2_ch):
            x_ACC_cf[j,k,i] = corr2_coeff(x_ACC[r,:,j],x_ACC[r,:,k])
        x_ACC_cf[j,j,i] = 0

#To removethe least related dimensions            
print(x_EMG_cf.shape)
print(x_ACC_cf.shape)
freq1 = np.zeros((nClasses,8))
I1 = np.argmax(x_EMG_cf, axis = 1)
freq2 = np.zeros((nClasses,2))
I2 = np.argmax(x_ACC_cf, axis = 1)
for i in range(nClasses):
     hist1, b1 = np.histogram(I1[:,i], bins=np.arange(0, x1_ch))
     #small to big
     freq1[i] = np.argsort(hist1)[-8:]
     hist2, b2 = np.histogram(I2[:,i], bins=np.arange(0, x2_ch))
     freq2[i] = np.argsort(hist2)[-2:]
ch1 = np.array(freq1, dtype = int)
ch2 = np.array(freq2,dtype = int)
ch = np.concatenate((ch1,ch2), axis = 1)
#print(ch)

for i in range(nClasses):#10 exercises
    r,r1 = np.where(Y == (i+1))#r1 is a redundent variable
    if(i == 0):
        X_data = np.concatenate((x_EMG[:,:,ch1[i,:]],x_ACC[:,:,ch2[i,:]]), axis = 2)[r]
        Y_data = np.array(Y[r])
        #print(X_data.shape)
        #print(Y_data.shape)
    else:
        Xdata = np.concatenate((x_EMG[:,:,ch1[i,:]],x_ACC[:,:,ch2[i,:]]), axis = 2)[r]
        X_data = np.concatenate((X_data,Xdata),axis = 0)
        Y_data = np.concatenate((Y_data,np.array(Y[r])))
        #print(X_data.shape)
        #print(Y_data.shape)

#formula for range of exercises
ex1 = 21
ex2 = 31
x_data = X_data[(ex1-1)*6*10:(ex2*10*6-1)]
y_data = Y_data[(ex1-1)*6*10:(ex2*10*6-1)] - ex1

# create white noise series
seed(1)
s = [gauss(0.0, 1.0) for i in range(2800)]
s = Series(s)
x_data = np.concatenate((x_data,(x_data+s)),axis=1)
y_data = np.concatenate((y_data,y_data),axis = 1)
print(x_data.shape)
