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
    for j in range(x2_ch):
        for k in range(x2_ch):
            x_ACC_cf[j,k,i] = corr2_coeff(x_ACC[r,:,j],x_ACC[r,:,k])
            
print(x_EMG_cf.shape)
print(x_ACC_cf.shape)
freq1 = np.zeros(nClasses)
I1 = np.argmin(x_EMG_cf, axis = 1)
freq2 = np.zeros(nClasses)
I2 = np.argmin(x_ACC_cf, axis = 1)
for i in range(nClasses):
     hist1, b1 = np.histogram(I1[:,i], bins=np.arange(0, x1_ch))
     freq1[i] = np.argmax(hist1)
     hist2, b2 = np.histogram(I2[:,i], bins=np.arange(0, x2_ch))
     freq2[i] = np.argmax(hist2)
hist1, b1 = np.histogram(freq1, bins=np.arange(0, x1_ch))
ch1 = np.argsort(hist1)[-8:]
hist2, b2 = np.histogram(freq2, bins=np.arange(0, x2_ch))
ch2 = np.argsort(hist2)[-8:]
ch = np.concatenate((ch1,ch2))
print(ch)

for i in range(10):#10 exercises
    r,r1 = np.where(Y == (i+1))#r1 is a redundent variable
    if(i == 0):
        tbx = np.concatenate((x_EMG[:,:,ch1],x_ACC[:,:,ch2]), axis = 2)[r]
        train_x = tbx[0:40]
        test_x = tbx[40:60]
        tby = np.array(Y[r])
        train_y = tby[0:40]
        test_y = tby[40:60]
        print(train_x.shape)
        print(train_y.shape)
        print(test_x.shape)
        print(test_y.shape)
    else:
        tbx = np.concatenate((x_EMG[:,:,ch1],x_ACC[:,:,ch2]), axis = 2)[r]
        train_x = np.concatenate((train_x,tbx[0:40]),axis=0)
        test_x = np.concatenate((test_x,tbx[40:60]),axis=0)
        tby = np.array(Y[r])
        train_y = np.concatenate((train_y,tby[0:40]))
        test_y = np.concatenate((test_y,tby[40:60]))
        print(train_x.shape)
        print(train_y.shape)
        print(test_x.shape)
        print(test_y.shape)
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
with open('input_train.pkl','wb') as f:
    pickle.dump(train_x, f)
with open('input_test.pkl','wb') as f:
    pickle.dump(test_x, f)
with open('outut_test.pkl','wb') as f:
    pickle.dump(test_y, f)
with open('output_train.pkl','wb') as f:
    pickle.dump(train_y, f)
np.save("input_train",train_x)
np.save("output_train",train_y)
np.save("input_test",test_x)
np.save("output_test",test_y)

