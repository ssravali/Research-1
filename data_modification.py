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
import pandas as pd

#Data loading
reader = csv.reader(open("/home/ssadhu2/Research1/Data/dataY.csv", "r"), delimiter=",")
f1 = list(reader)
y = (np.array(f1).astype("int") - 20)

f = "/home/ssadhu2/Research1/Data/dataXch1.csv"
reader = csv.reader(open(f, "r"), delimiter=",")
f1 = list(reader)
x1 = np.array(f1).astype("float")

f = "/home/ssadhu2/Research1/Data/dataXch2.csv"
reader = csv.reader(open(f, "r"), delimiter=",")
f1 = list(reader)
x2  = np.array(f1).astype("float")

f = "/home/ssadhu2/Research1/Data/dataXch3.csv"
reader = csv.reader(open(f, "r"), delimiter=",")
f1 = list(reader)
x3 = np.array(f1).astype("float")

f = "/home/ssadhu2/Research1/Data/dataXch4.csv"
reader = csv.reader(open(f, "r"), delimiter=",")
f1 = list(reader)
x4 = np.array(f1).astype("float")

f = "/home/ssadhu2/Research1/Data/dataXch5.csv"
reader = csv.reader(open(f, "r"), delimiter=",")
f1 = list(reader)
x5 = np.array(f1).astype("float")

f = "/home/ssadhu2/Research1/Data/dataXch6.csv"
reader = csv.reader(open(f, "r"), delimiter=",")
f1 = list(reader)
x6 = np.array(f1).astype("float")

f = "/home/ssadhu2/Research1/Data/dataXch7.csv"
reader = csv.reader(open(f, "r"), delimiter=",")
f1 = list(reader)
x7 = np.array(f1).astype("float")

f = "/home/ssadhu2/Research1/Data/dataXch8.csv"
reader = csv.reader(open(f, "r"), delimiter=",")
f1 = list(reader)
x8 = np.array(f1).astype("float")

f = "/home/ssadhu2/Research1/Data/dataXch8.csv"
reader = csv.reader(open(f, "r"), delimiter=",")
f1 = list(reader)
x9 = np.array(f1).astype("float")

f = "/home/ssadhu2/Research1/Data/dataXch8.csv"
reader = csv.reader(open(f, "r"), delimiter=",")
f1 = list(reader)
x10 = np.array(f1).astype("float")

f = "/home/ssadhu2/Research1/Data/dataXch8.csv"
reader = csv.reader(open(f, "r"), delimiter=",")
f1 = list(reader)
x11 = np.array(f1).astype("float")

f = "/home/ssadhu2/Research1/Data/dataXch8.csv"
reader = csv.reader(open(f, "r"), delimiter=",")
f1 = list(reader)
x12 = np.array(f1).astype("float")

f = "/home/ssadhu2/Research1/Data/dataXch8.csv"
reader = csv.reader(open(f, "r"), delimiter=",")
f1 = list(reader)
x13 = np.array(f1).astype("float")

f = "/home/ssadhu2/Research1/Data/dataXch8.csv"
reader = csv.reader(open(f, "r"), delimiter=",")
f1 = list(reader)
x14 = np.array(f1).astype("float")

f = "/home/ssadhu2/Research1/Data/dataXch8.csv"
reader = csv.reader(open(f, "r"), delimiter=",")
f1 = list(reader)
x15 = np.array(f1).astype("float")

f = "/home/ssadhu2/Research1/Data/dataXch8.csv"
reader = csv.reader(open(f, "r"), delimiter=",")
f1 = list(reader)
x16 = np.array(f1).astype("float")

x = np.array([x1,x2,x3,x4,x5,x6,x7,x8])
print ('The shape of output is: ',y.shape)
print ('The shape of input is: ',x.shape)
classes = np.unique(y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

[y_m,y_n] = y.shape
[x_ch,x_m,x_n] = x.shape

ratio = (4/6)

X_train =np.zeros((x_ch,int(ratio*x_m),x_n))
X_test = np.zeros((x_ch,int((1-ratio)*x_m),x_n))
y_train = np.zeros((int(ratio*y_m),y_n))
y_test = np.zeros((int((1-ratio)*y_m),y_n))
Y = np.zeros((int((1-ratio)*y_m),8))

for ch in range(8):#iterating through the chs
    j = 0 #count if it is training or testing
    cl = 1 #checking the class it belongs to\
    train_i = 0#training matrix
    test_i = 0
    for i in range(m):#iterating through the elements
        if(j<4):
            j = j + 1 
            X_train[ch,train_i,:] = x[ch,i,:]
            y_train[train_i,0] = y[i,0]
            train_i = train_i + 1
        elif(j<6):
            j = j + 1
            X_test[ch,test_i,:] = x[ch,i,:]
            y_test[test_i,0] = y[i,0]
            test_i = test_i + 1
        else:
            cl = cl + 1
        if(cl == 32):
            cl = 0
            
X_test = np.array(X_test)
X_train = np.array(X_train)
y_test = np.array(y_test)
y_train = np.array(y_train)
print ('The shape of train input is: ',X_train.shape)
print ('The shape of train output is: ',y_train.shape)
print ('The shape of test output is: ',y_test.shape)
print ('The shape of test input is: ',X_test.shape)
