
#step1. Define the function: reading data 

import os, struct
import matplotlib as plt
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros as np
import pandas as pd
from pandas import Series,DataFrame

DATA_PATH = '/Users/takeshi/MNIST_data'
TRAIN_IMG_NAME = 'train-images.idx3-ubyte'
TRAIN_LBL_NAME = 'train-labels.idx1-ubyte'
TEST_IMG_NAME = 't10k-images.idx'
TEST_LBL_NAME = 't10k-labels.idx'

# Set the path to save data
"""
Adapted from: http://cvxopt.org/applications/svm/index.html?highlight=mnist
"""
def load_mnist(dataset="training", digits=range(10), path=DATA_PATH):
#Set the filename
    if dataset == "training":
        fname_img = os.path.join(path, TRAIN_IMG_NAME)
        fname_lbl = os.path.join(path, TRAIN_LBL_NAME)
    elif dataset == "testing":
        fname_img = os.path.join(path, TEST_IMG_NAME)
        fname_lbl = os.path.join(path, TEST_LBL_NAME)
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels


#step2. Read Digit data from training dataset
from pylab import *
from numpy import *
import scipy.sparse as sparse
import scipy.linalg as linalg

# Set the digit to read
images, labels = load_mnist('training', digits=[0,1,2,3,4,5,6,7,8,9])

# converting from NX28X28 array into NX784 array
flatimages = list()
for i in images:
    flatimages.append(i.ravel())
Xtr= np.asarray(flatimages)

# Create the Label like 0, 1, 2, 3...9
flatlabels = list()
for ii in labels:
    flatlabels.append(ii.ravel())
Ttr= np.asarray(flatlabels)

#print (X.shape);
#μ=np.mean(X,axis=0); 
#np.savetxt("μ.csv", μ, fmt="%.05f",delimiter=",")

#Print the digit
print("The shape of matrix is : ", Xtr.shape)
print("Label is : ", Ttr.shape)
plt.imshow(Xtr[0].reshape(28, 28),interpolation='None', cmap=cm.gray)
show()


#step3. Read Digit data from test dataset
images, labels = load_mnist('testing', digits=[0,1,2,3,4,5,6,7,8,9])

# converting from NX28X28 array into NX784 array
flatimages = list()
for i in images:
    flatimages.append(i.ravel())
Xte= np.asarray(flatimages)

# Create the Label like 0, 1, 2, 3...9
flatlabels = list()
for ii in labels:
    flatlabels.append(ii.ravel())
Tte= np.asarray(flatlabels)

#Print the digit
print("The shape of matrix is : ", Xte.shape)
print("Label is : ", Tte.shape)
plt.imshow(Xte[0].reshape(28, 28),interpolation='None', cmap=cm.gray)
show()


#step4. Calculate Principal Component from the dataset(training and test dataset)
X = np.vstack((Xtr,Xte))
T = np.vstack((Ttr,Tte))
print (X.shape)
print (T.shape)

import numpy as np;
import numpy.linalg as LA;
μ=np.mean(X,axis=0);#print(μ);
Z=X-μ;#print(Z);
C=np.cov(Z,rowvar=False);#print(C);
[λ,V]=LA.eigh(C);#print(λ,'\n\n',V);
row=V[0,:];col=V[:,0];
np.dot(C,row)/(λ[0]*row) ;
np.dot(C,col)/(λ[0]*col);
λ=np.flipud(λ);V=np.flipud(V.T);
row=V[0,:]; 
np.dot(C,row)/(λ[0]*row);
P=np.dot(Z,V.T);#print(P);
R=np.dot(P,V);#print(R-Z);
Xrec=R+μ;#print(Xrec-X);


#step5. Build GaussianNB model
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

# Apply traing dataset to this model
# A: the number of training set
# B: the number of  dimension
A = 60000
B = 70
model.fit(P[0:A,0:B],T[0:A])


#step6.  Apply test dataset to this model and calculate this model's accuracy
from sklearn import metrics
predicted = model.predict(P[A:70001,0:B])
expected = T[A:70001,]
print ('The accuracy is : ', metrics.accuracy_score(expected, predicted)*100, '%')


#step7. print confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
print ('          === Classification Report ===')
print (metrics.classification_report(expected, predicted))

cm = metrics.confusion_matrix(expected, predicted)
plt.figure(figsize=(9, 6))
sns.heatmap(cm, linewidths=.9,annot=True,fmt='g')
plt.suptitle('MNIST Confusion Matrix (GaussianNativeBayesian)')
#plt.title('70 Dimension: 88.08%')
plt.show()

