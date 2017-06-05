Hi, This is Takeshi. This code is image recognition of MNIST.
I build a PCA(Principal Componet Analysis) from scrach not using built in function. So I'd like to share my code :-)

1. Content:
- To build image recognition model by using PCA and NativeBayes.

2. Objective：
- To share sample code who study Machine Learning and Deep Learning.

3. Environment：
- MacBook Air
- OSX 10.11.6
- Python 3.x
- numpy
- pandas
- sklearn

5. Summary：
- I got more than 80 % accuracy although I use original PCA !

■step1
- Download the datasets form this url (New York University LeCun Professor).
  http://yann.lecun.com/exdb/mnist/index.html
- Change the var on your environment.

```py
DATA_PATH = '/Users/takeshi/MNIST_data'
TRAIN_IMG_NAME = 'train-images.idx3-ubyte'
TRAIN_LBL_NAME = 'train-labels.idx1-ubyte'
TEST_IMG_NAME = 't10k-images.idx'
TEST_LBL_NAME = 't10k-labels.idx'
```
■step2, step3
Read the training and test dataset as numpy array.
You can check the data by using imshow.

```py
print("The shape of matrix is : ", Xtr.shape)
print("Label is : ", Ttr.shape)
plt.imshow(Xte[0].reshape(28, 28),interpolation='None', cmap=cm.gray)
show()
```
![digit7.png](https://qiita-image-store.s3.amazonaws.com/0/141816/8d03c163-3210-daa6-ac05-a36816262256.png)

■step４
This is main part of PCA. 
To create eigenvector from feature vector by using linear algebra technique.
*This is good explanation website about PCA theory ang algorithm.
 https://deeplearning4j.org/eigenvector.html

```py
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
```


■step5
To build image recognition model when we use just two eigenvectors.
(We have 784 eigenvectors.)

```py
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

# Apply traing dataset to this model
# A: the number of training set
# B: the number of dimension
A = 60000
B = 2
model.fit(P[0:A,0:B],T[0:A])
```

■step6
The result of accuracy isn't good when we calculate it by using test datasets.
44.7%

```py
from sklearn import metrics
predicted = model.predict(P[A:70001,0:B])
expected = T[A:70001,]
print ('The accuracy is : ', metrics.accuracy_score(expected, predicted)*100, '%')
```

■step７
To check Classification Report and Confusion Matrix.

```py
import matplotlib.pyplot as plt
import seaborn as sns
print ('          === Classification Report ===')
print (metrics.classification_report(expected, predicted))

cm = metrics.confusion_matrix(expected, predicted)
plt.figure(figsize=(9, 6))
sns.heatmap(cm, linewidths=.9,annot=True,fmt='g')
plt.suptitle('MNIST Confusion Matrix (GaussianNativeBayesian)')
plt.show()
```

When the eigenvector is 2, the accuracy of digit 1 is 83% and this is good result.
On the other hand, the digit 2 and 5 isn't recognized correctly.

![report.png](https://qiita-image-store.s3.amazonaws.com/0/141816/c0252eb7-3166-8f69-8624-ea1ea64ab256.png)

![cm.png](https://qiita-image-store.s3.amazonaws.com/0/141816/ce7ad026-791e-c67c-12fb-67e61c955e98.png)

Why ? 
This is because some digit of eigenvectors are overlapped.

This is good example.
There is no mistake between digit 1 and 4. On the other hand, There are lots of mistake between 4 and 9.
Here is a 3D plot of eigenvectors.
The conbination of digit 1 and 4 is separated cleary. But the conbination of digit 4 and 9 is overlapped. Hence it's impossible distinguish between digit 4 and 9.

■Digit 1 and 4

![image.png](https://qiita-image-store.s3.amazonaws.com/0/141816/54a022c1-aec0-41ba-bc40-45b3bd41b856.png)

■Digit 4 and 9

　　![image.png](https://qiita-image-store.s3.amazonaws.com/0/141816/5ef7d67a-9875-92a9-8410-207c77dd8636.png)

＊purple color is digit 4

So, we can improve the accuracy by increasing the number of enginvectors. I could get 80% accuracy from my this experiment
```py
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

# Apply traing dataset to this model
# A: the number of training set
# B: the number of dimension
A = 60000
B = 70 # <- incread this !
model.fit(P[0:A,0:B],T[0:A])
```
![cm70.png](https://qiita-image-store.s3.amazonaws.com/0/141816/8d21db36-5df8-ff0d-94a2-48c14ffb3de0.png)

■ The Difference between two feature vectors and 70 feature vecotrs.
The first picture is two feature vectors and the second picture is 70 feature vecotrs.
The contour of digit is clearer than  tow feature vectors.
* These feature vectors is re-constructed by eigenvectors.

![2.png](https://qiita-image-store.s3.amazonaws.com/0/141816/10ab8499-1aad-8a2d-50e5-521b9194b74b.png)

![70.png](https://qiita-image-store.s3.amazonaws.com/0/141816/61ddd0f7-dec2-9773-6954-0973aa74ee80.png)

```py
Xrec2=(np.dot(P[:,0:2],V[0:2,:]))+μ; #Reconstruction using 2 components
Xrec3=(np.dot(P[:,0:70],V[0:70,:]))+μ; #Reconstruction using 3 components
plt.imshow(Xrec2[1].reshape(28, 28),interpolation='None', cmap=cm.gray);
show()
plt.imshow(Xrec3[1].reshape(28, 28),interpolation='None', cmap=cm.gray);
show()
```

■Summary
- I got more than 80 % accuracy although I use original PCA !
- I attempt to build a Bayes from scrach next time in order to increase a accuracy.
