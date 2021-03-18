```python
import pandas as pd
```


```python
from sklearn import preprocessing #preprocessing is a func used to convert text to numerical form
```


```python
from sklearn.model_selection import train_test_split
```


```python
from sklearn.naive_bayes import GaussianNB
```


```python
from sklearn.metrics import accuracy_score #hoe many records are accurately classified
```


```python
from sklearn.metrics import confusion_matrix
```


```python
dataset=pd.read_csv('SVMtrain.csv')
```


```python
dataset.columns
```




    Index(['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
           'Fare', 'Embarked'],
          dtype='object')




```python
le=preprocessing.LabelEncoder() 
#text to numerical 
```


```python
le.fit(dataset["Sex"])

```




    LabelEncoder()




```python
print(le.classes_)
```

    ['Male' 'female']
    


```python
dataset['Sex']=le.transform(dataset["Sex"])
```


```python
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
dataset1=dataset.drop(["PassengerId"],axis=1)
```


```python
dataset1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
#survived = depedent var
#remaining idv
```


```python
y=dataset1["Survived"]
#y= pd.factorize(dataset1['Survived'])[0].reshape(-1,1) 
#y
```


```python
X=dataset1.drop(["Survived"],axis=1)
```


```python
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>0</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=True)#splitting data randomly
```


```python
X_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>114</th>
      <td>3</td>
      <td>0</td>
      <td>21.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>3</td>
    </tr>
    <tr>
      <th>872</th>
      <td>2</td>
      <td>1</td>
      <td>28.0</td>
      <td>1</td>
      <td>0</td>
      <td>24.0000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>76</th>
      <td>3</td>
      <td>0</td>
      <td>60.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>3</td>
    </tr>
    <tr>
      <th>874</th>
      <td>3</td>
      <td>0</td>
      <td>20.0</td>
      <td>0</td>
      <td>0</td>
      <td>9.8458</td>
      <td>3</td>
    </tr>
    <tr>
      <th>681</th>
      <td>3</td>
      <td>0</td>
      <td>20.0</td>
      <td>0</td>
      <td>0</td>
      <td>9.2250</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_train.head()
```




    114    0
    872    1
    76     0
    874    0
    681    0
    Name: Survived, dtype: int64




```python
X_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>386</th>
      <td>2</td>
      <td>1</td>
      <td>36.0</td>
      <td>0</td>
      <td>0</td>
      <td>13.0000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>258</th>
      <td>2</td>
      <td>1</td>
      <td>50.0</td>
      <td>0</td>
      <td>1</td>
      <td>26.0000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>770</th>
      <td>3</td>
      <td>0</td>
      <td>48.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.8542</td>
      <td>3</td>
    </tr>
    <tr>
      <th>207</th>
      <td>3</td>
      <td>1</td>
      <td>16.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.7500</td>
      <td>2</td>
    </tr>
    <tr>
      <th>682</th>
      <td>3</td>
      <td>0</td>
      <td>14.0</td>
      <td>5</td>
      <td>2</td>
      <td>46.9000</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_test.head()
```




    386    1
    258    1
    770    0
    207    1
    682    0
    Name: Survived, dtype: int64




```python
from sklearn.naive_bayes import *
```


```python
clf=BernoulliNB()  #BernoulliNB() is the func name for naivr bayes
```


```python
y_pred=clf.fit(X_train,y_train).predict(X_test)#based on training dataset we are prediciting test data set
```


```python
accuracy_score(y_test,y_pred,normalize=True)  #based on training dataset , the test dataset is around 82 %,82% the data is classified
```




    0.8202247191011236




```python
confusion_matrix(y_test,y_pred)  #tells how d data is correctly classified and not classfied
```




    array([[144,  22],
           [ 26,  75]], dtype=int64)




```python
clf_nb = GaussianNB().fit(X_train,y_train)
```


```python
print("Survived")
print('Accuracy of Gaussian_NB classifier on training set:{:.2f}'.format(clf_nb.score(X_train,y_train)))
print('Accuracy of Gaussian_NB classifier on test set:{:.2f}'.format(clf_nb.score(X_test,y_test)))

```

    Survived
    Accuracy of Gaussian_NB classifier on training set:0.77
    Accuracy of Gaussian_NB classifier on test set:0.81
    


```python
#dv =pclass
y=dataset1["Pclass"]
```


```python
X=dataset1.drop(["Pclass"],axis=1)

```


```python
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
```


```python
clf=BernoulliNB()
```


```python
y_pred=clf.fit(X_train,y_train).predict(X_test)
```


```python
accuracy_score(y_test,y_pred,normalize=True)
```




    0.5617977528089888




```python
confusion_matrix(y_test,y_pred) #since pclass has 3 values,1,2,3
```




    array([[ 27,   8,  35],
           [ 15,   4,  30],
           [ 24,   5, 119]], dtype=int64)




```python
clf_nb = GaussianNB().fit(X_train,y_train)
```


```python
print("pclass")
print('Accuracy of Gaussian_NB classifier on training set:{:.2f}'.format(clf_nb.score(X_train,y_train)))
print('Accuracy of Gaussian_NB classifier on test set:{:.2f}'.format(clf_nb.score(X_test,y_test)))

```

    pclass
    Accuracy of Gaussian_NB classifier on training set:0.69
    Accuracy of Gaussian_NB classifier on test set:0.69
    


```python
#Gaussiannb for irid data
```


```python
data=pd.read_csv("iris.data")
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>5.1</th>
      <th>3.5</th>
      <th>1.4</th>
      <th>0.2</th>
      <th>Iris-setosa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.4</td>
      <td>3.9</td>
      <td>1.7</td>
      <td>0.4</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>




```python
col=['sepal length','sepal width',' petal length','petal width','class']
```


```python
data.columns=col
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length</th>
      <th>sepal width</th>
      <th>petal length</th>
      <th>petal width</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.4</td>
      <td>3.9</td>
      <td>1.7</td>
      <td>0.4</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.datasets import load_iris 
iris = load_iris() 
  
# store the feature matrix (X) and response vector (y) 
X = iris.data 
y = iris.target 

  
# splitting X and y into training and testing sets 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1) 
  
# training the model on training set 
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB() 
gnb.fit(X_train, y_train) 
  
# making predictions on the testing set 
y_pred = gnb.predict(X_test)   
# comparing actual response values (y_test) with predicted response values (y_pred) 
```


```python
from sklearn import metrics 
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)

```

    Gaussian Naive Bayes model accuracy(in %): 95.0
    


```python

```


```python

```
