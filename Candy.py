#importing the basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


#reading the datasets
df = pd.read_csv('candy-data.csv')

#checking if there are any missing values
pd.isnull(df).sum()  ''' There is no missing value found'''

#checking if there is any outlier in our data set 
df['winpercent'].plot.box() ''' No outlier is found'''

df['winpercent'].plot.hist()
df.dtypes
#since there are strings present in our data set as names of the candies and rest of the data is float so we have to create dummy variables 
df = pd.get_dummies(df)

#dividing the datas into x and y 
x= df.drop('chocolate',axis=1)
y = df['chocolate']


#spliting our datasets into training and test models 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#importing logistic regression from sklearn module 
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

#fiting the x_train and y_train data into lr i.e. logistic regression 
lr.fit(x_train,y_train)

#predicting the outcome 
pred_y = lr.predict(x_test)

#checking the accuracy score and the confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(y_test,pred_y) ''' Our accuracy score is 0.9411764705882353 '''
cm = confusion_matrix(y_test,pred_y)

#ploting the histogram between the test value and the predicted value  
plt.hist(pred_y)
plt.hist(y_test)


