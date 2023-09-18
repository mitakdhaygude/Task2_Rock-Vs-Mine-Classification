
#Importing theDependencies

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#data collection and data processing
#loading the dataset to a pandas dataframe
sonar_data = pd.read_csv("C:/Users/omc/Desktop/Task/sonar data.csv", header = None)
sonar_data.head()
print(sonar_data.head())

# number of rows and columns
sonar_data.shape
print(sonar_data.shape)
sonar_data.describe()
print(sonar_data.describe())  #describe --> measures of the data
sonar_data[60].value_counts()
print(sonar_data[60].value_counts())

# M--> Mine
# R --> Rock

sonar_data.groupby(60).mean()
print(sonar_data.groupby(60).mean())

# separating data and labels 
X = sonar_data.drop(columns=60,axis=1)
Y = sonar_data[60]
print(X)
print(Y)

#Training and test data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size= 0.1,stratify= Y, random_state=1)
print(X.shape, X_train.shape , X_test.shape)

print(X_train)
print(Y_train)

# Model Training  --> Logistic Regression
model = LogisticRegression()
# training the logistic regression modul with training data
model.fit(X_train, Y_train)
print(model.fit(X_train, Y_train))

# Model Evalution
# Accuarcy on training data
X_train_prediction = model.predict(X_train)
train_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuarcy on training data : ",train_data_accuracy)

# testing data on Accuarcy
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuarcy on testing data : ",test_data_accuracy)

# MAKING A PREDICTIVE DATA
input_data = (0.0228,0.0106,0.0130,0.0842,0.1117,0.1506,0.1776,0.0997,0.1428,0.2227,0.2621,0.3109,0.2859,0.3316,0.3755,0.4499,0.4765,0.6254,0.7304,0.8702,0.9349,0.9614,0.9126,0.9443,1.0000,0.9455,0.8815,0.7520,0.7068,0.5986,0.3857,0.2510,0.2162,0.0968,0.1323,0.1344,0.2250,0.3244,0.3939,0.3806,0.3258,0.3654,0.2983,0.1779,0.1535,0.1199,0.0959,0.0765,0.0649,0.0313,0.0185,0.0098,0.0178,0.0077,0.0074,0.0095,0.0055,0.0045,0.0063,0.0039)
# changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]=='R'):
    print("The object is a Rock")
else:
    print("The object is a Mine")
