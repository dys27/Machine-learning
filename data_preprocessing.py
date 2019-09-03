#Basic template for all models - importing Libraries, dataset, training/testing, feature scaling for some
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing data-set
dataSet = pd.read_csv("Data.csv")
X = dataSet.iloc[:,:-1].values
Y = dataSet.iloc[:,-1].values

#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(np.nan,'mean',0)
imputer = imputer.fit(X[:, 1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#Splitting the dataset into training data and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
