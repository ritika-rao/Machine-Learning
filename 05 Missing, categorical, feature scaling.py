# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1] 
#X is dataframe prepared by taking all the columns of datset except last one, -1 is last col X is predictors on x-axis
y = dataset.iloc[:, -1] 
#y is dataframe prepared by taking the last column of datset as the result, y is what we r going to predict

# Taking care of missing data
from sklearn.impute import SimpleImputer #sklearn is library

imputer = SimpleImputer(missing_values=np.nan, strategy='mean') #nan is not a number
imputer = imputer.fit(X.iloc[:, 1:]) # fit calculates mean of age & salary
# 0 is col 1, 1 is col 2, col 2 is excluded from X bcoz u dont want imputer to act on col 1
X.iloc[:, 1:] = imputer.transform(X.iloc[:, 1:]) #transform means to apply

# ft and transform can be applied in one go as well using imputer.fit_transform

# Encoding categorical data ie converting categorical to numerical
# 1st approach: Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X.iloc[:, 0] = labelencoder_X.fit_transform(X.iloc[:, 0]) # 0 is col 1
# label encoder has converted France as 1, Germany 2 and spain as 3

# 2nd approach: Make dummy variables
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train[:, 3:] = sc_X.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc_X.transform(X_test[:, 3:])

#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train.reshape(-1, 1))
