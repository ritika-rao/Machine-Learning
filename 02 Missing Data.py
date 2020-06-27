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
from sklearn.impute import SimpleImputer  #sklearn is library

imputer = SimpleImputer(missing_values=np.nan, strategy='mean') #nan is not a number
imputer = imputer.fit(X.iloc[:, 1:]) # fit calculates mean of age & salary
# 0 is col 1, 1 is col 2, col 2 is excluded from X bcoz u dont want imputer to act on col 1
X.iloc[:, 1:] = imputer.transform(X.iloc[:, 1:]) #transform means to apply

# ft and transform can be applied in one go as well using imputer.fit_transform