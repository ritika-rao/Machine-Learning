# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('SAT GPA.csv')
X = dataset.iloc[:, :-1] #Take all the columns except last one
y = dataset.iloc[:, -1] #Take the last column as the result

# Splitting the dataset into the Training set (Xtrain & Y train) and Test set (X Test & Y Test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) # ft returns 4 subsets

# no missing values, no categorical data in dataset

'''# Feature Scaling, library will do automatically for you, no code required
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train[:, 3:] = sc_X.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc_X.transform(X_test[:, 3:])

#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train.reshape(-1, 1))'''

from sklearn.linear_model import LinearRegression

regressor = LinearRegression() # object regressor is created
# Fitting Linear Regression to the Training set
regressor.fit(X_train, y_train) 

# Testing: Predicting
y_pred = regressor.predict(X_test) 

# Visualising the Training set results
import seaborn as sns
sns.set(color_codes=True)

dataframe_training = pd.DataFrame() 
dataframe_training['SAT'] = X_train['SAT'] 
dataframe_training['GPA'] = y_train
ax = sns.regplot(x="SAT", y="GPA", data= dataframe_training)

# Visualising the Test set results
dataframe_test = pd.DataFrame()
dataframe_test['SAT'] = X_test['SAT']
dataframe_test['GPA'] = y_test
ax = sns.regplot(x="SAT", y="GPA", data= dataframe_training)

print('y-intercept c: \n', regressor.coef_)

from sklearn.metrics import mean_squared_error, r2_score

# The mean squared error
print("Mean squared error: {}".format(mean_squared_error(y_test, y_pred)))
print("R square: {}".format(r2_score(y_test, y_pred)))

import statsmodels.api as sm

# Add a constant. Essentially, we are adding a new column (equal in lenght to x), which consists only of 1s
x = sm.add_constant(X)
# Fit the model using OLS model
results = sm.OLS(endog = y, exog=x).fit() # dependent variable y, independent variable x
results.summary() # Statsmodel will Print the summary of the regression