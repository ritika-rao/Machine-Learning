# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
dataset.head()

# Visualize the data (lineplot)
ax = sns.lineplot(x="Level", y="Salary", data=dataset)

# dataframes
X = dataset.iloc[:, 1]
y = dataset.iloc[:, 2]

#visualizing the data (scaltter plot)
plt.scatter(X, y, color='green')
plt.xlabel('Temperature')
plt.ylabel('Yield');

# convert the array into a single column matrix
'''

For ex an array([[ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]) reshape (-1,1) will convert it into array([[ 1],
[ 2],
[ 3],
[ 4],
[ 5],
[ 6],
[ 7],
[ 8],
[ 9],
[10],
[11],
[12]])

'''
X = np.array(X).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression().fit(X, y)
#linear_regressor = LinearRegression().fit(X_train, y_train) # in case u r doing feature scaling

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, linear_regressor.predict(X), color = 'blue')
plt.title('Salary Projection (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#SLR model evaluation
from sklearn.metrics import mean_squared_error, r2_score
mean_squared_error(y, linear_regressor.predict(X))
r2_score(y, linear_regressor.predict(X))

# Predicting a new result with Linear Regression
linear_regressor.predict(np.array(6.5).reshape(-1, 1))
#linear_regressor.predict(X_test) #if u r doing feature scaling

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polynomial_regressor = PolynomialFeatures(degree = 4) #try 2,3 and 4
X_polynomial = polynomial_regressor.fit_transform(X)
linear_regressor_2 = LinearRegression()
linear_regressor_2.fit(X_polynomial, y)

# Visualising the Polynomial Regression results (piecewise linear)
plt.scatter(X, y, color = 'red')
plt.plot(X, linear_regressor_2.predict(polynomial_regressor.fit_transform(X)), color = 'blue')
plt.title('Salary Projection (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, linear_regressor_2.predict(polynomial_regressor.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Polynomial Regression
linear_regressor_2.predict(polynomial_regressor.fit_transform(np.array(6.5).reshape(-1, 1)))