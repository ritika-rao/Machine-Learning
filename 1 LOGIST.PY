# Logistic Regression

'''
Binary (two-class classification) logistic regression requires dependent variable to be binary (0,1)
 and ordinal/oredered (multi-class) logistic regression requires dependent variable to be ordinal (e.g Too little, About Right, Too much)

Assumptions in Logistic regression 
1 No (or little) multicollinarity
2 No autocorrelaton between errors
3 Linear relationship between indelendent variables and Log Odds i.e. log(P/(1-P))
this does not mean that the dependent and independent variables to be related linearly
4 No outliers in continous indelendent variable
5 MLE neeeds atleast 10 cases per independent variable
i.e if you have 5 independent variables and the expected probability of your 
 least frequent outcome is .10, then you would need a minimum sample 
 size of 500 (10*5 / .10).
'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]] # col 1 which is categorical variable has been dropped
y = dataset.iloc[:, -1]

# ASSUMPTIONS

# to check for presence of an outlier
sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.boxplot(data= X).set_title("Outlier Box Plot")

#Linearity Assumption
sns.regplot(x= 'Age', y= 'Purchased', data= dataset, logistic= True).set_title("Age Log Odds Linear Plot")
sns.regplot(x= 'EstimatedSalary', y= 'Purchased', data= dataset, logistic= True).set_title("Age Log Odds Linear Plot")
# Box-Tidwell test
linearity_check_df = pd.concat([pd.DataFrame(X),y],axis=1) 
linearity_check_df.columns = 'Male Age Salary Purchased'.split()
linearity_check_df["Age"] = linearity_check_df["Age"].apply(np.log)
sns.regplot(x= 'Age', y= 'Purchased', data= linearity_check_df, logistic= True).set_title("Age Log Odds Linear Plot")
sns.regplot(x= 'Salary', y= 'Purchased', data= linearity_check_df, logistic= True).set_title("Salary Log Odds Linear Plot")
sns.regplot(x= 'Male', y= 'Purchased', data= linearity_check_df, logistic= True).set_title("Gender Log Odds Linear Plot")

# check for multicollinearity
multicolinearity_check = dataset.corr()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling (to be done after splitting)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# K-Fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
model_accuracy = accuracies.mean()
model_standard_deviation = accuracies.std()

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred) # n=a+b+c+d=80 which is 20% of 400

# Genarate Reports

import statsmodels.api as sm
logit_model=sm.Logit(y_train,X_train)
result=logit_model.fit()
print(result.summary())
print(result.summary2())

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#ROC Curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
area_under_curve = roc_auc_score(y_test, classifier.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % area_under_curve)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

# 2 feature data
# Visualising the Training set results
def visualise_logestic_regression_results(X_set, y_set, xlabel, ylabel):
    from matplotlib.colors import ListedColormap
    
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.5, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('Logistic Regression')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
    
# Visualising the Test set results
visualise_logestic_regression_results(X_test, y_test, 'Age', 'Estimated Salary')