import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns

from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression


# load the dataset from sklearn datasets
dib = datasets.load_diabetes()

# show what we have got as the data set
# print(dib.values())
# print(dib.feature_names)

X = dib.data
Y = dib.target

print(X.shape, Y.shape)

#split the data in to test and training
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

#dimensions
print(X_train.shape, Y_train.shape)

#implement model
model = LinearRegression()
model.fit(X_train, Y_train)

predictions = model.predict(X_test)
print(predictions)

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, predictions))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, predictions))
     
sns.scatterplot(Y_test, predictions, alpha=0.5)
plt.show()