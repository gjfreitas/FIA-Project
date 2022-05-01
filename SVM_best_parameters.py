# Importing the libraries
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

# Importing the dataset
dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, [0,1, 2, 3,4,5,6,7,8,9,10,11,12]].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
# 80:20 data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)



from sklearn.model_selection import GridSearchCV

# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf', 'poly', 'sigmoid']}
#'kernel': ['rbf', 'poly', 'sigmoid']
grid = GridSearchCV(SVC(),param_grid, scoring='neg_mean_absolute_error')

# fitting the model for grid search
grid.fit(X_train, y_train)

# print best parameter after tuning
print(grid.best_params_)

# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

grid_predictions = grid.predict(X_test)
# print classification report
print(classification_report(y_test, grid_predictions))
