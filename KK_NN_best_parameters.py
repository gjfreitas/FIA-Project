import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

# Import the data set
df = pd.read_csv('dataset.csv')
X = df.iloc[:, [0,1, 2, 3,4,5,6,7,8,9,10,11,12]].values
y = df.iloc[:, -1].values

#Create KNN Object.
knn = KNeighborsClassifier()
#Split data into training and testing.
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
#Training the model.
knn.fit(x_train, y_train)
#Predict test data set.
y_pred = knn.predict(x_test)
#Checking performance our model with classification report.
print(classification_report(y_test, y_pred))
#Checking performance our model with ROC Score.
roc_auc_score(y_test, y_pred)

# Hyperparameter Tunning
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
#List Hyperparameters that we want to tune.
leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))
p=[1,2]
#Convert to dictionary
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
#Create new KNN object
knn_2 = KNeighborsClassifier()
#Use GridSearch
clf = GridSearchCV(knn_2, hyperparameters,  scoring='neg_mean_absolute_error',cv=10)
#Fit the model
best_model = clf.fit(x_train,y_train)
#Print The value of best Hyperparameters
print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
# print how our model looks after hyper-parameter tuning
print(clf.best_estimator_)

