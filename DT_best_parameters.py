# Importing the Libraries
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Import the data set
df = pd.read_csv('dataset.csv')
X = df.iloc[:, [0,1, 2, 3,4,5,6,7,8,9,10,11,12]].values
y = df.iloc[:, -1].values
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'rest_ecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal', 'condition']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.tree import DecisionTreeClassifier
model2 = DecisionTreeClassifier(random_state=42)
model2.fit(X_train, y_train)

#Visualizing a Decision Tree
from sklearn.tree import plot_tree, export_text
plt.figure(figsize =(80,50))

plot_tree(model2, feature_names=columns, max_depth=1, filled=True)
#plt.show()

#Hyper Paramter Tunning
parameters = {"splitter": ['best', 'random'], "max_depth": [1, 3, 5, 7, 9, 11, 1],
                "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "min_weight_fraction_leaf": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                "max_features": ["auto", "log2", "sqrt", None],
                "max_leaf_nodes": [None, 10, 20, 30, 40, 50, 60, 70, 80, 90]}

# 80:20 data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = DecisionTreeClassifier()
grid = GridSearchCV(model,parameters, scoring='neg_mean_absolute_error')
# fitting the model for grid search
grid.fit(X_train, y_train)

# print best parameter after tuning
print(grid.best_params_)

# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

#print('Best Score: %s' % result.best_score_)
#print('Best Hyperparameters: %s' % result.best_params_)
