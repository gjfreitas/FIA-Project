# Import Libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
# load dataset
dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, [0,1, 2, 3,4,5,6,7,8,9,10,11,12]].values
y = dataset.iloc[:, -1].values

# define model
model = LogisticRegression(max_iter=1000, random_state=0)
# define evaluation
from sklearn.model_selection import KFold
kf = KFold(n_splits=3)
# define search space
param = dict()
param['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
param['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
param['C'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]

# There will be some warning messages but we can ignore as they are just warning and not errors messages

# 80:20 data split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# define search
search = GridSearchCV(model, param, scoring='neg_mean_absolute_error', cv=kf)
# execute search
result = search.fit(X_train, y_train)


# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)
