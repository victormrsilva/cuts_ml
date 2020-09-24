import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize, forest_minimize

le = preprocessing.LabelEncoder()

csv = pd.read_csv('combined_csv.csv', delimiter=';')
y = abs(1 - csv['label'])
X = csv.drop('label', axis=1)
X['cut_type'] = le.fit_transform(X['cut_type'])

# define the space of hyperparameters to search
# parameters = list()
# parameters.append(Integer(2, 50, 'log-uniform', name='max_depth'))
# parameters.append(Integer(2, 10, name='min_samples_split'))
# parameters.append(Integer(1, 30, name='min_samples_leaf'))
# parameters.append(Integer(3, len(X.columns), 'log-uniform', name='max_features'))
# parameters.append(Categorical(['gini', 'entropy'], name='criterion'))
#
# @use_named_args(parameters)
# def evaluate_model(**params):
#     #  configure the model with specific hyperparameters
#     model = DecisionTreeClassifier()
#     model.set_params(**params)
#     # define test harness
#     cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
#     # calculate 5-fold cross validation
#     res = cross_val_score(model, X, y, cv=cv, n_jobs=-1, scoring='recall')
#     # calculate the mean of the scores
#     estimate = mean(res)
#     # convert from a maximizing score to a minimizing score
#     return 1.0 - estimate
#
# # perform optimization
# result = forest_minimize(evaluate_model, parameters, verbose=True)
# # summarizing finding:
# print('Best Recall for Decision Tree: %.3f' % (1.0 - result.fun))
# print('Best Parameters for Decision Tree: %s' % result.x)
#
# print('--------------------------')

# define the space of hyperparameters to search
parameters = list()
parameters.append(Integer(100, 5000, 'log-uniform', name='n_estimators'))
parameters.append(Real(10**-5, 10**0, "log-uniform", name='learning_rate'))

@use_named_args(parameters)
def evaluate_model(**params):
    #  configure the model with specific hyperparameters
    model = AdaBoostClassifier()
    model.set_params(**params)
    # define test harness
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # calculate 5-fold cross validation
    res = cross_val_score(model, X, y, cv=cv, n_jobs=-1, scoring='recall')
    # calculate the mean of the scores
    estimate = mean(res)
    # convert from a maximizing score to a minimizing score
    return 1.0 - estimate

# perform optimization
result = gp_minimize(evaluate_model, parameters, verbose=True)
# summarizing finding:
print('Best Recall for AdaBoost: %.3f' % (1.0 - result.fun))
print('Best Parameters for AdaBoost: %s' % result.x)

# define the space of hyperparameters to search
# parameters = list()
# parameters.append(Integer(100, 5000, 'log-uniform', name='n_estimators'))
# parameters.append(Integer(50, 110, 'log-uniform', name='max_depth'))
# parameters.append(Integer(2, 10, name='min_samples_split'))
# parameters.append(Integer(1, 30, name='min_samples_leaf'))
# parameters.append(Integer(3, len(X.columns), 'log-uniform', name='max_features'))
# parameters.append(Categorical(['gini', 'entropy'], name='criterion'))
#
# @use_named_args(parameters)
# def evaluate_model(**params):
#     #  configure the model with specific hyperparameters
#     model = RandomForestClassifier()
#     model.set_params(**params)
#     # define test harness
#     cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
#     # calculate 5-fold cross validation
#     res = cross_val_score(model, X, y, cv=cv, n_jobs=-1, scoring='recall')
#     # calculate the mean of the scores
#     estimate = mean(res)
#     # convert from a maximizing score to a minimizing score
#     return 1.0 - estimate
#
# # perform optimization
# result = gp_minimize(evaluate_model, parameters, verbose=True)
# # summarizing finding:
# print('Best Recall for Random Forest: %.3f' % (1.0 - result.fun))
# print('Best Parameters for Random Forest: %s' % result.x)
#
# print('--------------------------')


