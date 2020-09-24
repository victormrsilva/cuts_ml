import pandas as pd
import numpy as np
from numpy import mean, std
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, RepeatedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import preprocessing
import matplotlib.pyplot as plt

from skopt.space import Real, Categorical, Integer

le = preprocessing.LabelEncoder()

csv = pd.read_csv('combined_csv_teste1.csv', delimiter=';')
cut_types = csv.cut_type.unique()
# y = abs(1 - csv['label'])
# X['cut_type'] = le.fit_transform(X['cut_type'])

depth = [2, 5, 10, 20, 30, 50, 75, 100]
samples_split = [2, 4, 6]
samples_leaf = [1, 2, 4, 6]
print('depth;samples_split;samples_leaf;recall;std')
for d in depth:
    for split in samples_split:
        for leaf in samples_leaf:
            for cut in cut_types:
                cond_type = csv['cut_type'] == cut
                teste = csv[cond_type]
                y = teste['label']
                X = teste.drop('label', axis=1)
                dt = DecisionTreeClassifier(max_depth=110, min_samples_split=2, min_samples_leaf=1, criterion='entropy')
                cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
                # calculate 5-fold cross validation
                res = cross_val_score(dt, X, y, cv=cv, n_jobs=-1, scoring='recall')
                # calculate the mean of the scores
                estimate = mean(res)
                stdv = std(res)
                print('{};{};{};{};{};{}'.format(d, split, leaf, cut, estimate, stdv))

# dt = DecisionTreeClassifier(max_depth=110, min_samples_split=2, min_samples_leaf=1, criterion='entropy')
# cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# # calculate 5-fold cross validation
# res = cross_val_score(dt, X, y, cv=cv, n_jobs=-1, scoring='recall')
# # calculate the mean of the scores
# estimate = mean(res)
# stdv = std(res)
# print('110;2;1;{};{}'.format(estimate, stdv))

#
# model = AdaBoostClassifier()
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# # calculate 5-fold cross validation
# res = cross_val_score(model, X, y, cv=cv, n_jobs=-1, scoring='recall')
# # calculate the mean of the scores
# estimate = mean(res)
# stdv = std(res)
# print('AdaBoostClassifier default =', estimate, '+-', stdv)
#
# model = AdaBoostClassifier(base_estimator=dt)
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# # calculate 5-fold cross validation
# res = cross_val_score(model, X, y, cv=cv, n_jobs=-1, scoring='recall')
# # calculate the mean of the scores
# estimate = mean(res)
# stdv = std(res)
# print('AdaBoostClassifier best tree =', estimate, '+-', stdv)
