import pandas as pd
import numpy as np
from numpy import mean, std
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn import preprocessing
import matplotlib.pyplot as plt
import graphviz

from skopt.space import Real, Categorical, Integer

# le = preprocessing.LabelEncoder()

csv = pd.read_csv('combined_csv_teste1.csv', delimiter=';')
y = csv['label']
X = csv.drop('label', axis=1)

feature_names = X.columns.values
class_names = ['corte certo', 'corte errado']
# input(feature_names)
import os
os.environ["PATH"] += os.pathsep + 'D:/Graphviz/bin/'

cond_type = csv['cut_type'] == 4
teste = csv[cond_type]
y = teste['label']
X = teste.drop('label', axis=1)
dt = DecisionTreeClassifier(max_depth=10, min_samples_split=2, min_samples_leaf=1, criterion='entropy')
dt.fit(X, y)
predicted = dt.predict(X)
print(np.sum(y))
print(np.sum(predicted))
print(np.sum(abs(y - predicted)))
tn, fp, fn, tp = confusion_matrix(y, predicted).ravel()
print('TN = {}\tFP = {}\tFN = {}\tTP = {}'.format(tn, fp, fn, tp))

recall = recall_score(y, dt.predict(X))
print('recall ', recall)
if np.sum(abs(y - predicted)) < 20:
    print("Feature ranking:")
    feature_importances = dt.feature_importances_
    indices = np.argsort(feature_importances)[::-1]
    for f in range(X.shape[1]):
        if feature_importances[indices[f]] > 0.000001:
            print("%d. feature %d (%s) = %f" % (f + 1, indices[f], feature_names[indices[f]], feature_importances[indices[f]]))
    print('-----------')
    # for f in range(X.shape[1]):
    #     print("feature %d (%s) = %f" % (f, feature_names[f], feature_importances[f]))
    # input('verificar')
    # Plot the impurity-based feature importances of the forest

    fig = plt.figure(figsize=(25,20))
    _ = plot_tree(dt,
                       feature_names=feature_names,
                       class_names=class_names,
                       filled=True)
    fig.savefig("dt_{}_{}_{}_{}.png".format(10, 2, 1, 4))

    dot_data = export_graphviz(dt, out_file=None,
                                    feature_names=feature_names,
                                    class_names=class_names,
                                    filled=True,
                                    label='all',
                                    proportion=False,
                                    precision=8)

    # Draw graph
    graph = graphviz.Source(dot_data, format="png")
    graph.render("dt_{}_{}_{}_{}".format(10, 2, 1, 4))
    plt.close(fig)