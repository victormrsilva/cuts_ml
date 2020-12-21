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

csv = pd.read_csv('results_cbc_nocuts.csv', delimiter=';')
pd.set_option('display.max_rows', 50000)
pd.set_option('display.max_columns', 50000)
pd.set_option('display.width', 50000)
# input(feature_names)
import os
os.environ["PATH"] += os.pathsep + 'D:/Graphviz/bin/'

# cond_type = csv['cut_type'] == 2
# teste = csv[cond_type]
teste = csv
# teste = teste[(teste['label'].eq(0)) | (teste['label'].eq(1) & teste['diff'].abs().ge(1e-4))]
# input(teste[['label','diff']].query('label == 1'))
# input(teste.columns.tolist())
# y = teste['label']
# X = teste.drop(['label', 'instance'], axis=1)
# X = teste.drop(['label', 'instance', 'relax_iteration', 'nonzeros', 'pct_nonzeros', 'unsatisfied_var', 'pct_unsatisfied_var', 'cut_type', 'n_variables_coef_nonzero', 'xvar_zero', 'coeff_leq_0.5', 'coeff_leq_1', 'coeff_geq_1', 'minor_coef', 'abs_minor_coef', 'major_coef', 'abs_major_coef', 'abs_ratio_minor_major_coef', 'ratio_abs_minor_major_coef', 'abs_rhs', 'rhs', 'diff', 'away', 'lub', 'eps_coeff', 'eps_coeff_lub', 'lhs', 'abs_lhs', 'abs_ratio_lhs_rhs', 'abs_ratio_min_max_coeff_rhs', 'abs_ratio_min_coeff_rhs'], axis=1)
# X = teste.drop(['label', 'instance', 'cols', 'rows', 'colsPerRow', 'equalities', 'nzEqualities', 'percEqualities', 'percNzEqualities', 'inequalities', 'nzInequalities', 'nz', 'density', 'bin', 'genInt', 'integer', 'continuous', 'percInteger', 'percBin', 'nUnbounded1', 'percUnbounded1', 'nUnbounded2', 'percUnbounded2', 'rPartitioning', 'rPercPartitioning', 'rPacking', 'rPercPacking', 'rPartPacking', 'rPercRowsPartPacking', 'rCovering', 'rPercCovering', 'rCardinality', 'rPercCardinality', 'rKnapsack', 'rPercKnapsack', 'rIntegerKnapsack', 'rPercIntegerKnapsack', 'rInvKnapsack', 'rPercInvKnapsack', 'rSingleton', 'rPercSingleton', 'rAggre', 'rPercAggre', 'rPrec', 'rPercPrec', 'rVarBnd', 'rPercVarBnd', 'rBinPacking', 'rPercBinPacking', 'rMixedBin', 'rPercMixedBin', 'rGenInt', 'rPercGenInt', 'rFlowBin', 'rPercFlowBin', 'rFlowMx', 'rPercFlowMx', 'rNzRowsPartitioning', 'rNzPercRowsPartitioning', 'rNzRowsPacking', 'rNzPercRowsPacking', 'rNzrowsPartPacking', 'rNzpercRowsPartPacking', 'rNzRowsCovering', 'rNzPercRowsCovering', 'rNzRowsCardinality', 'rNzPercRowsCardinality', 'rNzRowsKnapsack', 'rNzPercRowsKnapsack', 'rNzRowsIntegerKnapsack', 'rNzPercRowsIntegerKnapsack', 'rNzRowsInvKnapsack', 'rNzPercRowsInvKnapsack', 'rNzRowsSingleton', 'rNzPercRowsSingleton', 'rNzRowsAggr', 'rNzPercRowsAggr', 'rNzRowsPrec', 'rNzPercRowsPrec', 'rNzRowsVarBnd', 'rNzPercRowsVarBnd', 'rNzRowsBinPacking', 'rNzPercRowsBinPacking', 'rNzRowsMixedBin', 'rNzPercRowsMixedBin', 'rNzRowsGenInt', 'rNzPercRowsGenInt', 'rNzRowsFlowBin', 'rNzPercRowsFlowBin', 'rNzRowsFlowMx', 'rNzPercRowsFlowMx', 'aMin', 'aMax', 'aAvg', 'aStdDev', 'aRatioLSA', 'aAllInt', 'aPercInt', 'aDiffVal', 'anShortInts', 'apercShortInts', 'objMin', 'objMax', 'objAvg', 'objStdDev', 'objRatioLSA', 'objAllInt', 'objPercInt', 'objDiffVal', 'objnShortInts', 'objpercShortInts', 'rhsMin', 'rhsMax', 'rhsAvg', 'rhsStdDev', 'rhsRatioLSA', 'rhsAllInt', 'rhsPercInt', 'rhsDiffVal', 'rhsnShortInts', 'rhspercShortInts', 'rowNzMin', 'rowNzMax', 'rowNzAvg', 'rowNzStdDev', 'colNzMin', 'colNzMax', 'colNzAvg', 'colNzStdDev', 'rowsLess4Nz', 'rowsLess8Nz', 'rowsLess16Nz', 'rowsLess32Nz', 'rowsLess64Nz', 'rowsLess128Nz', 'rowsLess256Nz', 'rowsLess512Nz', 'rowsLess1024Nz', 'percRowsLess4Nz', 'percRowsLess8Nz', 'percRowsLess16Nz', 'percRowsLess32Nz', 'percRowsLess64Nz', 'percRowsLess128Nz', 'percRowsLess256Nz', 'percRowsLess512Nz', 'percRowsLess1024Nz', 'rowsLeast4Nz', 'rowsLeast8Nz', 'rowsLeast16Nz', 'rowsLeast32Nz', 'rowsLeast64Nz', 'rowsLeast128Nz', 'rowsLeast256Nz', 'rowsLeast512Nz', 'rowsLeast1024Nz', 'rowsLeast2048Nz', 'rowsLeast4096Nz', 'percRowsLeast4Nz', 'percRowsLeast8Nz', 'percRowsLeast16Nz', 'percRowsLeast32Nz', 'percRowsLeast64Nz', 'percRowsLeast128Nz', 'percRowsLeast256Nz', 'percRowsLeast512Nz', 'percRowsLeast1024Nz', 'percRowsLeast2048Nz', 'percRowsLeast4096Nz', 'colsLess4Nz', 'colsLess8Nz', 'colsLess16Nz', 'colsLess32Nz', 'colsLess64Nz', 'colsLess128Nz', 'colsLess256Nz', 'colsLess512Nz', 'colsLess1024Nz', 'percColsLess4Nz', 'percColsLess8Nz', 'percColsLess16Nz', 'percColsLess32Nz', 'percColsLess64Nz', 'percColsLess128Nz', 'percColsLess256Nz', 'percColsLess512Nz', 'percColsLess1024Nz', 'colsLeast4Nz', 'colsLeast8Nz', 'colsLeast16Nz', 'colsLeast32Nz', 'colsLeast64Nz', 'colsLeast128Nz', 'colsLeast256Nz', 'colsLeast512Nz', 'colsLeast1024Nz', 'colsLeast2048Nz', 'colsLeast4096Nz', 'percColsLeast4Nz', 'percColsLeast8Nz', 'perccolsLeast16Nz', 'perccolsLeast32Nz', 'perccolsLeast64Nz', 'perccolsLeast128Nz', 'perccolsLeast256Nz', 'perccolsLeast512Nz', 'perccolsLeast1024Nz', 'perccolsLeast2048Nz', 'perccolsLeast4096Nz'], axis=1)

rskf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
y = teste['label'].copy()
X = teste.drop(['label', 'instance', ], axis=1).copy()

dropcolumns = list()
dropcolumns.append(['cols', 'rows', 'colsPerRow', 'equalities', 'nzEqualities', 'percEqualities', 'percNzEqualities', 'inequalities', 'nzInequalities', 'nz', 'density', 'bin', 'genInt', 'integer', 'continuous', 'percInteger', 'percBin', 'nUnbounded1', 'percUnbounded1', 'nUnbounded2', 'percUnbounded2', 'rPartitioning', 'rPercPartitioning', 'rPacking', 'rPercPacking', 'rPartPacking', 'rPercRowsPartPacking', 'rCovering', 'rPercCovering', 'rCardinality', 'rPercCardinality', 'rKnapsack', 'rPercKnapsack', 'rIntegerKnapsack', 'rPercIntegerKnapsack', 'rInvKnapsack', 'rPercInvKnapsack', 'rSingleton', 'rPercSingleton', 'rAggre', 'rPercAggre', 'rPrec', 'rPercPrec', 'rVarBnd', 'rPercVarBnd', 'rBinPacking', 'rPercBinPacking', 'rMixedBin', 'rPercMixedBin', 'rGenInt', 'rPercGenInt', 'rFlowBin', 'rPercFlowBin', 'rFlowMx', 'rPercFlowMx', 'rNzRowsPartitioning', 'rNzPercRowsPartitioning', 'rNzRowsPacking', 'rNzPercRowsPacking', 'rNzrowsPartPacking', 'rNzpercRowsPartPacking', 'rNzRowsCovering', 'rNzPercRowsCovering', 'rNzRowsCardinality', 'rNzPercRowsCardinality', 'rNzRowsKnapsack', 'rNzPercRowsKnapsack', 'rNzRowsIntegerKnapsack', 'rNzPercRowsIntegerKnapsack', 'rNzRowsInvKnapsack', 'rNzPercRowsInvKnapsack', 'rNzRowsSingleton', 'rNzPercRowsSingleton', 'rNzRowsAggr', 'rNzPercRowsAggr', 'rNzRowsPrec', 'rNzPercRowsPrec', 'rNzRowsVarBnd', 'rNzPercRowsVarBnd', 'rNzRowsBinPacking', 'rNzPercRowsBinPacking', 'rNzRowsMixedBin', 'rNzPercRowsMixedBin', 'rNzRowsGenInt', 'rNzPercRowsGenInt', 'rNzRowsFlowBin', 'rNzPercRowsFlowBin', 'rNzRowsFlowMx', 'rNzPercRowsFlowMx', 'aMin', 'aMax', 'aAvg', 'aStdDev', 'aRatioLSA', 'aAllInt', 'aPercInt', 'aDiffVal', 'anShortInts', 'apercShortInts', 'objMin', 'objMax', 'objAvg', 'objStdDev', 'objRatioLSA', 'objAllInt', 'objPercInt', 'objDiffVal', 'objnShortInts', 'objpercShortInts', 'rhsMin', 'rhsMax', 'rhsAvg', 'rhsStdDev', 'rhsRatioLSA', 'rhsAllInt', 'rhsPercInt', 'rhsDiffVal', 'rhsnShortInts', 'rhspercShortInts', 'rowNzMin', 'rowNzMax', 'rowNzAvg', 'rowNzStdDev', 'colNzMin', 'colNzMax', 'colNzAvg', 'colNzStdDev', 'rowsLess4Nz', 'rowsLess8Nz', 'rowsLess16Nz', 'rowsLess32Nz', 'rowsLess64Nz', 'rowsLess128Nz', 'rowsLess256Nz', 'rowsLess512Nz', 'rowsLess1024Nz', 'percRowsLess4Nz', 'percRowsLess8Nz', 'percRowsLess16Nz', 'percRowsLess32Nz', 'percRowsLess64Nz', 'percRowsLess128Nz', 'percRowsLess256Nz', 'percRowsLess512Nz', 'percRowsLess1024Nz', 'rowsLeast4Nz', 'rowsLeast8Nz', 'rowsLeast16Nz', 'rowsLeast32Nz', 'rowsLeast64Nz', 'rowsLeast128Nz', 'rowsLeast256Nz', 'rowsLeast512Nz', 'rowsLeast1024Nz', 'rowsLeast2048Nz', 'rowsLeast4096Nz', 'percRowsLeast4Nz', 'percRowsLeast8Nz', 'percRowsLeast16Nz', 'percRowsLeast32Nz', 'percRowsLeast64Nz', 'percRowsLeast128Nz', 'percRowsLeast256Nz', 'percRowsLeast512Nz', 'percRowsLeast1024Nz', 'percRowsLeast2048Nz', 'percRowsLeast4096Nz', 'colsLess4Nz', 'colsLess8Nz', 'colsLess16Nz', 'colsLess32Nz', 'colsLess64Nz', 'colsLess128Nz', 'colsLess256Nz', 'colsLess512Nz', 'colsLess1024Nz', 'percColsLess4Nz', 'percColsLess8Nz', 'percColsLess16Nz', 'percColsLess32Nz', 'percColsLess64Nz', 'percColsLess128Nz', 'percColsLess256Nz', 'percColsLess512Nz', 'percColsLess1024Nz', 'colsLeast4Nz', 'colsLeast8Nz', 'colsLeast16Nz', 'colsLeast32Nz', 'colsLeast64Nz', 'colsLeast128Nz', 'colsLeast256Nz', 'colsLeast512Nz', 'colsLeast1024Nz', 'colsLeast2048Nz', 'colsLeast4096Nz', 'percColsLeast4Nz', 'percColsLeast8Nz', 'perccolsLeast16Nz', 'perccolsLeast32Nz', 'perccolsLeast64Nz', 'perccolsLeast128Nz', 'perccolsLeast256Nz', 'perccolsLeast512Nz', 'perccolsLeast1024Nz', 'perccolsLeast2048Nz', 'perccolsLeast4096Nz'])
dropcolumns.append([])


for i in range(len(dropcolumns)):
    tree=0
    print('*******************')
    for train_index, test_index in rskf.split(X, y):
        tree = tree + 1
        print('---------------------')
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        X_train = X_train.drop(dropcolumns[i], axis=1)
        X_test = X_test.drop(dropcolumns[i], axis=1)
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        dt = DecisionTreeClassifier(max_depth=20, min_samples_split=2, min_samples_leaf=1, criterion='entropy')
        dt.fit(X_train, y_train)

        feature_names = X_train.columns.values
        class_names = ['corte certo', 'corte errado']

        predicted = dt.predict(X_test)
        print(np.sum(y_test))
        print('previstos como errados: ', np.sum(predicted))
        print('previstos como certos: ', np.sum(1-predicted))
        print(np.sum(abs(y_test - predicted)))
        tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
        print('TN = {}\tFP = {}\tFN = {}\tTP = {}'.format(tn, fp, fn, tp))

        recall = recall_score(y_test, predicted)
        print('recall ', recall)

        x_leaves = dt.apply(X_test)
        unique, counts = np.unique(x_leaves, return_counts=True)

        positions = {x: list() for x in unique}
        positions_index = {x: list() for x in unique}
        for ind in range(len(x_leaves)):
            positions[x_leaves[ind]].append(test_index[ind])
            positions_index[x_leaves[ind]].append(ind)
        threshold = dt.tree_.threshold
        feature = dt.tree_.feature

        node_indicator = dt.decision_path(X_test)

        for index, a in positions.items():
            if len(a) > 4:
                print('leaf ', index, 'total', len(a))
                print('predicted:')
                print(np.asarray((np.unique(predicted[positions_index[index]], return_counts=True))).T)
                print(teste[['instance','label']].loc[a].groupby(['instance', 'label']).size())
                # input()

        # print(node_indicator.indptr[17650])
        # for index,a in positions_index.items():
        #     if len(a) > 0:
        #         sample_id = a[0]
        #         node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
        #                                             node_indicator.indptr[sample_id + 1]]
        #         print('Rules used to predict sample %s: ' % sample_id)
        #         for node_id in node_index:
        #             if x_leaves[sample_id] == node_id:
        #                 continue
        #
        #             if X_test[sample_id, feature[node_id]] <= threshold[node_id]:
        #                 threshold_sign = "<="
        #             else:
        #                 threshold_sign = ">"
        #
        #             print("decision id node %s : (X_test[%s, %s] (= %s) %s %s)"
        #                   % (node_id,
        #                      sample_id,
        #                      feature[node_id],
        #                      X_test[sample_id, feature[node_id]],
        #                      threshold_sign,
        #                      threshold[node_id]))
        #         input()

        # if np.sum(abs(y_test - predicted)) > 5:
        #     print("Feature ranking:")
        #     feature_importances = dt.feature_importances_
        #     indices = np.argsort(feature_importances)[::-1]
        #     for f in range(X_test.shape[1]):
        #         if feature_importances[indices[f]] > 0.000001:
        #             print("%d. feature %d (%s) = %f" % (f + 1, indices[f], feature_names[indices[f]], feature_importances[indices[f]]))
        #     print('-----------')
        #     # for f in range(X.shape[1]):
        #     #     print("feature %d (%s) = %f" % (f, feature_names[f], feature_importances[f]))
        #     # input('verificar')
        #     # Plot the impurity-based feature importances of the forest
        #
        #     fig = plt.figure(figsize=(25,20))
        #     _ = plot_tree(dt,
        #                        feature_names=feature_names,
        #                        class_names=class_names,
        #                        filled=True)
        #     # fig.savefig("dt-columns-{}-tree-{}.png".format(i, tree))
        #
        #     dot_data = export_graphviz(dt, out_file=None,
        #                                     feature_names=feature_names,
        #                                     class_names=class_names,
        #                                     filled=True,
        #                                     label='all',
        #                                     proportion=False,
        #                                     precision=8)
        #
        #     # Draw graph
        #     graph = graphviz.Source(dot_data, format="png")
        #     graph.render("dt-columns-{}-tree-{}".format(i, tree))
        #     plt.close(fig)
