import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
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

csv = pd.read_csv('results_cbc_nocuts.csv', delimiter=';')
# input(csv.columns.tolist())
cut_types = csv.cut_type.unique()
# y = abs(1 - csv['label'])
# X['cut_type'] = le.fit_transform(X['cut_type'])

depth = [10, 20, 30, 50]
samples_split = [2, 4]
samples_leaf = [1, 2]
# estimators = [500, 1000]
estimators = [-1]
# ccp_alpha = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4]

# rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5)
# teste = csv
# y = teste['label'].copy()
# X = teste.drop(['label', 'instance', ], axis=1).copy()
#
# for i in range(6,10):
#     print('*******************')
#     print('depth ', i)
#     for train_index, test_index in rskf.split(X, y):
#         print('---------------------')
#         X_train, X_test = X.loc[train_index], X.loc[test_index]
#         X_train = X_train.drop(
#             ['relax_iteration', 'nonzeros', 'pct_nonzeros', 'unsatisfied_var', 'pct_unsatisfied_var',
#              'cut_type', 'n_variables_coef_nonzero', 'xvar_zero', 'coeff_leq_0.5', 'coeff_leq_1', 'coeff_geq_1',
#              'minor_coef', 'abs_minor_coef', 'major_coef', 'abs_major_coef', 'abs_ratio_minor_major_coef',
#              'ratio_abs_minor_major_coef', 'abs_rhs', 'rhs', 'diff', 'away', 'lub', 'eps_coeff', 'eps_coeff_lub', 'lhs',
#              'abs_lhs', 'abs_ratio_lhs_rhs', 'abs_ratio_min_max_coeff_rhs', 'abs_ratio_min_coeff_rhs'], axis=1)
#         X_test = X_test.drop(
#             ['relax_iteration', 'nonzeros', 'pct_nonzeros', 'unsatisfied_var', 'pct_unsatisfied_var',
#              'cut_type', 'n_variables_coef_nonzero', 'xvar_zero', 'coeff_leq_0.5', 'coeff_leq_1', 'coeff_geq_1',
#              'minor_coef', 'abs_minor_coef', 'major_coef', 'abs_major_coef', 'abs_ratio_minor_major_coef',
#              'ratio_abs_minor_major_coef', 'abs_rhs', 'rhs', 'diff', 'away', 'lub', 'eps_coeff', 'eps_coeff_lub', 'lhs',
#              'abs_lhs', 'abs_ratio_lhs_rhs', 'abs_ratio_min_max_coeff_rhs', 'abs_ratio_min_coeff_rhs'], axis=1)
#         y_train, y_test = y.loc[train_index], y.loc[test_index]
#         dt = DecisionTreeClassifier(max_depth=i, min_samples_split=2, min_samples_leaf=1, criterion='entropy')
#         dt.fit(X_train, y_train)
#         predicted = dt.predict(X_test)
#         print(len(y_test), len(predicted))
#         print(np.sum(y_test))
#         print('previstos como errados: ', np.sum(predicted))
#         print('previstos como certos: ', np.sum(1 - predicted))
#         print(np.sum(abs(y_test - predicted)))
#         tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
#         print('TN = {}\tFP = {}\tFN = {}\tTP = {}'.format(tn, fp, fn, tp))
#         index0 = np.where(predicted == 0)[0].tolist()
#         index1 = np.where(predicted == 1)[0].tolist()
#         # divide the train to only those wrong, and re-train with cut columns
#         dt_new = DecisionTreeClassifier(max_depth=30, min_samples_split=2, min_samples_leaf=1, criterion='entropy')
#         if len(index0) > 0:
#             print('=====================')
#             print("with predicted label = 0")
#             X_train, X_test = X.loc[train_index], X.loc[index0]
#             X_train = X_train.drop(
#                 ['cols', 'rows', 'colsPerRow', 'equalities', 'nzEqualities', 'percEqualities', 'percNzEqualities', 'inequalities', 'nzInequalities', 'nz', 'density', 'bin', 'genInt', 'integer', 'continuous', 'percInteger', 'percBin', 'nUnbounded1', 'percUnbounded1', 'nUnbounded2', 'percUnbounded2', 'rPartitioning', 'rPercPartitioning', 'rPacking', 'rPercPacking', 'rPartPacking', 'rPercRowsPartPacking', 'rCovering', 'rPercCovering', 'rCardinality', 'rPercCardinality', 'rKnapsack', 'rPercKnapsack', 'rIntegerKnapsack', 'rPercIntegerKnapsack', 'rInvKnapsack', 'rPercInvKnapsack', 'rSingleton', 'rPercSingleton', 'rAggre', 'rPercAggre', 'rPrec', 'rPercPrec', 'rVarBnd', 'rPercVarBnd', 'rBinPacking', 'rPercBinPacking', 'rMixedBin', 'rPercMixedBin', 'rGenInt', 'rPercGenInt', 'rFlowBin', 'rPercFlowBin', 'rFlowMx', 'rPercFlowMx', 'rNzRowsPartitioning', 'rNzPercRowsPartitioning', 'rNzRowsPacking', 'rNzPercRowsPacking', 'rNzrowsPartPacking', 'rNzpercRowsPartPacking', 'rNzRowsCovering', 'rNzPercRowsCovering', 'rNzRowsCardinality', 'rNzPercRowsCardinality', 'rNzRowsKnapsack', 'rNzPercRowsKnapsack', 'rNzRowsIntegerKnapsack', 'rNzPercRowsIntegerKnapsack', 'rNzRowsInvKnapsack', 'rNzPercRowsInvKnapsack', 'rNzRowsSingleton', 'rNzPercRowsSingleton', 'rNzRowsAggr', 'rNzPercRowsAggr', 'rNzRowsPrec', 'rNzPercRowsPrec', 'rNzRowsVarBnd', 'rNzPercRowsVarBnd', 'rNzRowsBinPacking', 'rNzPercRowsBinPacking', 'rNzRowsMixedBin', 'rNzPercRowsMixedBin', 'rNzRowsGenInt', 'rNzPercRowsGenInt', 'rNzRowsFlowBin', 'rNzPercRowsFlowBin', 'rNzRowsFlowMx', 'rNzPercRowsFlowMx', 'aMin', 'aMax', 'aAvg', 'aStdDev', 'aRatioLSA', 'aAllInt', 'aPercInt', 'aDiffVal', 'anShortInts', 'apercShortInts', 'objMin', 'objMax', 'objAvg', 'objStdDev', 'objRatioLSA', 'objAllInt', 'objPercInt', 'objDiffVal', 'objnShortInts', 'objpercShortInts', 'rhsMin', 'rhsMax', 'rhsAvg', 'rhsStdDev', 'rhsRatioLSA', 'rhsAllInt', 'rhsPercInt', 'rhsDiffVal', 'rhsnShortInts', 'rhspercShortInts', 'rowNzMin', 'rowNzMax', 'rowNzAvg', 'rowNzStdDev', 'colNzMin', 'colNzMax', 'colNzAvg', 'colNzStdDev', 'rowsLess4Nz', 'rowsLess8Nz', 'rowsLess16Nz', 'rowsLess32Nz', 'rowsLess64Nz', 'rowsLess128Nz', 'rowsLess256Nz', 'rowsLess512Nz', 'rowsLess1024Nz', 'percRowsLess4Nz', 'percRowsLess8Nz', 'percRowsLess16Nz', 'percRowsLess32Nz', 'percRowsLess64Nz', 'percRowsLess128Nz', 'percRowsLess256Nz', 'percRowsLess512Nz', 'percRowsLess1024Nz', 'rowsLeast4Nz', 'rowsLeast8Nz', 'rowsLeast16Nz', 'rowsLeast32Nz', 'rowsLeast64Nz', 'rowsLeast128Nz', 'rowsLeast256Nz', 'rowsLeast512Nz', 'rowsLeast1024Nz', 'rowsLeast2048Nz', 'rowsLeast4096Nz', 'percRowsLeast4Nz', 'percRowsLeast8Nz', 'percRowsLeast16Nz', 'percRowsLeast32Nz', 'percRowsLeast64Nz', 'percRowsLeast128Nz', 'percRowsLeast256Nz', 'percRowsLeast512Nz', 'percRowsLeast1024Nz', 'percRowsLeast2048Nz', 'percRowsLeast4096Nz', 'colsLess4Nz', 'colsLess8Nz', 'colsLess16Nz', 'colsLess32Nz', 'colsLess64Nz', 'colsLess128Nz', 'colsLess256Nz', 'colsLess512Nz', 'colsLess1024Nz', 'percColsLess4Nz', 'percColsLess8Nz', 'percColsLess16Nz', 'percColsLess32Nz', 'percColsLess64Nz', 'percColsLess128Nz', 'percColsLess256Nz', 'percColsLess512Nz', 'percColsLess1024Nz', 'colsLeast4Nz', 'colsLeast8Nz', 'colsLeast16Nz', 'colsLeast32Nz', 'colsLeast64Nz', 'colsLeast128Nz', 'colsLeast256Nz', 'colsLeast512Nz', 'colsLeast1024Nz', 'colsLeast2048Nz', 'colsLeast4096Nz', 'percColsLeast4Nz', 'percColsLeast8Nz', 'perccolsLeast16Nz', 'perccolsLeast32Nz', 'perccolsLeast64Nz', 'perccolsLeast128Nz', 'perccolsLeast256Nz', 'perccolsLeast512Nz', 'perccolsLeast1024Nz', 'perccolsLeast2048Nz', 'perccolsLeast4096Nz'], axis=1)
#             X_test = X_test.drop(
#                 ['cols', 'rows', 'colsPerRow', 'equalities', 'nzEqualities', 'percEqualities', 'percNzEqualities', 'inequalities', 'nzInequalities', 'nz', 'density', 'bin', 'genInt', 'integer', 'continuous', 'percInteger', 'percBin', 'nUnbounded1', 'percUnbounded1', 'nUnbounded2', 'percUnbounded2', 'rPartitioning', 'rPercPartitioning', 'rPacking', 'rPercPacking', 'rPartPacking', 'rPercRowsPartPacking', 'rCovering', 'rPercCovering', 'rCardinality', 'rPercCardinality', 'rKnapsack', 'rPercKnapsack', 'rIntegerKnapsack', 'rPercIntegerKnapsack', 'rInvKnapsack', 'rPercInvKnapsack', 'rSingleton', 'rPercSingleton', 'rAggre', 'rPercAggre', 'rPrec', 'rPercPrec', 'rVarBnd', 'rPercVarBnd', 'rBinPacking', 'rPercBinPacking', 'rMixedBin', 'rPercMixedBin', 'rGenInt', 'rPercGenInt', 'rFlowBin', 'rPercFlowBin', 'rFlowMx', 'rPercFlowMx', 'rNzRowsPartitioning', 'rNzPercRowsPartitioning', 'rNzRowsPacking', 'rNzPercRowsPacking', 'rNzrowsPartPacking', 'rNzpercRowsPartPacking', 'rNzRowsCovering', 'rNzPercRowsCovering', 'rNzRowsCardinality', 'rNzPercRowsCardinality', 'rNzRowsKnapsack', 'rNzPercRowsKnapsack', 'rNzRowsIntegerKnapsack', 'rNzPercRowsIntegerKnapsack', 'rNzRowsInvKnapsack', 'rNzPercRowsInvKnapsack', 'rNzRowsSingleton', 'rNzPercRowsSingleton', 'rNzRowsAggr', 'rNzPercRowsAggr', 'rNzRowsPrec', 'rNzPercRowsPrec', 'rNzRowsVarBnd', 'rNzPercRowsVarBnd', 'rNzRowsBinPacking', 'rNzPercRowsBinPacking', 'rNzRowsMixedBin', 'rNzPercRowsMixedBin', 'rNzRowsGenInt', 'rNzPercRowsGenInt', 'rNzRowsFlowBin', 'rNzPercRowsFlowBin', 'rNzRowsFlowMx', 'rNzPercRowsFlowMx', 'aMin', 'aMax', 'aAvg', 'aStdDev', 'aRatioLSA', 'aAllInt', 'aPercInt', 'aDiffVal', 'anShortInts', 'apercShortInts', 'objMin', 'objMax', 'objAvg', 'objStdDev', 'objRatioLSA', 'objAllInt', 'objPercInt', 'objDiffVal', 'objnShortInts', 'objpercShortInts', 'rhsMin', 'rhsMax', 'rhsAvg', 'rhsStdDev', 'rhsRatioLSA', 'rhsAllInt', 'rhsPercInt', 'rhsDiffVal', 'rhsnShortInts', 'rhspercShortInts', 'rowNzMin', 'rowNzMax', 'rowNzAvg', 'rowNzStdDev', 'colNzMin', 'colNzMax', 'colNzAvg', 'colNzStdDev', 'rowsLess4Nz', 'rowsLess8Nz', 'rowsLess16Nz', 'rowsLess32Nz', 'rowsLess64Nz', 'rowsLess128Nz', 'rowsLess256Nz', 'rowsLess512Nz', 'rowsLess1024Nz', 'percRowsLess4Nz', 'percRowsLess8Nz', 'percRowsLess16Nz', 'percRowsLess32Nz', 'percRowsLess64Nz', 'percRowsLess128Nz', 'percRowsLess256Nz', 'percRowsLess512Nz', 'percRowsLess1024Nz', 'rowsLeast4Nz', 'rowsLeast8Nz', 'rowsLeast16Nz', 'rowsLeast32Nz', 'rowsLeast64Nz', 'rowsLeast128Nz', 'rowsLeast256Nz', 'rowsLeast512Nz', 'rowsLeast1024Nz', 'rowsLeast2048Nz', 'rowsLeast4096Nz', 'percRowsLeast4Nz', 'percRowsLeast8Nz', 'percRowsLeast16Nz', 'percRowsLeast32Nz', 'percRowsLeast64Nz', 'percRowsLeast128Nz', 'percRowsLeast256Nz', 'percRowsLeast512Nz', 'percRowsLeast1024Nz', 'percRowsLeast2048Nz', 'percRowsLeast4096Nz', 'colsLess4Nz', 'colsLess8Nz', 'colsLess16Nz', 'colsLess32Nz', 'colsLess64Nz', 'colsLess128Nz', 'colsLess256Nz', 'colsLess512Nz', 'colsLess1024Nz', 'percColsLess4Nz', 'percColsLess8Nz', 'percColsLess16Nz', 'percColsLess32Nz', 'percColsLess64Nz', 'percColsLess128Nz', 'percColsLess256Nz', 'percColsLess512Nz', 'percColsLess1024Nz', 'colsLeast4Nz', 'colsLeast8Nz', 'colsLeast16Nz', 'colsLeast32Nz', 'colsLeast64Nz', 'colsLeast128Nz', 'colsLeast256Nz', 'colsLeast512Nz', 'colsLeast1024Nz', 'colsLeast2048Nz', 'colsLeast4096Nz', 'percColsLeast4Nz', 'percColsLeast8Nz', 'perccolsLeast16Nz', 'perccolsLeast32Nz', 'perccolsLeast64Nz', 'perccolsLeast128Nz', 'perccolsLeast256Nz', 'perccolsLeast512Nz', 'perccolsLeast1024Nz', 'perccolsLeast2048Nz', 'perccolsLeast4096Nz'], axis=1)
#             y_train, y_test = y.loc[train_index], y.loc[index0]
#             dt_new.fit(X_train, y_train)
#             predicted = dt_new.predict(X_test)
#             print(len(y_test), len(predicted))
#             print(np.sum(y_test))
#             print('previstos como errados: ', np.sum(predicted))
#             print('previstos como certos: ', np.sum(1 - predicted))
#             print(np.sum(abs(y_test - predicted)))
#             tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
#             print('TN = {}\tFP = {}\tFN = {}\tTP = {}'.format(tn, fp, fn, tp))
#         if len(index1) > 0:
#             print('=====================')
#             print("with predicted label = 1")
#             input()
#             X_train, X_test = X.loc[train_index], X.loc[index1]
#             X_train = X_train.drop(
#                 ['cols', 'rows', 'colsPerRow', 'equalities', 'nzEqualities', 'percEqualities', 'percNzEqualities', 'inequalities', 'nzInequalities', 'nz', 'density', 'bin', 'genInt', 'integer', 'continuous', 'percInteger', 'percBin', 'nUnbounded1', 'percUnbounded1', 'nUnbounded2', 'percUnbounded2', 'rPartitioning', 'rPercPartitioning', 'rPacking', 'rPercPacking', 'rPartPacking', 'rPercRowsPartPacking', 'rCovering', 'rPercCovering', 'rCardinality', 'rPercCardinality', 'rKnapsack', 'rPercKnapsack', 'rIntegerKnapsack', 'rPercIntegerKnapsack', 'rInvKnapsack', 'rPercInvKnapsack', 'rSingleton', 'rPercSingleton', 'rAggre', 'rPercAggre', 'rPrec', 'rPercPrec', 'rVarBnd', 'rPercVarBnd', 'rBinPacking', 'rPercBinPacking', 'rMixedBin', 'rPercMixedBin', 'rGenInt', 'rPercGenInt', 'rFlowBin', 'rPercFlowBin', 'rFlowMx', 'rPercFlowMx', 'rNzRowsPartitioning', 'rNzPercRowsPartitioning', 'rNzRowsPacking', 'rNzPercRowsPacking', 'rNzrowsPartPacking', 'rNzpercRowsPartPacking', 'rNzRowsCovering', 'rNzPercRowsCovering', 'rNzRowsCardinality', 'rNzPercRowsCardinality', 'rNzRowsKnapsack', 'rNzPercRowsKnapsack', 'rNzRowsIntegerKnapsack', 'rNzPercRowsIntegerKnapsack', 'rNzRowsInvKnapsack', 'rNzPercRowsInvKnapsack', 'rNzRowsSingleton', 'rNzPercRowsSingleton', 'rNzRowsAggr', 'rNzPercRowsAggr', 'rNzRowsPrec', 'rNzPercRowsPrec', 'rNzRowsVarBnd', 'rNzPercRowsVarBnd', 'rNzRowsBinPacking', 'rNzPercRowsBinPacking', 'rNzRowsMixedBin', 'rNzPercRowsMixedBin', 'rNzRowsGenInt', 'rNzPercRowsGenInt', 'rNzRowsFlowBin', 'rNzPercRowsFlowBin', 'rNzRowsFlowMx', 'rNzPercRowsFlowMx', 'aMin', 'aMax', 'aAvg', 'aStdDev', 'aRatioLSA', 'aAllInt', 'aPercInt', 'aDiffVal', 'anShortInts', 'apercShortInts', 'objMin', 'objMax', 'objAvg', 'objStdDev', 'objRatioLSA', 'objAllInt', 'objPercInt', 'objDiffVal', 'objnShortInts', 'objpercShortInts', 'rhsMin', 'rhsMax', 'rhsAvg', 'rhsStdDev', 'rhsRatioLSA', 'rhsAllInt', 'rhsPercInt', 'rhsDiffVal', 'rhsnShortInts', 'rhspercShortInts', 'rowNzMin', 'rowNzMax', 'rowNzAvg', 'rowNzStdDev', 'colNzMin', 'colNzMax', 'colNzAvg', 'colNzStdDev', 'rowsLess4Nz', 'rowsLess8Nz', 'rowsLess16Nz', 'rowsLess32Nz', 'rowsLess64Nz', 'rowsLess128Nz', 'rowsLess256Nz', 'rowsLess512Nz', 'rowsLess1024Nz', 'percRowsLess4Nz', 'percRowsLess8Nz', 'percRowsLess16Nz', 'percRowsLess32Nz', 'percRowsLess64Nz', 'percRowsLess128Nz', 'percRowsLess256Nz', 'percRowsLess512Nz', 'percRowsLess1024Nz', 'rowsLeast4Nz', 'rowsLeast8Nz', 'rowsLeast16Nz', 'rowsLeast32Nz', 'rowsLeast64Nz', 'rowsLeast128Nz', 'rowsLeast256Nz', 'rowsLeast512Nz', 'rowsLeast1024Nz', 'rowsLeast2048Nz', 'rowsLeast4096Nz', 'percRowsLeast4Nz', 'percRowsLeast8Nz', 'percRowsLeast16Nz', 'percRowsLeast32Nz', 'percRowsLeast64Nz', 'percRowsLeast128Nz', 'percRowsLeast256Nz', 'percRowsLeast512Nz', 'percRowsLeast1024Nz', 'percRowsLeast2048Nz', 'percRowsLeast4096Nz', 'colsLess4Nz', 'colsLess8Nz', 'colsLess16Nz', 'colsLess32Nz', 'colsLess64Nz', 'colsLess128Nz', 'colsLess256Nz', 'colsLess512Nz', 'colsLess1024Nz', 'percColsLess4Nz', 'percColsLess8Nz', 'percColsLess16Nz', 'percColsLess32Nz', 'percColsLess64Nz', 'percColsLess128Nz', 'percColsLess256Nz', 'percColsLess512Nz', 'percColsLess1024Nz', 'colsLeast4Nz', 'colsLeast8Nz', 'colsLeast16Nz', 'colsLeast32Nz', 'colsLeast64Nz', 'colsLeast128Nz', 'colsLeast256Nz', 'colsLeast512Nz', 'colsLeast1024Nz', 'colsLeast2048Nz', 'colsLeast4096Nz', 'percColsLeast4Nz', 'percColsLeast8Nz', 'perccolsLeast16Nz', 'perccolsLeast32Nz', 'perccolsLeast64Nz', 'perccolsLeast128Nz', 'perccolsLeast256Nz', 'perccolsLeast512Nz', 'perccolsLeast1024Nz', 'perccolsLeast2048Nz', 'perccolsLeast4096Nz'], axis=1)
#             X_test = X_test.drop(
#                 ['cols', 'rows', 'colsPerRow', 'equalities', 'nzEqualities', 'percEqualities', 'percNzEqualities', 'inequalities', 'nzInequalities', 'nz', 'density', 'bin', 'genInt', 'integer', 'continuous', 'percInteger', 'percBin', 'nUnbounded1', 'percUnbounded1', 'nUnbounded2', 'percUnbounded2', 'rPartitioning', 'rPercPartitioning', 'rPacking', 'rPercPacking', 'rPartPacking', 'rPercRowsPartPacking', 'rCovering', 'rPercCovering', 'rCardinality', 'rPercCardinality', 'rKnapsack', 'rPercKnapsack', 'rIntegerKnapsack', 'rPercIntegerKnapsack', 'rInvKnapsack', 'rPercInvKnapsack', 'rSingleton', 'rPercSingleton', 'rAggre', 'rPercAggre', 'rPrec', 'rPercPrec', 'rVarBnd', 'rPercVarBnd', 'rBinPacking', 'rPercBinPacking', 'rMixedBin', 'rPercMixedBin', 'rGenInt', 'rPercGenInt', 'rFlowBin', 'rPercFlowBin', 'rFlowMx', 'rPercFlowMx', 'rNzRowsPartitioning', 'rNzPercRowsPartitioning', 'rNzRowsPacking', 'rNzPercRowsPacking', 'rNzrowsPartPacking', 'rNzpercRowsPartPacking', 'rNzRowsCovering', 'rNzPercRowsCovering', 'rNzRowsCardinality', 'rNzPercRowsCardinality', 'rNzRowsKnapsack', 'rNzPercRowsKnapsack', 'rNzRowsIntegerKnapsack', 'rNzPercRowsIntegerKnapsack', 'rNzRowsInvKnapsack', 'rNzPercRowsInvKnapsack', 'rNzRowsSingleton', 'rNzPercRowsSingleton', 'rNzRowsAggr', 'rNzPercRowsAggr', 'rNzRowsPrec', 'rNzPercRowsPrec', 'rNzRowsVarBnd', 'rNzPercRowsVarBnd', 'rNzRowsBinPacking', 'rNzPercRowsBinPacking', 'rNzRowsMixedBin', 'rNzPercRowsMixedBin', 'rNzRowsGenInt', 'rNzPercRowsGenInt', 'rNzRowsFlowBin', 'rNzPercRowsFlowBin', 'rNzRowsFlowMx', 'rNzPercRowsFlowMx', 'aMin', 'aMax', 'aAvg', 'aStdDev', 'aRatioLSA', 'aAllInt', 'aPercInt', 'aDiffVal', 'anShortInts', 'apercShortInts', 'objMin', 'objMax', 'objAvg', 'objStdDev', 'objRatioLSA', 'objAllInt', 'objPercInt', 'objDiffVal', 'objnShortInts', 'objpercShortInts', 'rhsMin', 'rhsMax', 'rhsAvg', 'rhsStdDev', 'rhsRatioLSA', 'rhsAllInt', 'rhsPercInt', 'rhsDiffVal', 'rhsnShortInts', 'rhspercShortInts', 'rowNzMin', 'rowNzMax', 'rowNzAvg', 'rowNzStdDev', 'colNzMin', 'colNzMax', 'colNzAvg', 'colNzStdDev', 'rowsLess4Nz', 'rowsLess8Nz', 'rowsLess16Nz', 'rowsLess32Nz', 'rowsLess64Nz', 'rowsLess128Nz', 'rowsLess256Nz', 'rowsLess512Nz', 'rowsLess1024Nz', 'percRowsLess4Nz', 'percRowsLess8Nz', 'percRowsLess16Nz', 'percRowsLess32Nz', 'percRowsLess64Nz', 'percRowsLess128Nz', 'percRowsLess256Nz', 'percRowsLess512Nz', 'percRowsLess1024Nz', 'rowsLeast4Nz', 'rowsLeast8Nz', 'rowsLeast16Nz', 'rowsLeast32Nz', 'rowsLeast64Nz', 'rowsLeast128Nz', 'rowsLeast256Nz', 'rowsLeast512Nz', 'rowsLeast1024Nz', 'rowsLeast2048Nz', 'rowsLeast4096Nz', 'percRowsLeast4Nz', 'percRowsLeast8Nz', 'percRowsLeast16Nz', 'percRowsLeast32Nz', 'percRowsLeast64Nz', 'percRowsLeast128Nz', 'percRowsLeast256Nz', 'percRowsLeast512Nz', 'percRowsLeast1024Nz', 'percRowsLeast2048Nz', 'percRowsLeast4096Nz', 'colsLess4Nz', 'colsLess8Nz', 'colsLess16Nz', 'colsLess32Nz', 'colsLess64Nz', 'colsLess128Nz', 'colsLess256Nz', 'colsLess512Nz', 'colsLess1024Nz', 'percColsLess4Nz', 'percColsLess8Nz', 'percColsLess16Nz', 'percColsLess32Nz', 'percColsLess64Nz', 'percColsLess128Nz', 'percColsLess256Nz', 'percColsLess512Nz', 'percColsLess1024Nz', 'colsLeast4Nz', 'colsLeast8Nz', 'colsLeast16Nz', 'colsLeast32Nz', 'colsLeast64Nz', 'colsLeast128Nz', 'colsLeast256Nz', 'colsLeast512Nz', 'colsLeast1024Nz', 'colsLeast2048Nz', 'colsLeast4096Nz', 'percColsLeast4Nz', 'percColsLeast8Nz', 'perccolsLeast16Nz', 'perccolsLeast32Nz', 'perccolsLeast64Nz', 'perccolsLeast128Nz', 'perccolsLeast256Nz', 'perccolsLeast512Nz', 'perccolsLeast1024Nz', 'perccolsLeast2048Nz', 'perccolsLeast4096Nz'], axis=1)
#             y_train, y_test = y.loc[train_index], y.loc[index1]
#             dt_new.fit(X_train, y_train)
#             predicted = dt_new.predict(X_test)
#             print(len(y_test), len(predicted))
#             print(np.sum(y_test))
#             print('previstos como errados: ', np.sum(predicted))
#             print('previstos como certos: ', np.sum(1 - predicted))
#             print(np.sum(abs(y_test - predicted)))
#             tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
#             print('TN = {}\tFP = {}\tFN = {}\tTP = {}'.format(tn, fp, fn, tp))
#
# print('encerrado')
# exit(0)


dropcolumns = list()
dropcolumns.append(['label', 'instance', 'relax_iteration', 'nonzeros', 'pct_nonzeros', 'unsatisfied_var', 'pct_unsatisfied_var',
                             'cut_type', 'n_variables_coef_nonzero', 'xvar_zero', 'coeff_leq_0.5', 'coeff_leq_1', 'coeff_geq_1',
                             'minor_coef', 'abs_minor_coef', 'major_coef', 'abs_major_coef', 'abs_ratio_minor_major_coef',
                             'ratio_abs_minor_major_coef', 'abs_rhs', 'rhs', 'diff', 'away', 'lub', 'eps_coeff', 'eps_coeff_lub', 'lhs',
                             'abs_lhs', 'abs_ratio_lhs_rhs', 'abs_ratio_min_max_coeff_rhs', 'abs_ratio_min_coeff_rhs'])
dropcolumns.append(['label', 'instance', 'cols', 'rows', 'colsPerRow', 'equalities', 'nzEqualities', 'percEqualities', 'percNzEqualities', 'inequalities', 'nzInequalities', 'nz', 'density', 'bin', 'genInt', 'integer', 'continuous', 'percInteger', 'percBin', 'nUnbounded1', 'percUnbounded1', 'nUnbounded2', 'percUnbounded2', 'rPartitioning', 'rPercPartitioning', 'rPacking', 'rPercPacking', 'rPartPacking', 'rPercRowsPartPacking', 'rCovering', 'rPercCovering', 'rCardinality', 'rPercCardinality', 'rKnapsack', 'rPercKnapsack', 'rIntegerKnapsack', 'rPercIntegerKnapsack', 'rInvKnapsack', 'rPercInvKnapsack', 'rSingleton', 'rPercSingleton', 'rAggre', 'rPercAggre', 'rPrec', 'rPercPrec', 'rVarBnd', 'rPercVarBnd', 'rBinPacking', 'rPercBinPacking', 'rMixedBin', 'rPercMixedBin', 'rGenInt', 'rPercGenInt', 'rFlowBin', 'rPercFlowBin', 'rFlowMx', 'rPercFlowMx', 'rNzRowsPartitioning', 'rNzPercRowsPartitioning', 'rNzRowsPacking', 'rNzPercRowsPacking', 'rNzrowsPartPacking', 'rNzpercRowsPartPacking', 'rNzRowsCovering', 'rNzPercRowsCovering', 'rNzRowsCardinality', 'rNzPercRowsCardinality', 'rNzRowsKnapsack', 'rNzPercRowsKnapsack', 'rNzRowsIntegerKnapsack', 'rNzPercRowsIntegerKnapsack', 'rNzRowsInvKnapsack', 'rNzPercRowsInvKnapsack', 'rNzRowsSingleton', 'rNzPercRowsSingleton', 'rNzRowsAggr', 'rNzPercRowsAggr', 'rNzRowsPrec', 'rNzPercRowsPrec', 'rNzRowsVarBnd', 'rNzPercRowsVarBnd', 'rNzRowsBinPacking', 'rNzPercRowsBinPacking', 'rNzRowsMixedBin', 'rNzPercRowsMixedBin', 'rNzRowsGenInt', 'rNzPercRowsGenInt', 'rNzRowsFlowBin', 'rNzPercRowsFlowBin', 'rNzRowsFlowMx', 'rNzPercRowsFlowMx', 'aMin', 'aMax', 'aAvg', 'aStdDev', 'aRatioLSA', 'aAllInt', 'aPercInt', 'aDiffVal', 'anShortInts', 'apercShortInts', 'objMin', 'objMax', 'objAvg', 'objStdDev', 'objRatioLSA', 'objAllInt', 'objPercInt', 'objDiffVal', 'objnShortInts', 'objpercShortInts', 'rhsMin', 'rhsMax', 'rhsAvg', 'rhsStdDev', 'rhsRatioLSA', 'rhsAllInt', 'rhsPercInt', 'rhsDiffVal', 'rhsnShortInts', 'rhspercShortInts', 'rowNzMin', 'rowNzMax', 'rowNzAvg', 'rowNzStdDev', 'colNzMin', 'colNzMax', 'colNzAvg', 'colNzStdDev', 'rowsLess4Nz', 'rowsLess8Nz', 'rowsLess16Nz', 'rowsLess32Nz', 'rowsLess64Nz', 'rowsLess128Nz', 'rowsLess256Nz', 'rowsLess512Nz', 'rowsLess1024Nz', 'percRowsLess4Nz', 'percRowsLess8Nz', 'percRowsLess16Nz', 'percRowsLess32Nz', 'percRowsLess64Nz', 'percRowsLess128Nz', 'percRowsLess256Nz', 'percRowsLess512Nz', 'percRowsLess1024Nz', 'rowsLeast4Nz', 'rowsLeast8Nz', 'rowsLeast16Nz', 'rowsLeast32Nz', 'rowsLeast64Nz', 'rowsLeast128Nz', 'rowsLeast256Nz', 'rowsLeast512Nz', 'rowsLeast1024Nz', 'rowsLeast2048Nz', 'rowsLeast4096Nz', 'percRowsLeast4Nz', 'percRowsLeast8Nz', 'percRowsLeast16Nz', 'percRowsLeast32Nz', 'percRowsLeast64Nz', 'percRowsLeast128Nz', 'percRowsLeast256Nz', 'percRowsLeast512Nz', 'percRowsLeast1024Nz', 'percRowsLeast2048Nz', 'percRowsLeast4096Nz', 'colsLess4Nz', 'colsLess8Nz', 'colsLess16Nz', 'colsLess32Nz', 'colsLess64Nz', 'colsLess128Nz', 'colsLess256Nz', 'colsLess512Nz', 'colsLess1024Nz', 'percColsLess4Nz', 'percColsLess8Nz', 'percColsLess16Nz', 'percColsLess32Nz', 'percColsLess64Nz', 'percColsLess128Nz', 'percColsLess256Nz', 'percColsLess512Nz', 'percColsLess1024Nz', 'colsLeast4Nz', 'colsLeast8Nz', 'colsLeast16Nz', 'colsLeast32Nz', 'colsLeast64Nz', 'colsLeast128Nz', 'colsLeast256Nz', 'colsLeast512Nz', 'colsLeast1024Nz', 'colsLeast2048Nz', 'colsLeast4096Nz', 'percColsLeast4Nz', 'percColsLeast8Nz', 'perccolsLeast16Nz', 'perccolsLeast32Nz', 'perccolsLeast64Nz', 'perccolsLeast128Nz', 'perccolsLeast256Nz', 'perccolsLeast512Nz', 'perccolsLeast1024Nz', 'perccolsLeast2048Nz', 'perccolsLeast4096Nz'])
dropcolumns.append(['label', 'instance'])

for columns in dropcolumns:
    print()
    print(columns)
    print('depth;samples_split;samples_leaf;estimators;ccp_alpha;recall;std')
    for d in depth:
        for split in samples_split:
            for leaf in samples_leaf:
                for e in estimators:
                    # for a in ccp_alpha:
                        # for cut in cut_types:
                        #     cond_type = csv['cut_type'] == cut
                        #     teste = csv[cond_type]
                        teste = csv
                    # diff = abs(csv['diff']) >= 1e-4
                    #     teste = teste[(teste['label'].eq(0)) | (teste['label'].eq(1) & teste['diff'].abs().ge(1e-4))]
                        # input(teste[['label','diff']].query('label == 1'))
                        y = teste['label']
                        # X = teste.drop(['label', 'instance', 'relax_iteration', 'nonzeros', 'pct_nonzeros', 'unsatisfied_var', 'pct_unsatisfied_var',
                        #          'cut_type', 'n_variables_coef_nonzero', 'xvar_zero', 'coeff_leq_0.5', 'coeff_leq_1', 'coeff_geq_1',
                        #          'minor_coef', 'abs_minor_coef', 'major_coef', 'abs_major_coef', 'abs_ratio_minor_major_coef',
                        #          'ratio_abs_minor_major_coef', 'abs_rhs', 'rhs', 'diff', 'away', 'lub', 'eps_coeff', 'eps_coeff_lub', 'lhs',
                        #          'abs_lhs', 'abs_ratio_lhs_rhs', 'abs_ratio_min_max_coeff_rhs', 'abs_ratio_min_coeff_rhs'], axis=1)
                        # X = teste.drop(['label', 'instance', 'cols', 'rows', 'colsPerRow', 'equalities', 'nzEqualities', 'percEqualities', 'percNzEqualities', 'inequalities', 'nzInequalities', 'nz', 'density', 'bin', 'genInt', 'integer', 'continuous', 'percInteger', 'percBin', 'nUnbounded1', 'percUnbounded1', 'nUnbounded2', 'percUnbounded2', 'rPartitioning', 'rPercPartitioning', 'rPacking', 'rPercPacking', 'rPartPacking', 'rPercRowsPartPacking', 'rCovering', 'rPercCovering', 'rCardinality', 'rPercCardinality', 'rKnapsack', 'rPercKnapsack', 'rIntegerKnapsack', 'rPercIntegerKnapsack', 'rInvKnapsack', 'rPercInvKnapsack', 'rSingleton', 'rPercSingleton', 'rAggre', 'rPercAggre', 'rPrec', 'rPercPrec', 'rVarBnd', 'rPercVarBnd', 'rBinPacking', 'rPercBinPacking', 'rMixedBin', 'rPercMixedBin', 'rGenInt', 'rPercGenInt', 'rFlowBin', 'rPercFlowBin', 'rFlowMx', 'rPercFlowMx', 'rNzRowsPartitioning', 'rNzPercRowsPartitioning', 'rNzRowsPacking', 'rNzPercRowsPacking', 'rNzrowsPartPacking', 'rNzpercRowsPartPacking', 'rNzRowsCovering', 'rNzPercRowsCovering', 'rNzRowsCardinality', 'rNzPercRowsCardinality', 'rNzRowsKnapsack', 'rNzPercRowsKnapsack', 'rNzRowsIntegerKnapsack', 'rNzPercRowsIntegerKnapsack', 'rNzRowsInvKnapsack', 'rNzPercRowsInvKnapsack', 'rNzRowsSingleton', 'rNzPercRowsSingleton', 'rNzRowsAggr', 'rNzPercRowsAggr', 'rNzRowsPrec', 'rNzPercRowsPrec', 'rNzRowsVarBnd', 'rNzPercRowsVarBnd', 'rNzRowsBinPacking', 'rNzPercRowsBinPacking', 'rNzRowsMixedBin', 'rNzPercRowsMixedBin', 'rNzRowsGenInt', 'rNzPercRowsGenInt', 'rNzRowsFlowBin', 'rNzPercRowsFlowBin', 'rNzRowsFlowMx', 'rNzPercRowsFlowMx', 'aMin', 'aMax', 'aAvg', 'aStdDev', 'aRatioLSA', 'aAllInt', 'aPercInt', 'aDiffVal', 'anShortInts', 'apercShortInts', 'objMin', 'objMax', 'objAvg', 'objStdDev', 'objRatioLSA', 'objAllInt', 'objPercInt', 'objDiffVal', 'objnShortInts', 'objpercShortInts', 'rhsMin', 'rhsMax', 'rhsAvg', 'rhsStdDev', 'rhsRatioLSA', 'rhsAllInt', 'rhsPercInt', 'rhsDiffVal', 'rhsnShortInts', 'rhspercShortInts', 'rowNzMin', 'rowNzMax', 'rowNzAvg', 'rowNzStdDev', 'colNzMin', 'colNzMax', 'colNzAvg', 'colNzStdDev', 'rowsLess4Nz', 'rowsLess8Nz', 'rowsLess16Nz', 'rowsLess32Nz', 'rowsLess64Nz', 'rowsLess128Nz', 'rowsLess256Nz', 'rowsLess512Nz', 'rowsLess1024Nz', 'percRowsLess4Nz', 'percRowsLess8Nz', 'percRowsLess16Nz', 'percRowsLess32Nz', 'percRowsLess64Nz', 'percRowsLess128Nz', 'percRowsLess256Nz', 'percRowsLess512Nz', 'percRowsLess1024Nz', 'rowsLeast4Nz', 'rowsLeast8Nz', 'rowsLeast16Nz', 'rowsLeast32Nz', 'rowsLeast64Nz', 'rowsLeast128Nz', 'rowsLeast256Nz', 'rowsLeast512Nz', 'rowsLeast1024Nz', 'rowsLeast2048Nz', 'rowsLeast4096Nz', 'percRowsLeast4Nz', 'percRowsLeast8Nz', 'percRowsLeast16Nz', 'percRowsLeast32Nz', 'percRowsLeast64Nz', 'percRowsLeast128Nz', 'percRowsLeast256Nz', 'percRowsLeast512Nz', 'percRowsLeast1024Nz', 'percRowsLeast2048Nz', 'percRowsLeast4096Nz', 'colsLess4Nz', 'colsLess8Nz', 'colsLess16Nz', 'colsLess32Nz', 'colsLess64Nz', 'colsLess128Nz', 'colsLess256Nz', 'colsLess512Nz', 'colsLess1024Nz', 'percColsLess4Nz', 'percColsLess8Nz', 'percColsLess16Nz', 'percColsLess32Nz', 'percColsLess64Nz', 'percColsLess128Nz', 'percColsLess256Nz', 'percColsLess512Nz', 'percColsLess1024Nz', 'colsLeast4Nz', 'colsLeast8Nz', 'colsLeast16Nz', 'colsLeast32Nz', 'colsLeast64Nz', 'colsLeast128Nz', 'colsLeast256Nz', 'colsLeast512Nz', 'colsLeast1024Nz', 'colsLeast2048Nz', 'colsLeast4096Nz', 'percColsLeast4Nz', 'percColsLeast8Nz', 'perccolsLeast16Nz', 'perccolsLeast32Nz', 'perccolsLeast64Nz', 'perccolsLeast128Nz', 'perccolsLeast256Nz', 'perccolsLeast512Nz', 'perccolsLeast1024Nz', 'perccolsLeast2048Nz', 'perccolsLeast4096Nz'], axis=1)
                        # X = teste.drop(['label', 'instance'], axis=1)
                        X = teste.drop(columns, axis=1)
                        dt = DecisionTreeClassifier(max_depth=d, min_samples_split=split, min_samples_leaf=leaf, criterion='entropy')
                        # dt = RandomForestClassifier(n_estimators=e, max_depth=d, min_samples_split=split, min_samples_leaf=leaf, criterion='entropy')
                        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
                        # calculate 5-fold cross validation
                        res = cross_val_score(dt, X, y, cv=cv, n_jobs=-1, scoring='recall', error_score=0)
                        # calculate the mean of the scores
                        estimate = mean(res)
                        stdv = std(res)
                        print('{};{};{};{};{};{}'.format(d, split, leaf, e, estimate, stdv))
                        # print('{};{};{};{};{};{}'.format(d, split, leaf, cut, estimate, stdv))

# dt = DecisionTreeClassifier(max_depth=110, min_samples_split=2, min_samples_leaf=1, criterion='entropy')
# cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# # # calculate 5-fold cross validation
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
