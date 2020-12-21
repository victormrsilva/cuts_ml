import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.model_selection import StratifiedKFold, cross_validate
import matplotlib.pyplot as plt
import graphviz


csv = pd.read_csv('results_cbc_nocuts.csv', delimiter=';')
csv2 = pd.read_csv('results_cbc_nocuts.csv', delimiter=';')

# csv['artificial1'] = -0.003403358048048588*csv['nonzeros'] -0.0033473539343378267*csv['pct_nonzeros'] +0.001093077610050175*csv['unsatisfied_var'] +0.0047309863585918155*csv['pct_unsatisfied_var'] -0.00018319249394804782*csv['cut_type'] -0.002065139320669984*csv['major_coef'] -0.001413178456856734*csv['diff'] +0.0*csv['abs_lhs'] +0.9820538327505921*csv['abs_ratio_lhs_rhs'] -0.00042836836215331964*csv['abs_ratio_min_max_coeff_rhs'] -0.0012815126647515163*csv['abs_ratio_min_coeff_rhs']
# csv['artificial2'] = +0.0066767595714056875*csv['nonzeros'] -0.007884582471085443*csv['pct_nonzeros'] +0.05744794928543838*csv['unsatisfied_var'] +0.029237225679894844*csv['pct_unsatisfied_var'] +0.04366802239206486*csv['cut_type'] +0.17315562862075484*csv['major_coef'] -0.6480067484374352*csv['diff'] +1.713431909776517e-13*csv['abs_lhs'] +1.8352431728807314e-16*csv['abs_ratio_lhs_rhs'] -0.025449753826173364*csv['abs_ratio_min_max_coeff_rhs'] +0.008473329715721876*csv['abs_ratio_min_coeff_rhs']
# csv['artificial3'] = +0.0034436501789731184*csv['nonzeros'] -0.000787941535273346*csv['pct_nonzeros'] -0.021256539389307966*csv['unsatisfied_var'] +0.001326902834249494*csv['pct_unsatisfied_var'] -0.007152040935053871*csv['cut_type'] -7.000476587304405e-15*csv['major_coef'] +0.059851440322681485*csv['diff'] -0.7908743668151631*csv['abs_lhs'] -0.10688050886863627*csv['abs_ratio_lhs_rhs'] +0.006119904185750139*csv['abs_ratio_min_max_coeff_rhs'] -0.0023067049348888644*csv['abs_ratio_min_coeff_rhs']
# csv['artificial4'] = -0.012942086822047446*csv['nonzeros'] -0.0012059393890139155*csv['pct_nonzeros'] +0.015465940623571478*csv['unsatisfied_var'] -0.000953987023307987*csv['pct_unsatisfied_var'] +0.005036324669971013*csv['cut_type'] +0.735442137921734*csv['major_coef'] -0.012340297229879296*csv['diff'] +0.0*csv['abs_lhs'] +0.21002887180669522*csv['abs_ratio_lhs_rhs'] -0.004572784719136009*csv['abs_ratio_min_max_coeff_rhs'] +0.0020116297946437952*csv['abs_ratio_min_coeff_rhs']
# csv['artificial5'] = -0.01244238766878612*csv['nonzeros'] +0.0018439799810793687*csv['pct_nonzeros'] -0.09713881388863378*csv['unsatisfied_var'] -0.008604796387652155*csv['pct_unsatisfied_var'] -0.0020963395220577456*csv['cut_type'] -0.4388163413135029*csv['major_coef'] +0.07897315876947161*csv['diff'] -0.16511895366855506*csv['abs_lhs'] -0.15668766366435502*csv['abs_ratio_lhs_rhs'] +0.027971322521546754*csv['abs_ratio_min_max_coeff_rhs'] -0.010306242614359347*csv['abs_ratio_min_coeff_rhs']

csv['artificial1'] = -0.1744807168326663*csv['nonzeros'] +0.010085703063228096*csv['pct_nonzeros'] +0.4742549013887064*csv['unsatisfied_var'] -0.051291447305102315*csv['pct_unsatisfied_var'] -0.0016723319240376352*csv['cut_type'] -0.06093343465342158*csv['major_coef'] +0.021651790661732295*csv['diff'] +0.0*csv['abs_lhs'] -0.004342257662599915*csv['abs_ratio_lhs_rhs'] +0.18645935966360977*csv['abs_ratio_min_max_coeff_rhs'] +0.014828056844895661*csv['abs_ratio_min_coeff_rhs']
csv['artificial2'] = +0.13539319286129284*csv['nonzeros'] -0.011999930559718234*csv['pct_nonzeros'] -0.31422719253725667*csv['unsatisfied_var'] +0.03037892639129457*csv['pct_unsatisfied_var'] +0.004258211369099517*csv['cut_type'] -0.012941731709491602*csv['major_coef'] +0.0048220625772758315*csv['diff'] +0.09249650302166813*csv['abs_lhs'] -0.26953432369433283*csv['abs_ratio_lhs_rhs'] -0.12296226474967395*csv['abs_ratio_min_max_coeff_rhs'] -0.0009856605288433935*csv['abs_ratio_min_coeff_rhs']
csv['artificial3'] = +0.1030207624028841*csv['nonzeros'] -0.007972923080385675*csv['pct_nonzeros'] -0.39086077431083116*csv['unsatisfied_var'] +0.0884701320441649*csv['pct_unsatisfied_var'] +0.010350047592909702*csv['cut_type'] +0.028604828879140683*csv['major_coef'] +3.660266534311063e-16*csv['diff'] +0.0*csv['abs_lhs'] +0.01211054391299166*csv['abs_ratio_lhs_rhs'] -0.3551940727431257*csv['abs_ratio_min_max_coeff_rhs'] -0.0034159150335669676*csv['abs_ratio_min_coeff_rhs']
csv['artificial4'] = -0.0881656937795462*csv['nonzeros'] +0.010430111336313375*csv['pct_nonzeros'] -0.12584880214324007*csv['unsatisfied_var'] -0.04248417291576515*csv['pct_unsatisfied_var'] -0.0154942910212815*csv['cut_type'] +0.03772700363221622*csv['major_coef'] -0.04167980426574991*csv['diff'] -0.42474851769640587*csv['abs_lhs'] -0.014938225206287313*csv['abs_ratio_lhs_rhs'] +0.19153441116119738*csv['abs_ratio_min_max_coeff_rhs'] +0.006948966841997013*csv['abs_ratio_min_coeff_rhs']
csv['artificial5'] = -0.08276516877824314*csv['nonzeros'] +0.007584018584595537*csv['pct_nonzeros'] +0.24845030830272297*csv['unsatisfied_var'] -0.022246420742644823*csv['pct_unsatisfied_var'] -0.0031614490358119335*csv['cut_type'] -2.7107793928616905e-14*csv['major_coef'] -0.5005663785146819*csv['diff'] +0.0*csv['abs_lhs'] -0.005011089670016249*csv['abs_ratio_lhs_rhs'] +0.12996688073525053*csv['abs_ratio_min_max_coeff_rhs'] +0.00024828563609125*csv['abs_ratio_min_coeff_rhs']

y = csv['label'].copy()
X = csv.drop(['label', 'instance' ], axis=1).copy()
rskf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
for train_index, test_index in rskf.split(X, y):
    print('---------------------')
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]
    dt = DecisionTreeClassifier(max_depth=20, min_samples_split=2, min_samples_leaf=1, criterion='entropy')
    dt.fit(X_train, y_train)
    feature_names = X_train.columns.values
    class_names = ['corte certo', 'corte errado']
    predicted = dt.predict(X_test)
    print('previstos como errados: ', np.sum(predicted), 'atualmente errados: ', np.sum(y_test))
    print('previstos como certos: ', np.sum(1-predicted), 'atualmente errados: ', np.sum(1-y_test))
    tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
    print('TN = {}\tFP = {}\tFN = {}\tTP = {}'.format(tn, fp, fn, tp))
    recall = recall_score(y_test, predicted)
    print('recall ', recall)
    print("Feature ranking:")
    feature_importances = dt.feature_importances_
    indices = np.argsort(feature_importances)[::-1]
    for f in range(0,10):
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

    dot_data = export_graphviz(dt, out_file=None,
                                    feature_names=feature_names,
                                    class_names=class_names,
                                    filled=True,
                                    label='all',
                                    proportion=False,
                                    precision=8)

    plt.close(fig)

# y = csv2['label'].copy()
# X = csv2.drop(['label', 'instance'], axis=1).copy()
# rskf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
# for train_index, test_index in rskf.split(X, y):
#     print('---------------------')
#     X_train, X_test = X.loc[train_index], X.loc[test_index]
#     y_train, y_test = y.loc[train_index], y.loc[test_index]
#     dt = DecisionTreeClassifier(max_depth=20, min_samples_split=2, min_samples_leaf=1, criterion='entropy')
#     dt.fit(X_train, y_train)
#     feature_names = X_train.columns.values
#     class_names = ['corte certo', 'corte errado']
#     predicted = dt.predict(X_test)
#     print('previstos como errados: ', np.sum(predicted), 'atualmente errados: ', np.sum(y_test))
#     print('previstos como certos: ', np.sum(1-predicted), 'atualmente errados: ', np.sum(1-y_test))
#     tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
#     print('TN = {}\tFP = {}\tFN = {}\tTP = {}'.format(tn, fp, fn, tp))
#     recall = recall_score(y_test, predicted)
#     print('recall ', recall)
#     print("Feature ranking:")
#     feature_importances = dt.feature_importances_
#     indices = np.argsort(feature_importances)[::-1]
#     for f in range(0, 10):
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
#
#     dot_data = export_graphviz(dt, out_file=None,
#                                     feature_names=feature_names,
#                                     class_names=class_names,
#                                     filled=True,
#                                     label='all',
#                                     proportion=False,
#                                     precision=8)
#
#     plt.close(fig)
