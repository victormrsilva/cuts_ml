import logging
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer as load_data
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
import teste

from pyoptree.optree import OptimalHyperTreeModel, OptimalTreeModel

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', )

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def run(filename: str, depth: int = 2):

    csv = pd.read_csv(filename, delimiter=';')
    csv = csv.drop(csv.columns.difference(['nonzeros',
'pct_nonzeros',
'unsatisfied_var',
'pct_unsatisfied_var',
'abs_ratio_min_coeff_rhs',
'abs_lhs',
'diff',
'major_coef',
'abs_ratio_lhs_rhs',
'abs_ratio_min_max_coeff_rhs',
'cut_type',
# 'coeff_leq_0.5',
# 'rhs',
# 'ratio_abs_minor_major_coef',
# 'abs_major_coef',
# 'xvar_zero',
# 'abs_rhs',
# 'n_variables_coef_nonzero',
# 'relax_iteration',
# 'away',
# 'abs_ratio_minor_major_coef',
# 'abs_minor_coef',
# 'lhs',
# 'lub',
# 'minor_coef',
# 'coeff_leq_1',
# 'coeff_geq_1',
'label']
), 1)
    y = csv['label']
    X = csv.drop(['label'], axis=1)
    column_names = X.columns.tolist()
    print(column_names)
    cv = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    accuracy = []
    f = open("arvores_{}_depth_{}.txt".format(filename, depth), "a")
    f.write("Arquivo {} Depth {}\n".format(filename, depth))
    f.write("Label 0: {}\n".format(csv[csv['label'] == 0].shape[0]))
    f.write("Label 1: {}\n".format(csv[csv['label'] == 1].shape[0]))
    print("Arquivo {} Depth {}\n".format(filename, depth))
    print("Label 0: {}\n".format(csv[csv['label'] == 0].shape[0]))
    print("Label 1: {}\n".format(csv[csv['label'] == 1].shape[0]))

    for train_index, test_index in cv.split(X, y):
        f.write("Tamanho treino {} Tamanho teste {}\n".format(len(train_index), len(test_index)))
        print("Tamanho treino {} Tamanho teste {}\n".format(len(train_index), len(test_index)))

        train = csv.iloc[train_index,].reset_index()
        test = csv.iloc[test_index,].reset_index()

        model = OptimalHyperTreeModel(column_names, 'label', tree_depth=depth, N_min=1)
        model.train(train, train_method="mio")
        # input(test)
        test = model.predict(test)
        acc = sum(test["prediction"]==test["label"]) / len(test["label"])
        linha = "PyOptree Library Tree Prediction Accuracy: {}\n".format(acc)
        accuracy.append(acc)
        f.write(linha)
        print(linha)

        # print(model.pprint())

        lista = list()
        for index, itens in model.a.items():
            eq = ''
            for c in range(len(itens)):
                if itens[c] >= 0:
                    eq = eq + '+{} {} '.format(itens[c], column_names[c])
                else:
                    eq = eq + '{} {} '.format(itens[c], column_names[c])
            eq = eq + '>= {}'.format(model.b[index])
            lista.append(eq)
            # print(eq)

        for index, itens in model.Nkt.items():
            leaves = ''
            for c in range(len(itens)):
                leaves = leaves + "class {}: {}\t".format(c, round(itens[c]))
            lista.append(leaves)
            # print(leaves)

        # print(lista)
        conv = teste.Conversion()
        conv.list(lista)
        conv.convertList2Binary()

        f.write("Inorder Traversal of the contructed Binary Tree is:\n")
        conv.inorderTraversal(conv.root, 0, f)

        # print(test[['index','label','prediction']])
        f.write('-----------------\n\n')
        print('-----------------\n\n')
    f.write("Accuracies: {}".format(accuracy))
    print("Accuracies: {}".format(accuracy))
    accuracy = np.array(accuracy)
    f.write("Mean accuracy: {}".format(np.mean(accuracy)))
    print("Mean accuracy: {}".format(np.mean(accuracy)))
    f.write("STD: {}".format(np.std(accuracy)))
    print("STD: {}".format(np.std(accuracy)))
    f.close()

def run_depth1(filename: str):

    csv = pd.read_csv(filename, delimiter=';')
    csv = csv.drop(csv.columns.difference(['nonzeros',
'pct_nonzeros',
'unsatisfied_var',
'pct_unsatisfied_var',
'abs_ratio_min_coeff_rhs',
'abs_lhs',
'diff',
'major_coef',
'abs_ratio_lhs_rhs',
'abs_ratio_min_max_coeff_rhs',
'cut_type',
# 'coeff_leq_0.5',
# 'rhs',
# 'ratio_abs_minor_major_coef',
# 'abs_major_coef',
# 'xvar_zero',
# 'abs_rhs',
# 'n_variables_coef_nonzero',
# 'relax_iteration',
# 'away',
# 'abs_ratio_minor_major_coef',
# 'abs_minor_coef',
# 'lhs',
# 'lub',
# 'minor_coef',
# 'coeff_leq_1',
# 'coeff_geq_1',
'label']
), 1)
    y = csv['label']
    X = csv.drop(['label'], axis=1)
    column_names = X.columns.tolist()
    print(column_names)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True, stratify=y)
    f = open("arvores_{}_depth_{}.txt".format(filename, 1), "a")
    f.write("Arquivo {} Depth {}\n".format(filename, 1))
    f.write("Label 0: {}\n".format(csv[csv['label'] == 0].shape[0]))
    f.write("Label 1: {}\n".format(csv[csv['label'] == 1].shape[0]))
    print("Arquivo {} Depth {}\n".format(filename, 1))
    print("Label 0: {}\n".format(csv[csv['label'] == 0].shape[0]))
    print("Label 1: {}\n".format(csv[csv['label'] == 1].shape[0]))

    train_index = X_train.index
    test_index = X_test.index

    f.write("Tamanho treino {} Tamanho teste {}\n".format(len(train_index), len(test_index)))
    print("Tamanho treino {} Tamanho teste {}\n".format(len(train_index), len(test_index)))

    f.write("Tamanho treino {} Tamanho teste {}\n".format(len(train_index), len(test_index)))
    print("Tamanho treino {} Tamanho teste {}\n".format(len(train_index), len(test_index)))

    train = csv.iloc[train_index,].reset_index()
    test = csv.iloc[test_index,].reset_index()

    model = OptimalHyperTreeModel(column_names, 'label', tree_depth=1, N_min=1)
    model.train(train, train_method="mio")
    # input(test)
    test = model.predict(test)
    acc = sum(test["prediction"]==test["label"]) / len(test["label"])
    linha = "PyOptree Library Tree Prediction Accuracy: {}\n".format(acc)

    f.write(linha)
    print(linha)

    # print(model.pprint())

    lista = list()
    for index, itens in model.a.items():
        eq = ''
        for c in range(len(itens)):
            if itens[c] >= 0:
                eq = eq + '+{} {} '.format(itens[c], column_names[c])
            else:
                eq = eq + '{} {} '.format(itens[c], column_names[c])
        eq = eq + '>= {}'.format(model.b[index])
        lista.append(eq)
        # print(eq)

    for index, itens in model.Nkt.items():
        leaves = ''
        for c in range(len(itens)):
            leaves = leaves + "class {}: {}\t".format(c, round(itens[c]))
        lista.append(leaves)
        # print(leaves)

    # print(lista)
    conv = teste.Conversion()
    conv.list(lista)
    conv.convertList2Binary()

    f.write("Inorder Traversal of the contructed Binary Tree is:\n")
    conv.inorderTraversal(conv.root, 0, f)

    # print(test[['index','label','prediction']])
    f.write('-----------------\n\n')
    print('-----------------\n\n')
    f.close()

def run_example(depth: int = 2):
    features, label = load_data(return_X_y=True)
    p = features.shape[1]
    column_names = ["x{0}".format(i) for i in range(p)]
    data = pd.DataFrame(data=features, columns=column_names)
    data["label"] = label
    np.random.seed(0)
    test_indices = np.random.random_integers(0, data.shape[0]-1, size=(int(data.shape[0] * 0.2), ))
    train_indices = [i for i in range(0, data.shape[0]) if i not in test_indices]

    train = data.iloc[train_indices, ].reset_index()
    test = data.iloc[test_indices, ].reset_index()

    print(train.shape)

    # Use sklearn
    train_features_sklearn = features[train_indices, ::]
    train_label_sklearn = label[train_indices]
    test_features_sklearn = features[test_indices, ::]
    test_label_sklearn = label[test_indices]
    cart_model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=1)
    clf = cart_model.fit(train_features_sklearn, train_label_sklearn)
    predicted_y = clf.predict(test_features_sklearn)

    # Use PyOptree
    model = OptimalHyperTreeModel(column_names, "label", tree_depth=depth, N_min=1)
    model.train(train, train_method="mio")

    test = model.predict(test)

    print("PyOptree Library Tree Prediction Accuracy: {}".format(sum(test["prediction"]==test["label"]) / len(test["label"])))
    print("SKLearn Library Tree Prediction Accuracy: {}".format(sum(predicted_y==test_label_sklearn) / len(test_label_sklearn)))

    print(model.pprint())

    lista = list()
    for index, itens in model.a.items():
        eq = ''
        for c in range(len(itens)):
            if itens[c] >= 0:
                eq = eq + '+{} {} '.format(itens[c], column_names[c])
            else:
                eq = eq + '{} {} '.format(itens[c], column_names[c])
        eq = eq + '>= {}'.format(model.b[index])
        lista.append(eq)
        print(eq)

    for index, itens in model.Nkt.items():
        leaves = ''
        for c in range(len(itens)):
            leaves = leaves + "class {}: {}\t".format(c, round(itens[c]))
        lista.append(leaves)
        print(leaves)

    print(lista)
    conv = teste.Conversion()
    conv.list(lista)
    conv.convertList2Binary()

    print("Inorder Traversal of the contructed Binary Tree is:")
    conv.inorderTraversal(conv.root, 0)

    print(test[['index','label','prediction']])
    input('fim')



if __name__ == "__main__":

    # data = pd.DataFrame({
    #     "index": ['A', 'C', 'D', 'E', 'F'],
    #     "x1": [1, 2, 2, 2, 3],
    #     "x2": [1, 2, 1, 0, 1],
    #     "y": [1, 1, 0, 0, 0]
    # })
    # test_data = pd.DataFrame({
    #     "index": ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    #     "x1": [1, 1, 2, 2, 2, 3, 3],
    #     "x2": [1, 2, 2, 1, 0, 1, 0],
    #     "y": [1, 1, 1, 0, 0, 0, 0]
    # })
    # model = OptimalHyperTreeModel(["x1", "x2"], "y", tree_depth=2, N_min=1, alpha=0.1, solver_name="gurobi")
    # model.train(data, train_method="mio")
    #
    # print(model.predict(test_data))
    # print(type(model.predict(test_data)))
    # input('fim teste')
    # run_example(depth=2)
    file = sys.argv[1]
    print('instance name: ', file)
    run_depth1(file)
