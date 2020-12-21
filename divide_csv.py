from math import ceil

import pandas as pd
import numpy as np

csv = pd.read_csv('results_cbc_nocuts.csv', delimiter=';')

index1 = csv[csv['label'] == 1]
print(index1)
index0 = csv[csv['label'] == 0]

np.random.seed(0)
sizes1 = [0.25, 0.5, 0.75, 1]
sizes0 = [1, 2, 3, 4, 5]
for size1 in sizes1:
    for size0 in sizes0:
        for i in range(5):
            indexes1 = np.random.random_integers(0, index1.shape[0] - 1, size=(ceil(int(index1.shape[0]) * size1),))
            indexes1 = index1.iloc[indexes1, ].index.tolist()
            indexes = np.random.random_integers(0, index0.shape[0]-1, size=(ceil(int(index1.shape[0]) * size1)* size0, ))
            indexes = index0.iloc[indexes,].index.tolist()
            indexes = np.append(indexes, indexes1)
            ds = csv.loc[indexes, ].reset_index()
            ds.drop(['instance'], axis=1).to_csv("small_size1{}_size0{}_iter{}.csv".format(size0, size1, i),sep=';', index=False,encoding='utf-8-sig')

