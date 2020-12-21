import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


csv = pd.read_csv('results_iis_nocuts.csv', delimiter=';')
cut_types = csv.cut_type.unique()
iteration = csv.relax_iteration.unique()
indexes = set()

for cut in cut_types:
	for iter in iteration:
		for label in [0, 1]:
			print('cut ', cut, 'iteration', iter, 'label', label)
			cond_type = csv['cut_type'] == cut
			cond_iter = csv['relax_iteration'] == iter
			cond_label = csv['label'] == label
			test = csv[cond_type & cond_iter & cond_label]
			for pos in range(len(test.index)):
				i = test.index[pos]
				search = (pairwise_distances(test, [test.loc[i]], metric='manhattan') < 1e-4)
				proximos = np.where(search == True)[0]
				if len(proximos) > 1:
					ind = []
					for j in range(len(proximos)):
						indexes.add(test.index[proximos[j]])
						ind.append(test.index[proximos[j]])
					distances = pairwise_distances(test.loc[ind], [test.loc[i]], metric='manhattan').reshape(1, -1)[0].tolist()
					# print(ind, distances, i)
				if pos % 100 == 0:
					print('pos', pos, 'of', len(test.index))
			print('----------------------')
print(indexes)
print(len(indexes))
print(csv.info())
csv_teste = csv.drop(list(indexes))
input(csv_teste.info())
csv_teste.to_csv("results_iis_nocuts_new.csv", sep=';', index=False, encoding='utf-8-sig')
csv = pd.read_csv('results_iis.csv', delimiter=';')
csv_teste = csv.drop(list(indexes))
csv_teste.to_csv("results_iis_new.csv", sep=';', index=False, encoding='utf-8-sig')
