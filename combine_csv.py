import glob
import pandas as pd
import re

all_filenames = [i for i in glob.glob('resultados/cbc/*dataset.csv')]
# combine all files in the list

combined_csv = pd.concat([pd.read_csv(f, delimiter=';').assign(instance=re.compile("\_([0-9])+\_[0-9]\_[0-9]\_dataset\.csv").split(f.split('/')[2])[0]) for f in all_filenames])

pd.options.display.max_rows = None

# combined_csv.drop(['instance', 'sense', 'diff'], axis=1, inplace=True)

print('REPORT')
print('Total rows', combined_csv.label.count())
print('Total by label')
print(combined_csv.groupby('label').size())
print('Total by cut_type and label')
print(combined_csv.groupby(['cut_type', 'label']).size())
print('Total by instance and label')
print(combined_csv.groupby(['instance', 'label']).size())
print('Total by iteration, cut_type and label')
print(combined_csv[combined_csv.label == 1].groupby(['relax_iteration', 'cut_type', 'label']).size())
print('Total by iteration and label')
print(combined_csv[combined_csv.label == 1].groupby(['relax_iteration', 'label']).size())

# export to csv
combined_csv.drop(combined_csv.columns.difference(['cut', 'x_values', 'instance']), 1).to_csv("results_cbc.csv", sep=';', index=False, encoding='utf-8-sig')
combined_new = combined_csv.drop(['cut', 'x_values'], axis=1)
combined_new.to_csv('results_cbc_nocuts.csv', sep=';', index=False, encoding='utf-8-sig')
print('exported to csv')
