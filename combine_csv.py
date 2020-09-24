import glob
import pandas as pd
extension = 'csv'
all_filenames = [i for i in glob.glob('resultados/teste/*.{}'.format(extension))]
# combine all files in the list
combined_csv = pd.concat([pd.read_csv(f, delimiter=';') for f in all_filenames])

pd.options.display.max_rows = None

# combined_csv.drop(['instance', 'sense', 'diff'], axis=1, inplace=True)

print('REPORT')
print('Total rows', combined_csv.label.count())
print('Total by label')
print(combined_csv.groupby('label').size())
print('Total by cut_type and label')
print(combined_csv.groupby(['cut_type', 'label']).size())
print('Total by iteration, cut_type and label')
print(combined_csv[combined_csv.label == 1].groupby(['relax_iteration', 'cut_type', 'label']).size())
print('Total by iteration and label')
print(combined_csv[combined_csv.label == 1].groupby(['relax_iteration', 'label']).size())

# export to csv
combined_csv.to_csv("combined_csv_teste.csv", sep=';', index=False, encoding='utf-8-sig')
print('exported to csv')
