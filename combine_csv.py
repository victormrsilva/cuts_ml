import os
import glob
import pandas as pd
extension = 'csv'
all_filenames = [i for i in glob.glob('resultados/test_mir/*.{}'.format(extension))]
#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f, delimiter=';') for f in all_filenames])

pd.options.display.max_rows = None

print('REPORT')
print('Total rows', combined_csv.label.count())
print('Total by label')
print(combined_csv.groupby('label').size())
print('Total by cut_type')
print(combined_csv.groupby('cut_type').size())
print('Total by cut_type and label')
print(combined_csv.groupby(['cut_type','label']).size())
print('Total by instance_name, cut_type and label')
print(combined_csv.groupby(['instance','cut_type', 'label']).size())
print('Total by iteration, cut_type and label')
print(combined_csv.groupby(['relax_iteration', 'cut_type', 'label']).size())


#export to csv
combined_csv.to_csv("combined_csv_mir.csv", sep=';', index=False, encoding='utf-8-sig')
print('exported to csv')
