import pandas as pd
import numpy as np

datasets = pd.read_csv('ad_click_dataset.csv')
datasets = datasets.drop(['id','full_name'],axis=1)
print(datasets.columns)

# Check for missing values
for i in range(len(datasets.columns)):
    print(datasets.columns[i])
    print(datasets[datasets.columns[i]].unique())
    print(datasets[datasets.columns[i]].isnull().sum())

# Fill missing values
for columns in datasets.columns:
    datasets[columns].fillna('missing',inplace=True)
    print(datasets[columns].unique())

datasets.to_csv('ad_click_dataset_cleaned.csv',index=False)
print('Data cleaning done!')