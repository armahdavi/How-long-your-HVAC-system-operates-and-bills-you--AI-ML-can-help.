# -*- coding: utf-8 -*-
"""
Feature engineering over runtime and IEQ features collected in QFF Evaluation study

The program includes:
    1) ETL to read data for the first for weeks (when HVAC ran normally)
    2) Cleaning and encoding runtime
    3) Correlational Matrix and KDE graphs per binary classification of Compressor ON and OFF
    
@author: alima
"""

import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')  
exec(open('notion_corrections.py').read())

########################################
### Step 1: Data ETL for ML Modeling ###
########################################
df = pd.read_excel('rt_data_master.xlsx')
df = df[df['visit'] < 5] 
## The last two visits (5 and 6 are excluded as representing unusual cases: 
## HVAC was intentionally running continously with fan only mode to collect more dust for QFF)
print(len(df))

df['Mode'].unique() # Correction to y by filling transient with ffil
df.replace('Transient', np.nan, inplace = True)

with warnings.catch_warnings(record=True) as w:
    df.fillna(method = 'ffill', inplace = True)

df['Mode'] = np.where(df['Mode'] == 'Compressor', 1, 0)

X = df[['Temp', 'RH', 'Temp_1st_der', 'Temp_2nd_der',
        'RH_1st_der', 'RH_2nd_der', 'CO2 In', 'CO2 Out']]


####################################
### Step 2: Correlational Matrix ###
####################################
correlation_matrix = X.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm', 
            square = True, fmt = ".2f", cbar_kws = {"shrink": .8})
plt.title('Correlation Matrix of HVAC Runtime Features', fontsize = 14)
plt.show()


####################################################################
### Step 3: Scatter Matrix and Kernel Density Estimations (KDEs) ###
####################################################################
color_map = df['Mode'].map({0: 'blue', 1: 'red'})
scatter_matrix(X, alpha=0.8, figsize=(12, 12), diagonal='kde', marker='o', c=color_map)
plt.suptitle('Scatter Matrix of Features', fontsize = 20)
plt.show()

for col in X.columns:
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df[df['Mode'] == 0][col], color='blue', fill=True, alpha=0.3, label = 'Compressor OFF')
    sns.kdeplot(df[df['Mode'] == 1][col], color='red', fill=True, alpha=0.3, label = 'Compressor ON')
    
    plt.title(f'Feature: {col}', fontsize = 14)
    plt.xlabel("Feature Values",  fontsize = 14)  
    plt.ylabel("Density", fontsize = 14)

    plt.legend()
    plt.show()

## Save the features and target for later ML modeling
pd.concat([X, df['Mode']], axis = 1).to_excel('rt_data_master_ml.xlsx')
