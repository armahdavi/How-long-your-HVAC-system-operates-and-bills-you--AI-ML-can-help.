# -*- coding: utf-8 -*-
"""
ML Modeling using Logistics Regression (LR) over runtime data

@author: alima
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')  
exec(open('notion_corrections.py').read())

########################################
### Step 1: Data ETL for ML Modeling ###
########################################

df = pd.read_excel('rt_data_master_ml.xlsx')
## Two dataframes: 1) all features, 2) features other than temperature, RH and their derivatives deleted
X = df[['Temp', 'RH', 'Temp_1st_der', 'Temp_2nd_der',
       'RH_1st_der', 'RH_2nd_der']]

y = df['Mode']


###############################################
### Step 2: Initiating the Cross Validation ###
###############################################

## Instead of train_test_split, data are manualy split due to time-series nature, 
## 25% testing set with 4 fold cross validation
n = len(df)  # Total number of indices
fold_size = n // 4  # Calculate the size of each fold dynamically
split_indices = np.full(n, -1)
split_indices[-fold_size:] = 0                 
split_indices[-2 * fold_size:-fold_size] = 1   
split_indices[-3 * fold_size:-2 * fold_size] = 2 
split_indices[:fold_size] = 3                   
custom_cv = PredefinedSplit(test_fold=split_indices)


#################################################
### Step 3: LR Model Development and Training ### 
#################################################

scoring = {
    'recall': 'recall',
    'precision': 'precision',
    'f1': 'f1',
    'accuracy': 'accuracy'
    }
metrics = scoring.keys()

lr = LogisticRegression()
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],   
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'lbfgs', 'saga'],        
    'class_weight': [None, 'balanced'] 
}

grid_search = GridSearchCV(estimator = lr, param_grid = param_grid, cv = custom_cv, scoring = scoring, refit = 'recall')
grid_search.fit(X, y)
best_index = grid_search.best_index_

## Retrieving all metric scores
fold_scores = {metric: [grid_search.cv_results_[f'split{i}_test_{metric}'][best_index] for i in range(4)] for metric in metrics}
mean_scores = {metric: grid_search.cv_results_[f'mean_test_{metric}'][best_index] for metric in metrics}
std_scores = {metric: grid_search.cv_results_[f'std_test_{metric}'][best_index] for metric in metrics}

score_results = {
    metric: {
        'fold_scores': [grid_search.cv_results_[f'split{i}_test_{metric}'][best_index] for i in range(4)],
        'mean_score': grid_search.cv_results_[f'mean_test_{metric}'][best_index],
        'std_score': grid_search.cv_results_[f'std_test_{metric}'][best_index]
    }
    for metric in metrics
}

## Printing results
print("Score results:")
for metric in metrics:
    print(f'{metric.capitalize()} scores for all folds:', fold_scores[metric])
    print(f'Best {metric}: {mean_scores[metric]:.4f} ± {std_scores[metric]:.4f}')

print('Best parameters:', grid_search.best_params_)


