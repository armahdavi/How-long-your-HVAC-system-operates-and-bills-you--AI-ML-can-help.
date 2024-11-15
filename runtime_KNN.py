# -*- coding: utf-8 -*-
"""
ML Modeling using KNN over runtime data

@author: alima
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')  
exec(open('notion_corrections.py').read())


########################################
### Step 1: Data ETL for ML Modeling ###
########################################

df = pd.read_excel('rt_data_master_ml.xlsx')
## Two dataframes: 1) all features, 2) features other than temperature, RH and their derivatives deleted
X1 = df[['Temp', 'RH', 'Temp_1st_der', 'Temp_2nd_der',
       'RH_1st_der', 'RH_2nd_der']]
X2 = X1.drop(['CO2 In', 'CO2 Out'], axis = 1)
y = df['Mode']


###############################################
### Step 2: Initiating the Cross Validation ###
###############################################

## Instead of train_test_split, data are manualy split due to time-series nature, 25% testing set with 4 fold cross validation
n = len(df)  # Total number of indices
fold_size = n // 4  # Calculate the size of each fold dynamically
split_indices = np.full(n, -1)
split_indices[-fold_size:] = 0                 
split_indices[-2 * fold_size:-fold_size] = 1   
split_indices[-3 * fold_size:-2 * fold_size] = 2 
split_indices[:fold_size] = 3                   
custom_cv = PredefinedSplit(test_fold=split_indices)


##################################################
### Step 3: KNN Model Development and Training ### 
##################################################

scoring = {
    'recall': 'recall',
    'precision': 'precision',
    'f1': 'f1',
    'accuracy': 'accuracy'
}
metrics = scoring.keys()


knn = KNeighborsClassifier()
param_grid = {'n_neighbors' : list(np.arange(1, 11, 1)),
              'weights' : ['uniform', 'distance'],         
              'metric' : ['euclidean', 'minkowski', 'manhattan'],       
              'p' : [1, 2]}  


### Model #1: With X1 fitting (all features)
grid_search1 = GridSearchCV(estimator = knn, param_grid = param_grid, cv = custom_cv, scoring = scoring, refit = 'recall')
grid_search1.fit(X1, y)
best_index = grid_search1.best_index_

## Retrieving all metric scores
fold_scores = {metric: [grid_search1.cv_results_[f'split{i}_test_{metric}'][best_index] for i in range(4)] for metric in metrics}
mean_scores = {metric: grid_search1.cv_results_[f'mean_test_{metric}'][best_index] for metric in metrics}
std_scores = {metric: grid_search1.cv_results_[f'std_test_{metric}'][best_index] for metric in metrics}

score_results1 = {
    metric: {
        'fold_scores': [grid_search1.cv_results_[f'split{i}_test_{metric}'][best_index] for i in range(4)],
        'mean_score': grid_search1.cv_results_[f'mean_test_{metric}'][best_index],
        'std_score': grid_search1.cv_results_[f'std_test_{metric}'][best_index]
    }
    for metric in metrics
}

## Printing results
print("Score results:")
for metric in metrics:
    print(f'{metric.capitalize()} scores for all folds for Model 1:', fold_scores[metric])
    print(f'Best {metric} for Model 1: {mean_scores[metric]:.4f} ± {std_scores[metric]:.4f}')

print('Best parameters for Model 1:', grid_search1.best_params_)



### Model #3: With X2 fitting (only T, RH and their derivatives)
grid_search2 = GridSearchCV(estimator = knn, param_grid = param_grid, cv = custom_cv, scoring = scoring, refit = 'recall')
grid_search2.fit(X2, y)
best_index = grid_search1.best_index_

## Retrieving all metric scores
fold_scores = {metric: [grid_search2.cv_results_[f'split{i}_test_{metric}'][best_index] for i in range(4)] for metric in metrics}
mean_scores = {metric: grid_search2.cv_results_[f'mean_test_{metric}'][best_index] for metric in metrics}
std_scores = {metric: grid_search2.cv_results_[f'std_test_{metric}'][best_index] for metric in metrics}

score_results2 = {
    metric: {
        'fold_scores': [grid_search2.cv_results_[f'split{i}_test_{metric}'][best_index] for i in range(4)],
        'mean_score': grid_search2.cv_results_[f'mean_test_{metric}'][best_index],
        'std_score': grid_search2.cv_results_[f'std_test_{metric}'][best_index]
    }
    for metric in metrics
}

## Printing results
print("Score results:")
for metric in metrics:
    print(f'{metric.capitalize()} scores for all folds for Model 2:', fold_scores[metric])
    print(f'Best {metric} for Model 2: {mean_scores[metric]:.4f} ± {std_scores[metric]:.4f}')

print('Best parameters for Model 2:', grid_search2.best_params_)
