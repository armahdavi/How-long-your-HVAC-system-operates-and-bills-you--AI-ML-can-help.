# -*- coding: utf-8 -*-
"""
Programt to extract features from the study residence for runtime prediction using Machine Learning
The features intially selected are:
    1) Timestamp (will be excluded for ML training as derivatives will link observations)
    2) Visit (proxy for week # of the study)
    3) Temperature (and its 1st and 2nd derivative)
    4) RH (and its 1st and 2nd derivatives)
    5) Indoor and outdoor CO2
    6) System runtime mode (off, fan only, compressor)

@author: alima
"""

import pandas as pd
import os
import glob
import numpy as np

exec(open(r'C:\Life\5- Career & Business Development\Learning\Python Practice\Generic Codes\notion_corrections.py').read())
exec(open(r'C:\Life\5- Career & Business Development\Learning\Python Practice\Generic Codes\sensor_master_generic_simple.py').read())

###############################################
### Step 1: Reading Temperature and RH data ###
###############################################
path = backslash_correct(r'C:\Life\5- Career & Business Development\Learning\Python Practice\Stata_Python_Booster\PhD - QFF\Raw\ambient\\')
exp_path = backslash_correct(r'C:\Life\5- Career & Business Development\Learning\Python Practice\Stata_Python_Booster\PhD - QFF\Processed\ambient\\')
os.chdir(path)

# i = 1
df_overall = pd.DataFrame([])
for file in glob.glob('*.csv'):
    df = hobo_u_read_in(path, exp_path, file, file)
    df_overall = pd.concat([df_overall, df], axis = 0)
        

df_overall['Time'] = pd.to_datetime(df_overall['Time'])
df_overall.sort_values('Time', inplace = True)

df = pd.read_excel(backslash_correct(r'C:\Life\5- Career & Business Development\Learning\Python Practice\Stata_Python_Booster\PhD - QFF\Processed\runtime_master.xlsx'))
df = df[['Time', 'visit', 'Mode']]

################################
### Step 2: Reading CO2 data ###
################################
path_co2 = backslash_correct(r'C:\Life\5- Career & Business Development\Learning\Python Practice\Stata_Python_Booster\PhD - QFF\Raw\co2\\')
exp_path_co2 = backslash_correct(r'C:\Life\5- Career & Business Development\Learning\Python Practice\Stata_Python_Booster\PhD - QFF\Processed')
df_co2 = hobo_co2_read_in(path_co2, exp_path_co2, 'qff_eval_co2iox_452_180829_am.csv', 'qff_eval_co2iox.xlsx')
df_co2['Time'] = pd.to_datetime(df_co2['Time'])

#################################################
### Step 3: Data Cleaning, merging and export ###
#################################################
df_overall.dropna(subset = ['Temp'],  inplace = True)

df_all = df.merge(df_overall, on = 'Time', how = 'inner')

df_all = pd.merge(df_overall, df_co2, on = 'Time', how = 'inner')
df_all = pd.merge(df_all, df, on = 'Time', how = 'inner')
df_all = df_all.dropna()

##############################################################
### Step 4: Adding derivatives of T and RH as new features ###
##############################################################
for col in ['Temp', 'RH']:
    df_all[col + '_1st_der'] = pd.Series(np.gradient(df_all[col]), df_all.index, name='slope')
    df_all[col + '_2nd_der'] = pd.Series(np.gradient(df_all[col + '_1st_der']), df_all.index, name='slope')

df_all.dropna(subset = ['RH_2nd_der'],  inplace = True)

df_all = df_all[['Time', 'visit', 'Temp', 'RH', 'Temp_1st_der', 'Temp_2nd_der', 'RH_1st_der', 'RH_2nd_der', 'CO2 In', 'CO2 Out', 'Mode']]
df_all.to_excel(backslash_correct(r'C:\Life\5- Career & Business Development\Learning\Python Practice\Stata_Python_Booster\PhD - QFF\Processed\ML\rt_data_master.xlsx'), index = False)
    