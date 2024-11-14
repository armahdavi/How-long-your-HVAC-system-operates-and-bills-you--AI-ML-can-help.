# -*- coding: utf-8 -*-
"""
Batch processing of CO2 data

@author: alima
"""

import pandas as pd
import os
import glob
import numpy as np

exec(open(r'C:\Career\Learning\Python Practice\Generic Codes\notion_corrections.py').read())
exec(open(r'C:\Career\Learning\Python Practice\Generic Codes\sensor_master_generic_simple.py').read())

os.chdir(backslash_correct('C:\Career\Learning\Python Practice\Stata_Python_Booster\PhD - QFF\Raw\co2'))

df_overall = pd.DataFrame([])
for file in glob.glob('*.csv'):
    df = pd.read_csv(file, skiprows=1)
    df = df.iloc[:,1:4] # df.drop(['#', 'Host Connected (LGR S/N: 10544452)', 'Stopped (LGR S/N: 10544452)', 'End Of File (LGR S/N: 10544452)'], axis = 1, inplace = True)
    
    df_overall = pd.concat([df_overall, df], axis = 0)

df_overall.columns  = ['Time', 'CO2_out', 'CO2_in']
df_overall['Time'] = '20' + df_overall['Time'].str[6:8] + '/' + df_overall['Time'].str[3:5] + '/' + df_overall['Time'].str[0:2] + ' ' + df_overall['Time'].str[-8:-3]
df_overall['Time'] = pd.to_datetime(df_overall['Time'])
