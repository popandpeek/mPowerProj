# Benjamin Pittman
# University of Washington-Bothell
# CS 581 Autumn 2017
# 
# This file will merge two .csv's containing
# features extracted from mPower data set
#

import pandas as pd
import numpy as np

main_data = pd.read_csv('mPowerFeatrues_outbound_1203_reduced.csv', low_memory=False)
join_data = pd.read_csv('mPowerFeatrues_rest_1203_reduced.csv', low_memory=False)

result = pd.merge(main_data, join_data, how='inner', on=['recordId', 'healthCode'])

# drop any rows with NaN values
result.dropna(inplace=True)
result.reset_index(inplace=True)

# drop rows with an iphone 5 due to different technology
booleans2 = []
for info in result['phoneInfo']:
    if '5' in info:
        booleans2.append(False)
    else:
        booleans2.append(True)

phone_ok = pd.Series(booleans2)

final_reduced_data_set = result[phone_ok]
final_reduced_data_set.reset_index(inplace=True)
final_reduced_data_set.drop(['Unnamed: 0_x', 'idx_x', 'Unnamed: 0_y', 'idx_y', 'level_0', 'index'], axis=1, inplace=True)
final_reduced_data_set.to_csv('final_data.csv')
print(final_reduced_data_set.shape)
