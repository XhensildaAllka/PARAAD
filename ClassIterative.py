# -*- coding: utf-8 -*-
"""
@author: xallka
"""

import pandas as pd
import numpy as np
import scipy.stats
from sklearn import metrics

def biasAnomalies(X_test, sigma_Sens, percentage_outlier, seed_d, seed_sh):
    pert_test_std = X_test.copy()
    
    number = round(pert_test_std.shape[0]*percentage_outlier)
    
    pert_test_indx =[]
    for sensor in range(X_test.shape[1]):
        Sens_std_test = pert_test_std[:, sensor, :]
        Sens_std_test = pd.DataFrame(Sens_std_test)
        np.random.seed(seed_d + sensor+1)
        sampled_test = Sens_std_test.sample(number)
        sampled_test_pert = sampled_test.copy()
        
        for i in range(0, len(sampled_test)):
            np.random.seed(seed_sh+i)
            shift = scipy.stats.truncnorm.rvs(0.2,2,0,scale=sigma_Sens[sensor], size = 1)
            sampled_test_pert.iloc[i, :] = sampled_test.iloc[i,:] + shift
        pert_test_std[sampled_test_pert.index, sensor, :] = sampled_test_pert
        
        outlier_test_Sens = pd.DataFrame()
        outlier_test_Sens['day_indx'] = list(range(0, pert_test_std.shape[0]))
        outlier_test_Sens['day_pert_index'] = 0
        outlier_test_Sens.loc[sampled_test_pert.index, 'day_pert_index'] = 1
        pert_test_indx.append(outlier_test_Sens.loc[:, 'day_pert_index'])
        
    pert_test_indx = pd.DataFrame(pert_test_indx).T
    for k in range(0, len(sigma_Sens)):
        if sigma_Sens[k] == 0:
            pert_test_indx.iloc[:, k] = 0  
        
    return pert_test_indx, pert_test_std

#%% CONCATENATE THE DATA
def DataConcatDaily(join_data):
    # Reshape each ref stat in daily meassurements
    data_join_datIndex = join_data.set_index('date')
    data_3D_all = []
    for i in range(data_join_datIndex.shape[1]):
        sensor = data_join_datIndex.iloc[:, i]
        sensor = pd.DataFrame(sensor)
        sensor.columns = [f'Senor{i+1}']
        ref_data_sensor = pd.pivot_table(sensor, index = sensor.index.date, columns = sensor.index.hour, values= f'Senor{i+1}').T
        data_3D_all.append(ref_data_sensor)
    data_3D_all = np.array(data_3D_all)
    # data_3D_all = data_3D_all
    return data_3D_all.transpose(2, 1, 0)
