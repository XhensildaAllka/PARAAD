# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 11:46:59 2023

@author: Usuario
"""

import pandas as pd
import numpy as np
import scipy.stats
from sklearn import metrics

def perturbate(X_test, sigma_Sens, percentage_outlier, seed_d, seed_sh):
    pert_test_std = X_test.copy()
    
    number = round(pert_test_std.shape[0]*percentage_outlier)
    
    pert_test_indx =[]
    for sensor in range(X_test.shape[1]):
        # sensor1
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
#%% PRINT THE RESULTS OF CONFUSION MATRIX

def conf_matr_Resuls(tru_val, pred_val):
    confusion_matrix_det_valid = metrics.confusion_matrix(tru_val, pred_val)
    if confusion_matrix_det_valid.shape == (2, 2):
        FN = confusion_matrix_det_valid[1,0]
        FP = confusion_matrix_det_valid[0,1]
        TP = confusion_matrix_det_valid[1,1]
        TN = confusion_matrix_det_valid[0,0]
        # Sensitivity, hit rate, recall, or true positive rate
        print(confusion_matrix_det_valid)

        Tpr = TP/(TP+FN)
        print('Tpr value is:',Tpr)
        # Specificity or true negative rate
        TNR = TN/(TN+FP) 
        print('TNR value is:',TNR)
        # precision or positive predictive value
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        # print('Recall value is:',recall)
        print('precision value is:',precision)
        # Negative predictive value
        NPV = TN/(TN+FN)
        # Fall out or false positive rate
        Fpr = FP/(FP+TN)
        print('Fpr value is:',Fpr)
        # False negative rate
        FNR = FN/(TP+FN)
        # False discovery rate
        FDR = FP/(TP+FP)
        # Overall accuracy
        ACC = (TP+TN)/(TP+FP+FN+TN)
        # False negative rate
        FNR = FN/(TP+FN)
        # False discovery rate
        FDR = FP/(TP+FP)
        # Overall accuracy
        ACC = (TP+TN)/(TP+FP+FN+TN)
        print('Accuracy  value is:',ACC)
        F1 = 2*(precision*recall)/(precision+recall)
        print('F1  value is:',F1)
    else:
        Tpr = 1
        TNR = 1
        precision = 1
        Fpr = 0
        ACC = 1
        F1 = 1
    return Tpr, TNR, precision, Fpr, ACC, F1

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
#%% PCA

from sklearn.decomposition import PCA

def reconstructPCA(data_train, data_test):
    num_samples_train, n_sensors, n_hours = data_train.shape
    num_samples_test = data_test.shape[0]
    data_2D_train = data_train.reshape(num_samples_train, n_sensors * n_hours)
    data_2D_test = data_test.reshape(num_samples_test, n_sensors * n_hours)
    
    pca = PCA(n_components=n_sensors) 
    pca.fit(data_2D_train)
    data_pca = pca.transform(data_2D_test)
    data_reconstructed2D = pca.inverse_transform(data_pca)
    data_reconstructed = data_reconstructed2D.reshape(num_samples_test, n_sensors, n_hours)
    return data_reconstructed

#%%

def reconstructIndependentlyPCA(X_train, X_test):
    # Reshape the data into 2D: (None, n_cities * temp_in_aDay)
    num_samples_train, n_cities, temp_in_aDay = X_train.shape
    num_samples_test = X_test.shape[0]

    X_train_2D = X_train.reshape(num_samples_train, n_cities * temp_in_aDay)
    X_test_2D = X_test.reshape(num_samples_test, n_cities * temp_in_aDay)

    # Reconstructed data storage
    reconstructed_data = np.zeros_like(X_test_2D)

    # Apply PCA to each time step independently
    for i in range(temp_in_aDay):
        pca = PCA(n_components=5)
        pca.fit(X_train_2D[:, i*n_cities:(i+1)*n_cities])
        X_test_pca = pca.transform(X_test_2D[:, i*n_cities:(i+1)*n_cities])
        X_test_reconstructed = pca.inverse_transform(X_test_pca)
        reconstructed_data[:, i*n_cities:(i+1)*n_cities] = X_test_reconstructed

    # Reshape the reconstructed data back to 3D: (num_samples_test, n_cities, temp_in_aDay)
    X_test_reconstructed = reconstructed_data.reshape(num_samples_test, n_cities, temp_in_aDay)

    return X_test_reconstructed

#%%
from scipy.interpolate import Rbf

def reconstructRBF(X_train, X_test):
    # Reshape the data into 2D: (None, n_cities * temp_in_aDay)
    num_samples_train, n_cities, temp_in_aDay = X_train.shape
    num_samples_test = X_test.shape[0]

    X_train_2D = X_train.reshape(num_samples_train, n_cities * temp_in_aDay)
    X_test_2D = X_test.reshape(num_samples_test, n_cities * temp_in_aDay)

    # Create the meshgrid for the coordinates
    x_coords, y_coords = np.meshgrid(np.arange(n_cities), np.arange(temp_in_aDay), indexing='ij')

    # Reconstructed data storage
    reconstructed_data = np.zeros_like(X_test_2D)

    # Apply RBF interpolation for each sample
    for i in range(num_samples_test):
        rbf = Rbf(x_coords.flatten(), y_coords.flatten(), X_train_2D[i])
        reconstructed_data[i] = rbf(x_coords.flatten(), y_coords.flatten())

    # Reshape the reconstructed data back to 3D: (num_samples_test, n_cities, temp_in_aDay)
    X_test_reconstructed = reconstructed_data.reshape(num_samples_test, n_cities, temp_in_aDay)

    return X_test_reconstructed

#%%
# from pykrige.ok import OrdinaryKriging

# def reconstruct_signal_with_kriging(X_train, X_test):
#     # Reshape the data into 2D: (None, n_cities * temp_in_aDay)
#     num_samples_train, n_cities, temp_in_aDay = X_train.shape
#     num_samples_test = X_test.shape[0]

#     X_train_2D = X_train.reshape(num_samples_train, n_cities * temp_in_aDay)
#     X_test_2D = X_test.reshape(num_samples_test, n_cities * temp_in_aDay)

#     # Create the meshgrid for the coordinates
#     x_coords, y_coords = np.meshgrid(np.arange(n_cities), np.arange(temp_in_aDay), indexing='ij')

#     # Reconstructed data storage
#     reconstructed_data = np.zeros_like(X_test_2D)

#     # Apply Kriging interpolation for each sample
#     for i in range(num_samples_test):
#         kriging = OrdinaryKriging(x_coords.flatten(), y_coords.flatten(), X_train_2D[i])
#         reconstructed_data[i] = kriging.execute('grid', x_coords.flatten(), y_coords.flatten()).data

#     # Reshape the reconstructed data back to 3D: (num_samples_test, n_cities, temp_in_aDay)
#     X_test_reconstructed = reconstructed_data.reshape(num_samples_test, n_cities, temp_in_aDay)

#     return X_test_reconstructed