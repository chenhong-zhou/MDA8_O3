import math
from random import shuffle
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, cross_validate,cross_val_predict
from sklearn.model_selection import RepeatedKFold,KFold
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor

from scipy.io import loadmat,savemat
import scipy.stats as stats
import xgboost as xgb
from xgboost import plot_importance
from xgboost import cv
from scipy.io import savemat
import pickle


mat = loadmat('./all_samples_data_all_info_new_all_sites_repeat_2013_2022.mat')


all_samples_variables = mat['all_samples_variables_all_info']  
all_samples_O3 = mat['all_samples_O3'][0]   
num_valid = mat['num_valid']
traing_day = mat['traing_day']




############用平均值填充缺失的值
for ii in range(23):
    lost_ind = np.where(np.isnan(all_samples_variables[:,ii]))[0]
    np.where(np.isnan(all_samples_variables[:,ii]))[0].shape
    all_samples_variables[lost_ind,ii] = np.nanmean(all_samples_variables[:,ii])


# ### 检测填充情况
for ii in range(23):
    np.where(np.isnan(all_samples_variables[:,ii]))[0].shape   



X = all_samples_variables  
y = all_samples_O3    



#########10 cv    sample-based 10 cv                                     
cv = KFold(n_splits=10, random_state=1, shuffle=True)
train_index_all_10cv = []
test_index_all_10cv = []
for train_index,test_index in cv.split(all_samples_O3):
    train_index_all_10cv.append(train_index)
    test_index_all_10cv.append(test_index)



len(train_index_all_10cv)    #10
len(test_index_all_10cv)    #10


r2_test_all = []
r2_train_all = []
mse_test_all = []
mse_train_all = []
mae_test_all = []
mae_train_all = []
mape_test_all = []
mape_train_all = []

#ii=0
for ii in range(10):
    
    print('ii:  ', ii)
    
    train_index_fold = train_index_all_10cv[ii]
    test_index_fold = test_index_all_10cv[ii]
    len_train_fold = len(train_index_fold)
    len_test_fold = len(test_index_fold)
    
    X_train = all_samples_variables[train_index_fold, :]
    X_test = all_samples_variables[test_index_fold, :]
    
    y_train = all_samples_O3[train_index_fold]
    y_test = all_samples_O3[test_index_fold]
    
    
    print('X_train.shape: ', X_train.shape)
    print('y_train.shape: ', y_train.shape)
    print('X_test.shape: ', X_test.shape)
    print('y_test.shape: ', y_test.shape)
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
    
    other_params = {'learning_rate': 0.1, 'n_estimators': 2000, 'max_depth': 15, 'min_child_weight': 10, 'seed': 0, 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
    model = xgb.XGBRegressor(**other_params) 
    
    model.fit(X_train, y_train)
    
    savepath = './results/'
    
    
    print('n_estimator: ', other_params['n_estimators'])
    print('max_depth: ', other_params['max_depth'])
    print('min_child_weight: ', other_params['min_child_weight'])
    
    
    y_test_predicted = model.predict(X_test)
    y_train_predicted = model.predict(X_train)
    
    
    r2_test = r2_score(y_test, y_test_predicted)
    r2_train = r2_score(y_train, y_train_predicted)
    
    mse_test = mean_squared_error(y_test, y_test_predicted)
    mse_train = mean_squared_error(y_train, y_train_predicted)
    
    mae_test = mean_absolute_error(y_test, y_test_predicted)
    mae_train = mean_absolute_error(y_train, y_train_predicted)
    
    mape_test = mean_absolute_percentage_error(y_test[np.where(y_test!=0)[0]], y_test_predicted[np.where(y_test!=0)[0]])
    mape_train = mean_absolute_percentage_error(y_train[np.where(y_train!=0)[0]], y_train_predicted[np.where(y_train!=0)[0]])
    
    
    filename = savepath + "/save_predicted_10cv_" + str(ii) + "th.mat"
    mdict = {'r2_test': r2_test, 'r2_train': r2_train, 'mse_test': mse_test, 'mse_train': mse_train, 'mae_test': mae_test, 'mae_train': mae_train, 'mape_test': mape_test, 'mape_train': mape_train, 'ii': ii, 'train_index_all_10cv': train_index_all_10cv, 'test_index_all_10cv': test_index_all_10cv, 'y_test':y_test, 'y_test_predicted':y_test_predicted, 'y_train': y_train, 'y_train_predicted': y_train_predicted}
    savemat(filename, mdict)
    
    
    
    r2_test_all.append(r2_test)
    r2_train_all.append(r2_train)
    mse_test_all.append(mse_test)
    mse_train_all.append(mse_train)
    mae_test_all.append(mae_test)
    mae_train_all.append(mae_train)
    mape_test_all.append(mape_test)
    mape_train_all.append(mape_train)



print('train r2 mean: ', np.mean(np.array(r2_train_all)))
print('train r2 std: ', np.std(np.array(r2_train_all)))
print('train mse mean: ', np.mean(np.sqrt(np.array(mse_train_all))))
print('train mse std: ', np.std(np.sqrt(np.array(mse_train_all))))
print('train mae mean: ', np.mean(np.array(mae_train_all)))
print('train mae std: ', np.std(np.array(mae_train_all)))
print('train mape mean: ', np.mean(np.array(mape_train_all)))
print('train mape std: ', np.std(np.array(mape_train_all)))

print('test: ')
print('r2 mean: ', np.mean(np.array(r2_test_all)))
print('r2 std: ', np.std(np.array(r2_test_all)))
print('mse mean: ', np.mean(np.sqrt(np.array(mse_test_all))))
print('mse std: ', np.std(np.sqrt(np.array(mse_test_all))))
print('mae mean: ', np.mean(np.array(mae_test_all)))
print('mae std: ', np.std(np.array(mae_test_all)))
print('mape mean: ', np.mean(np.array(mape_test_all)))
print('mape std: ', np.std(np.array(mape_test_all)))



filename = savepath + "10cv_all_results.mat"
mdict = {'r2_test_all': r2_test_all, 'r2_train_all': r2_train_all, 'mse_test_all': mse_test_all, 'mse_train_all': mse_train_all, 'mae_test_all': mae_test_all, 'mae_train_all': mae_train_all, 'mape_test_all': mape_test_all, 'mape_train_all': mape_train_all, 'train_index_all_10cv': train_index_all_10cv, 'test_index_all_10cv': test_index_all_10cv}
savemat(filename, mdict)








