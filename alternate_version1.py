# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 14:24:45 2018

@author: fabdellah
"""


import matplotlib.pyplot as plt
import datetime
#from helpers import *
import numpy as np
import pandas as pd
from datetime import timedelta, date, datetime
from scipy.optimize import differential_evolution
from dateutil.relativedelta import relativedelta
from scipy import optimize
from numpy import linalg as LA
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import RidgeCV

# Import data
file = 'External_Data.xls'
df = pd.read_excel(file)
#print(df.columns)
df_subset = df.dropna(how='any')
row = df_subset.shape[1]
col = df_subset.shape[0]
df_subset = df_subset[3:col]

russia_Index = df_subset[['Data Type','USD.20']].reset_index()[2:df_subset.shape[0]]
russia_Index.drop('index', axis=1, inplace=True)
russia_Index = russia_Index.reset_index()

df = pd.read_excel('daily_oil.xls')
df.columns = ['date', 'oil', 'power', 'coal', 'gas']
df = df.dropna(how='any')
df = df[df.applymap(np.isreal).any(1)]
df = df.reset_index()
df.drop('index', axis=1, inplace=True)
df1 = df[['date', 'oil']]
df1 = df1.dropna(how='any')
df1 = df1[df1.applymap(np.isreal).any(1)]
df1 = df1.reset_index()
df1.drop('index', axis=1, inplace=True)







def MA(date_d, df, lag:int, period,df_avg): 
    start_date = datetime.strptime(date_d,"%Y-%m-%d %H:%M:%S") - relativedelta(months=int(lag)) -  relativedelta(months=int(period)) 
    end_date = start_date + relativedelta(months=int(period)) 
    df['date'] = pd.to_datetime(df['date'])  
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    df_avg = df.loc[mask].reset_index()  
    df_avg.drop('index', axis=1, inplace=True)
    return df_avg['oil'].mean()


def MA_vect(df,i, lag:int, period, ma_vect): 
    nbr_months = 3 # = reset_period 
    df = df.reset_index()  
    df.drop('index', axis=1, inplace=True)
    start_date = datetime.strptime('2016-01-31 00:00:00',"%Y-%m-%d %H:%M:%S")+ relativedelta(months=i) 
    k=0
    ma_vect =  np.zeros(nbr_months)   
    for single_date in pd.date_range(start_date, periods=nbr_months, freq='BM'):
        ma_vect[k] = MA(str(single_date),df,lag,period,np.empty(1) )
        k = k+1
    return ma_vect


def MA_vectorized(df, vect_lag, vect_period, ma_vectorized):
    reset_period = 3
    reset_range = range(0,len(vect_lag),reset_period)
    #j = 0
    for i in reset_range:   
        for k in range(3):
            ma_vectorized[i+k] = MA_vect(df, i, vect_lag[i], vect_period[i], np.empty(3))[k]
        #j = j+1
    return ma_vectorized    
        
        

def func_lag_period(parameters, *data):
    
    lag, period = parameters
    x, y, i, coef, values = data
    values = compute_mse(y, MA_vect(x,i,lag,period,np.empty(1)), coef)
    return values

        


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def compute_mse(y, x, coef):
    """compute the loss by mse."""
    e = y - x*coef
    mse = e.dot(e) / (2 * len(y))
    return mse


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx*w
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # gradient w by descent update
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        perc_err = LA.norm(err)/LA.norm(y)
        #print("Gradient Descent({bi}/{ti}): loss={l}, w={w}".format(
        #     bi=n_iter, ti=max_iters - 1, l=loss, w=w))
    return loss, w, perc_err




def ridge_regression(X_train,y_train, X_test, y_test):    
    """Ridge regression algorithm."""
    # select the best alpha with RidgeCV (cross-validation)
    # alpha=0 is equivalent to linear regression
    alpha_range = 10.**np.arange(-2, 3)
    ridgeregcv = RidgeCV(alphas=alpha_range, normalize=False, scoring='mean_squared_error') 
    ridgeregcv.fit(X_train, y_train)
    #print('best alpha=',ridgeregcv.alpha_)
    #print('ridgeregcv.coef: ',ridgeregcv.coef_)
    # predict method uses the best alpha value
    y_pred = ridgeregcv.predict(X_test)
    #return (np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    err = metrics.mean_squared_error(y_test, y_pred)
    return ridgeregcv.coef_, err
    




if __name__ == '__main__':
              
    import warnings
    warnings.filterwarnings("ignore")
    
    xx = df1[210:966]                   #daily from 2014 to 2017
    xx = xx.reset_index()  
    xx.drop('index', axis=1, inplace=True)
       
    yy = russia_Index[225:237]           #monthly from 2016 to 2017
    yy = yy.reset_index()  
    yy.drop('index', axis=1, inplace=True)
    yy.drop('level_0', axis=1, inplace=True)
    y = yy['USD.20']
    #x, mean_x, std_x = standardize(xx['oil'])
    #xx['oil'] = x

    max_period = 24
    max_lag = 12
    reset_period = 3
    nbr_parameters = 2
    nbr_reset_periods = len(yy)//reset_period
    reset_range = range(0,len(yy),reset_period)
    opt_lag_period = np.zeros((nbr_reset_periods,nbr_parameters))
    j = 0
    bounds = [(1, max_lag), (1, max_period)]
    
    max_iters = 100
    gamma = 0.7
    nbr_iterations = 10
    init_coef = 0.006
    init_lag = 11
    init_period = 24
    
    coef = np.array([init_coef])
    vect_lag = init_lag*np.ones(len(yy))
    vect_period = init_period*np.ones(len(yy))
    vect_loss = np.zeros(nbr_iterations)
    vect_coef = np.zeros(nbr_iterations)
    
    
    from time import time
    t0 = time()
    
    
    for itera in range(nbr_iterations):
        
        print('///////////////////////////////////////')
        print('Iteration: ', itera )
        
        
        j = 0
        for i in reset_range:
            #print('*****************************')
            #print('Reset period:', i )
            
            start_date = datetime.strptime('2016-01-31 00:00:00',"%Y-%m-%d %H:%M:%S") + relativedelta(months=i) 
            start_date_x = start_date - relativedelta(months=max_lag) - relativedelta(months=max_period)             
            end_date = start_date + relativedelta(months=reset_period) 
            
            xx['date'] = pd.to_datetime(xx['date'])  
            mask = (xx['date'] >= start_date_x) & (xx['date'] <= end_date)
            df_x = xx.loc[mask].reset_index()  
            df_x.drop('index', axis=1, inplace=True)
            y3 = yy[i:i+reset_period]['USD.20']
            
            args = (df_x , y3, i, coef, np.empty(1))
        
            result = differential_evolution(func_lag_period, bounds, args=args)
            
            opt_lag_period[j] = (round(result.x[0]),round(result.x[1]))
            for k in range(3):
                vect_lag[i+k] = round(result.x[0])
                vect_period[i+k] = round(result.x[1])
            j = j+1
            
            
        print('vect lag: ', vect_lag)
        
        X_array = MA_vectorized(xx, vect_lag, vect_period, np.empty(len(yy)))      
        X = pd.DataFrame(X_array, columns=['oil']) #sert a rien
                  
        w_initial = coef

        gradient_loss, gradient_w, perc_err = gradient_descent(standardize(y), standardize(X_array), w_initial, max_iters, gamma)
        
        print('loss', gradient_loss)
        print('perc err', perc_err)
        vect_loss[itera] = gradient_loss
        vect_coef[itera] = gradient_w

        # update coef
        coef = gradient_w
        
       
        X_train, X_test, y_train, y_test = train_test_split(standardize(X_array.reshape(12,1)), standardize(y), random_state=1)
        
        res_ridge = ridge_regression(X_train,y_train, X_test, y_test)
        y_pred_GD = coef*X_test
        print('Coef GD:', coef ,'Error GD:', metrics.mean_squared_error(y_test, y_pred_GD))
        print('Coef RR:', res_ridge[0] ,'Error RR', res_ridge[1])
        
    t1 = time();
    d1 = t1 - t0
    print ("Duration in Seconds %6.3f" % d1)          
      
   
        
        
           
            






























































