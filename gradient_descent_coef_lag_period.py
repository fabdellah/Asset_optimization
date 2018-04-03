
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 14:28:17 2018

@author: fabdellah
"""
"""Problem Sheet 2.

Gradient Descent
"""


import matplotlib.pyplot as plt


import datetime
from helpers import *

import numpy as np
import pandas as pd
from datetime import timedelta, date, datetime
from scipy.optimize import differential_evolution
from dateutil.relativedelta import relativedelta
from scipy import optimize



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
    start_date = datetime.strptime('2016-01-04 00:00:00',"%Y-%m-%d %H:%M:%S")+ relativedelta(months=i) 
    k=0
    ma_vect =  np.zeros(nbr_months)   
    for single_date in pd.date_range(start_date, periods=nbr_months, freq='BM'):
        ma_vect[k] = MA(str(single_date),df,lag,period,np.empty(1) )
        k = k+1
    return ma_vect




def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


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
        #print("Gradient Descent({bi}/{ti}): loss={l}, w={w}".format(
        #     bi=n_iter, ti=max_iters - 1, l=loss, w=w))
    return loss, w










if __name__ == '__main__':
    
    import warnings
    warnings.filterwarnings("ignore")
    
    xx = df1[210:1091]                   #daily from 2014 to 2017,6months
    xx = xx.reset_index()  
    xx.drop('index', axis=1, inplace=True)
   
    yy = russia_Index[225:243]           #monthly from 2016 to 2017,6months
    yy = yy.reset_index()  
    yy.drop('index', axis=1, inplace=True)
    yy.drop('level_0', axis=1, inplace=True)
    
    x, mean_x, std_x = standardize(xx['oil'])
    xx['oil'] = x
    #y, tx = build_model_data(xx, yy)

    
    max_period = 24
    max_lag = 12
    reset_period = 3
    nbr_parameters = 3
    nbr_reset_periods = len(yy)//reset_period
   
    
    #coef_range = range(0,0.15,0.05)
    lag_range = range(1, max_lag,1)
    period_range = range(1, max_period,1)
    reset_range = range(0,len(yy),reset_period)
    # Define the parameters of the algorithm.
    max_iters = 100
    gamma = 0.7
    vect_coef = np.zeros(nbr_reset_periods)
    j = 0
    
    for i in reset_range:
        print('*****************************')
        print('Reset period:', i )
        
        start_date = datetime.strptime('2016-01-31 00:00:00',"%Y-%m-%d %H:%M:%S") + relativedelta(months=i) 
        end_date = start_date + relativedelta(months=reset_period) 
        loss = np.zeros((max_period-1,max_lag-1))
        coef = np.zeros((max_period-1,max_lag-1))
        
        for period in period_range:
            #print('----------------------')
            #print('Period:', period)
            for lag in lag_range:
                #print('lag', lag)
                start_date_x = start_date - relativedelta(months=lag) - relativedelta(months=period) 
                
                xx['date'] = pd.to_datetime(xx['date'])  
                mask = (xx['date'] >= start_date_x) & (xx['date'] <= end_date)
                df_x = xx.loc[mask].reset_index()  
                df_x.drop('index', axis=1, inplace=True)
                
                X_array = MA_vect(df_x,i,lag,period,np.empty(1))               
                X = pd.DataFrame(X_array, columns=['oil']) 
                y = yy[i:i+reset_period]['USD.20']
                        
                # Initialization
                w_initial = np.array([0.06])
                

                gradient_loss, gradient_w = gradient_descent(y, X_array, w_initial, max_iters, gamma)
                #print('loss', gradient_loss, 'coef', gradient_w)
                loss[period-1,lag-1] = gradient_loss
                coef[period-1,lag-1] = gradient_w
                
        opt_period, opt_lag = np.unravel_index(loss.argmin(), loss.shape)
        print('optimal period:',opt_period+1,', optimal lag:', opt_lag+1)
        print('optimal coef:', coef[opt_period,opt_lag])   
        print('loss:', loss[opt_period,opt_lag])   
        vect_coef[j] = coef[opt_period,opt_lag]
        j= j+1











