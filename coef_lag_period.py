# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 16:55:41 2018

@author: fabdellah
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 14:26:04 2018

@author: fabdellah
"""


# marche bien pour coef+lag+period pour une annee


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
df1 = pd.read_excel('daily_oil.xls')
df1.columns = ['date', 'price']
df1 = df1.dropna(how='any')
df1 = df1[df1.applymap(np.isreal).any(1)]
df1 = df1.reset_index()
df1.drop('index', axis=1, inplace=True)




def MA(date_d, df, lag:int, period): 
    start_date = datetime.strptime(date_d,"%Y-%m-%d %H:%M:%S") - relativedelta(months=int(lag)) -  relativedelta(months=int(period)) 
    end_date = start_date + relativedelta(months=int(period)) 
    df['date'] = pd.to_datetime(df['date'])  
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    df_avg = df.loc[mask].reset_index()  
    df_avg.drop('index', axis=1, inplace=True)
    return df_avg['price'].mean()


def MA_vect(df,i, lag:int, period): 
    nbr_months = 3 # = reset_period
    df = df.reset_index()  
    df.drop('index', axis=1, inplace=True)
    start_date = datetime.strptime('2016-01-04 00:00:00',"%Y-%m-%d %H:%M:%S")+ relativedelta(months=i)  
    k=0
    ma_vect =  np.zeros(nbr_months)   
    for single_date in pd.date_range(start_date, periods=nbr_months, freq='BM'):
        ma_vect[k] = MA(str(single_date),df,lag,period )
        k = k+1
    return ma_vect


def compute_mse(y, x, coef):
    """compute the loss by mse."""
    e = y - x*coef
    mse = e.dot(e) / (2 * len(y))
    return mse

def func(parameters, *data):
    #lag = 1
    #period = 3
    #coef = 1
    coef,lag, period = parameters
    x, y, i = data
    result = compute_mse(y['USD.20'], MA_vect(x,i,lag,period), coef)
    return result


if __name__ == '__main__':

    #producing "experimental" data 
    x = df1[210:1091]                   #daily from 2014 to 2017,6months
    x = x.reset_index()  
    x.drop('index', axis=1, inplace=True)
   
    y = russia_Index[225:243]           #monthly from 2016 to 2017,6months
    y = y.reset_index()  
    y.drop('index', axis=1, inplace=True)
    y.drop('level_0', axis=1, inplace=True)
    
    max_period = 18  #it should be 24
    max_lag = 12
    reset_period = 3
    nbr_parameters = 3
    
    rranges = (slice(0,0.3,0.1),slice(1, max_lag,1),slice(1, max_period,1))
    nbr_reset_periods = len(y)//reset_period
    resbrute = np.zeros((nbr_reset_periods , nbr_parameters ))
    

    for i in range(0,len(y),reset_period):
           
        print('------------------')
        print('Reset period:', i )
        start_date = datetime.strptime('2016-01-04 00:00:00',"%Y-%m-%d %H:%M:%S") + relativedelta(months=i) 
        start_date_x = start_date - relativedelta(months=max_lag) - relativedelta(months=max_period) 
        
        end_date = start_date + relativedelta(months=reset_period) 
        
        x['date'] = pd.to_datetime(x['date'])  
        mask = (x['date'] >= start_date_x) & (x['date'] <= end_date)
        df_x = x.loc[mask].reset_index()  
        df_x.drop('index', axis=1, inplace=True)
            
        #packing "experimental" data into args
        args = (df_x,y[i:i+reset_period],i)
        res = optimize.brute(func, rranges, args=args, full_output=True,     # erreur ici
                                  finish=None)
        #resbrute[i] = res[0]
        
        #print('resultat:' , res[0])
        print('coef:',res[0][0], 'lag:' , res[0][1], 'period:', res[0][2])  # global minimum   
        print('minimum value:', res[1])

        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
