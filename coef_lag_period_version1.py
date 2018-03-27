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
print(df.columns)
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
x = df1



def MA(date_d, df, lag:int, period): 
    start_date = datetime.strptime(date_d,"%Y-%m-%d %H:%M:%S") - relativedelta(months=int(lag)) -  relativedelta(months=int(period)) 
    end_date = start_date + relativedelta(months=int(period)) 
    df['date'] = pd.to_datetime(df['date'])  
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    df_avg = df.loc[mask].reset_index()  
    df_avg.drop('index', axis=1, inplace=True)
    return df_avg['price'].mean()


def MA_year(df, lag:int, period): 
    nbr_months = 12
    df = df.reset_index()  
    df.drop('index', axis=1, inplace=True)
    start_date = datetime.strptime('2016-01-04 00:00:00',"%Y-%m-%d %H:%M:%S") - relativedelta(months=int(period)) 
    i=0
    ma_year =  np.zeros(nbr_months)   
    for single_date in pd.date_range(start_date, periods=12, freq='BM'):
        ma_year[i] = MA(str(single_date),df,lag,period )
        i = i+1
    return ma_year


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
    x, y = data
    result = compute_mse(y['USD.20'], MA_year(x,lag,period), coef)
    return result


if __name__ == '__main__':
    max_period = 24
    max_lag = 12
    
    #producing "experimental" data 
    x = df1[462:966]
    y = russia_Index[228:240]
    
    #packing "experimental" data into args
    args = (x,y)
    
    rranges = (slice(-1,1,1),slice(0, max_lag,1),slice(0, max_period))
    resbrute = optimize.brute(func, rranges, args=args, full_output=True,
                             finish=None)
    print('coef:',resbrute[0][0], 'lag:' , resbrute[0][1], 'period:', resbrute[0][2])  # global minimum    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
