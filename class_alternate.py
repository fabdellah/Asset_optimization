# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 10:34:44 2018

@author: fabdellah
"""


import matplotlib.pyplot as plt
import datetime
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
import warnings
warnings.filterwarnings("ignore")
from time import time
    


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
df2 = df[['date', 'power']]
df3 = df[['date', 'coal']]
df4 = df[['date', 'gas']]


xx = df[210:966]                   #daily from 2014 to 2017
xx = xx.reset_index()  
xx.drop('index', axis=1, inplace=True)
xx1 = df1[210:966]                   #daily from 2014 to 2017
xx1 = xx1.reset_index()  
xx1.drop('index', axis=1, inplace=True)
xx2 = df2[210:966]                   #daily from 2014 to 2017
xx2 = xx2.reset_index()  
xx2.drop('index', axis=1, inplace=True)
xx3 = df3[210:966]                   #daily from 2014 to 2017
xx3 = xx3.reset_index()  
xx3.drop('index', axis=1, inplace=True)
xx4 = df[210:966]                   #daily from 2014 to 2017
xx4 = xx4.reset_index()  
xx4.drop('index', axis=1, inplace=True)
   
yy = russia_Index[225:237]           #monthly from 2016 to 2017
yy = yy.reset_index()  
yy.drop('index', axis=1, inplace=True)
yy.drop('level_0', axis=1, inplace=True)
y = yy['USD.20']
#x, mean_x, std_x = standardize(xx['oil'])
#xx['oil'] = x


#import forward data : daily from begining of 2016 until the end 2017
file = 'forward_curve.xls'
df_forward = pd.read_excel(file)
forward_curves = df_forward[['date','oil','power','coal','gas']].dropna(how='any')
forward_oil = forward_curves[['date', 'oil']]
forward_power = forward_curves[['date', 'power']]
forward_coal  = forward_curves[['date', 'coal']]
forward_gas  = forward_curves[['date', 'gas']]



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



def compute_mse_func(y, x1,x2,x3,x4, coef1,coef2,coef3,coef4):
    """compute the loss by mse."""
    e = y - (x1*coef1 + x2*coef2 + x3*coef3 + x4*coef4)
    mse = e.dot(e) / (2 * len(y))
    return mse


def compute_mse(y, x, coef):
    """compute the loss by mse."""
    e = y - np.dot(x,coef)
    mse = e.dot(e) / (2 * len(y))
    return mse


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - np.dot(tx,w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def predict(tx,coef):
    #tx = df[['oil','power','coal','gas']].values
    return  np.dot(tx,coef)
    

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm for linear regression."""
    # Define parameters to store w and loss
    #ws = [initial_w]
    #losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # gradient w by descent update
        w = w - gamma * grad
        # store w and loss
        #ws.append(w)
        #losses.append(loss)
        #perc_err = LA.norm(err)/LA.norm(y)
        #print("Gradient Descent({bi}/{ti}): loss={l}, w={w}".format(
        #     bi=n_iter, ti=max_iters - 1, l=loss, w=w))
    return loss, w




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
    

def MA_func_vect_daily(start_date_str, end_date_str, xx1,lag,period):
    start_date = datetime.strptime(start_date_str,"%Y-%m-%d %H:%M:%S") 
    end_date = datetime.strptime(end_date_str,"%Y-%m-%d %H:%M:%S") 
    start_date_x = start_date - relativedelta(months=int(lag)) - relativedelta(months=int(period))             
    xx1['date'] = pd.to_datetime(xx1['date'])  
    mask = (xx1['date'] >= start_date_x) & (xx1['date'] <= end_date- relativedelta(months=int(lag)))
    df_x = xx1.loc[mask].reset_index()  
    df_x.drop('index', axis=1, inplace=True)    
    df_x.iloc[:, [1]]= df_x.iloc[:, [1]].astype(float)   
    return pd.rolling_mean(df_x.set_index('date').resample('D'),window=int(period)).dropna(how='any') 
   

    
def MA_func_matrix_daily(start_date_str, end_date_str , df,  four_lags, four_periods):   
    vect1 = MA_func_vect_daily(start_date_str,end_date_str,df[['date', 'oil']], four_lags[0], four_periods[0])
    vect2 = MA_func_vect_daily(start_date_str,end_date_str,df[['date', 'power']],  four_lags[1], four_periods[1])
    vect3 = MA_func_vect_daily(start_date_str,end_date_str,df[['date', 'coal']],  four_lags[2], four_periods[2])
    vect4 = MA_func_vect_daily(start_date_str,end_date_str,df[['date', 'gas']],  four_lags[3], four_periods[3])
    return np.concatenate((vect1,vect2,vect3,vect4),axis=1)


class class_alternate(object):
    
    def __init__(self,start_date_str, nbr_iterations, max_lag, max_period, reset_period,  init_lag,init_period,init_coef1,init_coef2,init_coef3,init_coef4):
            self.max_lag = max_lag
            self.max_period = max_period
            self.reset_period = reset_period
            self.nbr_iterations = nbr_iterations
            self.init_lag = init_lag
            self.init_period = init_period
            self.init_coef1 = init_coef1
            self.init_coef2 = init_coef2
            self.init_coef3 = init_coef3
            self.init_coef4 = init_coef4
                              
            self.nbr_reset_periods = len(yy)//self.reset_period
            self.reset_range = range(0,len(yy),self.reset_period) 
            self.bounds = [(1, self.max_lag),(1, self.max_lag),(1, self.max_lag),(1,self.max_lag), (1, self.max_period), (1, self.max_period), (1, self.max_period), (1, self.max_period)]
               
            self.start_date_str = start_date_str  #start_date_str = '2016-01-31 00:00:00'
            
     
    def MA(self,date_d, df, lag:int, period,df_avg): 
        start_date = datetime.strptime(date_d,"%Y-%m-%d %H:%M:%S") - relativedelta(months=round(lag)) -  relativedelta(months=round(period)) 
        end_date = start_date + relativedelta(months=round(period)) 
        df['date'] = pd.to_datetime(df['date'])  
        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        df_avg = df.loc[mask].reset_index()  
        df_avg.drop('index', axis=1, inplace=True)
        return df_avg.iloc[:, [1]].mean()
    
    
    def MA_vect(self,df,i, lag:int, period, ma_vect): 
        start_date = datetime.strptime('2016-01-31 00:00:00',"%Y-%m-%d %H:%M:%S")+ relativedelta(months=i) 
        k=0
        for single_date in pd.date_range(start_date, periods=self.reset_period, freq='BM'):
            ma_vect[k] = self.MA(str(single_date),df,lag,period,np.empty(1) )
            k = k+1
        return ma_vect
    
    
    def MA_func_vect(self,xx1,i,lag,period,ma_vect):
        start_date = datetime.strptime(self.start_date_str,"%Y-%m-%d %H:%M:%S") + relativedelta(months=i) 
        start_date_x = start_date - relativedelta(months=int(round(lag))) - relativedelta(months=int(round(period) ))            
        end_date = start_date + relativedelta(months=self.reset_period-1)    
        xx1['date'] = pd.to_datetime(xx1['date'])  
        mask = (xx1['date'] >= start_date_x) & (xx1['date'] <= end_date)
        df_x = xx1.loc[mask].reset_index()  
        df_x.drop('index', axis=1, inplace=True)    
        df_x.iloc[:, [1]]= df_x.iloc[:, [1]].astype(float)   
        ma_vect = pd.rolling_mean(df_x.set_index('date').resample('BM'),window=int(round(period))).dropna(how='any').reset_index().iloc[0:self.reset_period, :].iloc[:, [1]]    
        return ma_vect.values
    
    
    def MA_vectorized(self,df, vect_lag, vect_period, ma_vectorized):
        for i in self.reset_range:   
            for k in range(3):
                ma_vectorized[i+k] = self.MA_vect(df, i, vect_lag[i], vect_period[i], np.empty(3))[k]
           
        return ma_vectorized    
    
    def MA_func_vectorized(self, df, vect_lag, vect_period, ma_vectorized ):
        for i in self.reset_range:   
            for k in range(3):
                ma_vectorized[i+k] = self.MA_func_vect(df, i, vect_lag[i], vect_period[i], np.empty(3))[k] 
        return ma_vectorized 
        
    
    def MA_all_matrix(self,df, matrix_lag, matrix_period, matrix):
        vect1 = self.MA_vectorized(df[['date', 'oil']], matrix_lag[:,0], matrix_period[:,0], np.empty(len(matrix_lag[:,0]))).reshape(12,1)
        vect2 = self.MA_vectorized(df[['date', 'power']], matrix_lag[:,1], matrix_period[:,1], np.empty(len(matrix_lag[:,0]))).reshape(12,1)
        vect3 = self.MA_vectorized(df[['date', 'coal']], matrix_lag[:,2], matrix_period[:,2], np.empty(len(matrix_lag[:,0]))).reshape(12,1)
        vect4 = self.MA_vectorized(df[['date', 'gas']], matrix_lag[:,3], matrix_period[:,3], np.empty(len(matrix_lag[:,0]))).reshape(12,1)
        return np.concatenate((vect1,vect2,vect3,vect4),axis=1)
    
    def MA_func_all_matrix(self,df, matrix_lag, matrix_period, matrix):
        vect1 = self.MA_func_vectorized(df[['date', 'oil']], matrix_lag[:,0], matrix_period[:,0], np.empty(len(matrix_lag[:,0]))).reshape(12,1)
        vect2 = self.MA_func_vectorized(df[['date', 'power']], matrix_lag[:,1], matrix_period[:,1], np.empty(len(matrix_lag[:,0]))).reshape(12,1)
        vect3 = self.MA_func_vectorized(df[['date', 'coal']], matrix_lag[:,2], matrix_period[:,2], np.empty(len(matrix_lag[:,0]))).reshape(12,1)
        vect4 = self.MA_func_vectorized(df[['date', 'gas']], matrix_lag[:,3], matrix_period[:,3], np.empty(len(matrix_lag[:,0]))).reshape(12,1)
        return np.concatenate((vect1,vect2,vect3,vect4),axis=1)
        
        
    def MA_3_matrix(self,df, i, four_lags, four_periods, matrix):
        vect1 = self.MA_vect(df[['date', 'oil']], i, four_lags[0], four_periods[0], np.empty(3)).reshape(3,1)
        vect2 = self.MA_vect(df[['date', 'power']], i, four_lags[1], four_periods[1], np.empty(3)).reshape(3,1)
        vect3 = self.MA_vect(df[['date', 'coal']],  i,four_lags[2], four_periods[2], np.empty(3)).reshape(3,1)
        vect4 = self.MA_vect(df[['date', 'gas']], i, four_lags[3], four_periods[3], np.empty(3)).reshape(3,1)
        return np.concatenate((vect1,vect2,vect3,vect4),axis=1)
          
        
    def MA_func_3_matrix(self,df, i, four_lags, four_periods, matrix):
        vect1 = self.MA_func_vect(df[['date', 'oil']], i, four_lags[0], four_periods[0], np.empty(3))
        vect2 = self.MA_func_vect(df[['date', 'power']], i, four_lags[1], four_periods[1], np.empty(3))
        vect3 = self.MA_func_vect(df[['date', 'coal']],  i,four_lags[2], four_periods[2], np.empty(3))
        vect4 = self.MA_func_vect(df[['date', 'gas']], i, four_lags[3], four_periods[3], np.empty(3))
        return np.concatenate((vect1,vect2,vect3,vect4),axis=1)
          
    
    
    def func_lag_period(self,parameters, *data):
        
        lag1, lag2, lag3, lag4, period1, period2, period3, period4  = parameters
        df, y, i, coef, values = data
        values = compute_mse(y , self.MA_func_3_matrix(df, i, np.array([lag1, lag2, lag3, lag4]), np.array([period1, period2, period3, period4]), np.empty((len(y),4))), coef) 
        return values

        
    def opt_lag_period(self,coef):
        
        vect_lag1 = self.init_lag*np.ones(len(yy)).reshape(12,1)
        vect_lag2 = self.init_lag*np.ones(len(yy)).reshape(12,1)
        vect_lag3 = self.init_lag*np.ones(len(yy)).reshape(12,1)
        vect_lag4 = self.init_lag*np.ones(len(yy)).reshape(12,1)
        
        vect_period1 = self.init_period*np.ones(len(yy)).reshape(12,1)
        vect_period2= self.init_period*np.ones(len(yy)).reshape(12,1)
        vect_period3 = self.init_period*np.ones(len(yy)).reshape(12,1)
        vect_period4 = self.init_period*np.ones(len(yy)).reshape(12,1)
        
            
        for i in self.reset_range:
            start_date = datetime.strptime(self.start_date_str,"%Y-%m-%d %H:%M:%S") + relativedelta(months=i) 
            start_date_x = start_date - relativedelta(months=self.max_lag) - relativedelta(months=self.max_period)             
            end_date = start_date + relativedelta(months=self.reset_period) 
            
            xx['date'] = pd.to_datetime(xx['date'])  
            mask = (xx['date'] >= start_date_x) & (xx['date'] <= end_date)
            df_x = xx.loc[mask].reset_index()  
            df_x.drop('index', axis=1, inplace=True)
          
            y3 = yy[i:i+self.reset_period]['USD.20']
            
            args = (df_x, y3, i, coef, np.empty(1))
            
            td0 = time()
            result = differential_evolution(self.func_lag_period, self.bounds, args=args)
            td1 = time();
            d4 = td1 - td0
            print ("Duration of differential_evolution in Seconds %6.3f" % d4)          
          
            ti0 = time()
            for k in range(3):
                vect_lag1[i+k] = round(result.x[0])
                vect_lag2[i+k] = round(result.x[1])
                vect_lag3[i+k] = round(result.x[2])
                vect_lag4[i+k] = round(result.x[3])
                vect_period1[i+k] = round(result.x[4])
                vect_period2[i+k] = round(result.x[5])
                vect_period3[i+k] = round(result.x[6])
                vect_period4[i+k] = round(result.x[7])                  
            matrix_lag =  np.concatenate((vect_lag1,vect_lag2,vect_lag3,vect_lag4),axis=1)    
            matrix_period =  np.concatenate((vect_period1,vect_period2,vect_period3,vect_period4),axis=1)   
            ti1 = time();
            d5 = ti1 - ti0
            print ("Duration of the for loop K in Seconds %6.3f" % d5)          
        
        return matrix_lag, matrix_period    
     
    def alternate(self):
        max_iters = 100
        gamma = 0.7
        coef =  np.array([self.init_coef1,self.init_coef2,self.init_coef3,self.init_coef4])
         
        for itera in range(self.nbr_iterations):
            
            print('///////////////////////////////////////')
            print('Iteration: ', itera )
            
            #process 1: opt lag and opt period given coef
            t00 = time()
            matrix_lag , matrix_period = self.opt_lag_period(coef)
            t11 = time()
            d1 = t11-t00
            print ("Duration of process 1 in Seconds %6.3f" %d1)      
            
            #process2: opt coeff given lag and period 
            t02 = time()
            X_df = self.MA_func_all_matrix(xx, matrix_lag, matrix_period, np.empty((matrix_lag.shape[0],4)))                 
            w_initial = coef
            gradient_loss, gradient_w = gradient_descent(standardize(y), standardize(X_df), w_initial, max_iters, gamma)            
           
            X_train, X_test, y_train, y_test = train_test_split(standardize(X_df.reshape(12,4)), standardize(y), random_state=1)
            
            res_ridge = ridge_regression(X_train,y_train, X_test, y_test)
            y_pred_GD = np.dot(X_test,gradient_w)
             
            # update coef
            coef = res_ridge[0]
            
            print('Coef GD:', gradient_w ,'Error GD:', metrics.mean_squared_error(y_test, y_pred_GD))
            print('Coef RR:', res_ridge[0] ,'Error RR', res_ridge[1])
            t12 = time()
            d2 = t12-t02
            print ("Duration of process 2 in Seconds %6.3f" % d2)             
        return coef , matrix_lag , matrix_period           
       
        

if __name__ == '__main__':
    
    #Step 1: optimizing lag, period and get the coefficients       
    #optimization = class_alternate('2016-01-31 00:00:00', 10, 12 , 24, 3,  1 , 1, 0.1, 0.1 , -0.003 , 0.0002)
    optimization = class_alternate('2016-01-31 00:00:00', 4, 12 , 24, 3,  1 , 1, 0.1, 0.1 , -0.003 , 0.0002)
    t0 = time()
    coef , matrix_lag , matrix_period  = optimization.alternate()    
    t1 = time()
    d = t1 - t0
    print ("Total duration in Seconds %6.3f" % d)               
    print('final coef GD: ', coef)

    #Step 2: prediction of the forward curve (daily) for the russian gas index
    forward_curves_ma = MA_func_matrix_daily('2017-01-31 00:00:00',  '2017-12-29 00:00:00' , forward_curves, matrix_lag[-1,:] , matrix_priod[-1,:])   
    forward_index = predict(forward_curves_ma,coef)
    spread_estimated = forward_index - forward_gas['gas'].values  #not same length :s

    #Step 3: Model calibration (Schwartz?)
    
    #Step 4: Gas storage MC


forward_curves_ma = MA_func_matrix_daily('2017-01-31 00:00:00',  '2017-12-29 00:00:00' , forward_curves, np.array([1,1,1,1]) , np.array([2,2,2,2]))   
forward_index = predict(forward_curves_ma,np.array([1,1,1,1]))
spread_estimated = forward_index - forward_gas['gas'].values












           
            