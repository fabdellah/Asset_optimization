# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 15:22:35 2018

@author: fabdellah
"""
# le code compile:
#total duration: 86.53 sec, optimla lag= 11.55 (pk real?), optimal ma_period= 11, optimal reset_period=6
#A faire: rajouter constant puis constant + power, coal, gas et voir si l'erreur diminue

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

df = pd.read_excel('spot_prices.xls')
df_oil = df[['date_oil', 'oil']]
df_oil.columns = ['date', 'oil']


yy = russia_Index[225:237]           #monthly from 2016 to 2017
yy = yy.reset_index()  
yy.drop('index', axis=1, inplace=True)
yy.drop('level_0', axis=1, inplace=True)
y = yy['USD.20']




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
    e = y - x*coef #np.dot(x,coef)
    mse = e.dot(e) / (2 * len(y))
    return mse


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx*w #np.dot(tx,w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def predict(tx,coef):
    #tx = df[['oil','power','coal','gas']].values
    return  tx*coef #np.dot(tx,coef)
    

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
 
  
     
def MA_plot(df1, start_date_str,end_date_str, lag, ma_period,reset_period):    
    nbr_months_per_year = 12
    nbr_reset_periods = int(nbr_months_per_year/reset_period)
    vect = MA_func_vect_out(df1, start_date_str,end_date_str, lag, ma_period,reset_period,np.empty(0))
    time = np.arange(0,nbr_months_per_year)
    plt.plot(time, vect)   
    plt.title('Moving average: Lag = '+str(lag)+ ' MA period = '+ str(ma_period)+ ' Reset period = '+ str(reset_period)  )
    plt.xlabel('Time')
    plt.ylabel('Average level')
    plt.show()
 
 

class class_alternate(object):
    
    def __init__(self, df_oil, start_date_str,end_date_str, nbr_months_per_year, nbr_iterations, max_lag, max_ma_period, max_reset_period ,init_coef1):
            self.max_lag = max_lag
            self.max_ma_period = max_ma_period
            self.max_reset_period = max_reset_period
            
            self.nbr_iterations = nbr_iterations
            self.init_coef1 = init_coef1
 
            self.nbr_months_per_year = nbr_months_per_year
            self.bounds = [(1, self.max_lag), (1, self.max_ma_period), (2, self.max_reset_period)]
              
            self.df_oil = df_oil

            self.start_date_str = start_date_str                                             #start_date_str = '2016-01-31 00:00:00'
            self.start_date = datetime.strptime(self.start_date_str,"%Y-%m-%d %H:%M:%S") 
            self.end_date_str = end_date_str                                                 #end_date_str = '2017-01-31 00:00:00'
            self.end_date = datetime.strptime(self.end_date_str,"%Y-%m-%d %H:%M:%S") 
            
            self.rranges = (slice(1, self.max_lag, 1), slice(1, self.max_ma_period, 1), slice(3, max_reset_period, 3))
        
    def MA_func_vect(self, lag: int, ma_period: int ,reset_period: int ,vect): #vect=np.empty(0) # pour reset_period=1 len(ma_vect)=11 au lieu de 12 je c pas pk
        #reset_period = 6
        start_date_x = self.start_date - relativedelta(months=int(round(lag))) - relativedelta(months=int(round(ma_period) ))            
        end_date =   self.end_date + relativedelta(months=int(round(lag))) + relativedelta(months=int(round(ma_period) ))+ relativedelta(months=1) #pour avoir 12 valeurs dans ma_vect
        self.df_oil['date'] = pd.to_datetime(self.df_oil['date'])  
        mask = (self.df_oil['date'] >= start_date_x) & (self.df_oil['date'] <= end_date)
        df_x = self.df_oil.loc[mask].reset_index()  
        df_x.drop('index', axis=1, inplace=True)    
        df_x.iloc[:, [1]] = df_x.iloc[:, [1]].astype(float)          
        #df_x_monthly = df_x.set_index('date').resample('1BM').mean().reset_index()
        #print('df_x_monthly : (dataframe? index date?): ', df_x_monthly)
        #ma_vect = np.zeros(12)
        ma_monthly = pd.rolling_mean(df_x.set_index('date').resample('1BM'),window=int(round(ma_period))).dropna(how='any').reset_index().iloc[0:12, [1]].values
        print('-----------------------------------------')
        print('ma_monthly : (dataframe? index date?): ', ma_monthly)
        print('longueur de ma_monthly: ',len(ma_monthly))
        print('lag = ', lag, ' ma_period =  ', ma_period, ' reset period  = ' , round(reset_period))
        ma_vect = [ ma_monthly[i] for i in range(0,self.nbr_months_per_year,int(round(reset_period))) ]
        #ma_vect =  pd.rolling_mean(df_x_monthly.iloc[ma_period-1:11, :].set_index('date').resample(str(reset_period)+'BM'),window=int(round(ma_period))).reset_index().iloc[:, [1]].values
          
        #ma_vect =  pd.rolling_mean(df_x_monthly.set_index('date').resample(str(reset_period)+'BM'),window=int(round(ma_period))).dropna(how='any').reset_index().iloc[:, [1]].values
       
        #ma_vect = df_x.set_index('date').rolling(str(int(round(ma_period)))+'BM').mean().resample(str(reset_period)+'BM').first()
        #ma_vect = pd.rolling_mean(df_x.set_index('date').resample(str(reset_period)+'BM'),window=int(round(ma_period))).dropna(how='any').reset_index().iloc[:, [1]].values  # ici problem avec la taille de vect    
        print('longueur de ma_vect: ', len(ma_vect))
       
        #ma_vect = pd.rolling_mean(df_x.set_index('date').resample('BM'),window=int(round(ma_period))).dropna(how='any').reset_index().iloc[:, [1]].values            
        #ma_vect = np.array([1,2,3,4,5,6]) # ca run avec ca
        nbr_reset_periods = int(self.nbr_months_per_year/int(round(reset_period)))
        print('nbr_reset_periods: ',nbr_reset_periods)
        vect = np.empty(0)
        for i in range(0,nbr_reset_periods):
            #print('ma_vect[i]',i, ma_vect[i])           
            vect = np.append(vect , ma_vect[i]*np.ones(int(round(reset_period))))      
        print('vect: ',vect)
        return vect
    
        
    def func_lag_period(self, parameters, *data):
        
        lag1, period1, reset1 = parameters
        df_oil, y, coef, values = data
        print('ici func_lag_period')
        values = compute_mse(y , self.MA_func_vect(lag1, period1, reset1, np.empty(0)), coef) 
        return values


    def brute_optimization(self, y, coef):
                
        args = (self.df_oil, y, coef, np.empty(1))
        resbrute = optimize.brute(self.func_lag_period, self.rranges, args=args, full_output=True,
                              finish=optimize.fmin)
        return resbrute[0]  # global minimum
        
    
    def alternate(self):
        max_iters = 100
        gamma = 0.7
        coef =  self.init_coef1
         
        for itera in range(self.nbr_iterations):
            
            print('///////////////////////////////////////')
            print('Iteration: ', itera )
            
            #process 1: opt lag and opt period given coef
            y = yy['USD.20']
            t00 = time()
            lag , ma_period, reset_period = self.brute_optimization(y,coef)
            t11 = time()
            d1 = t11-t00
            print ("Duration of process 1 in Seconds %6.3f" %d1)      
            
            #process2: opt coeff given lag and period 
            t02 = time()
            X_df = self.MA_func_vect(lag, ma_period, reset_period, np.empty(0))     
            #print('X_df', X_df)
            #XX = np.c_[np.ones(len(X_df)),X_df]   
            w_initial = coef
            gradient_loss, gradient_w = gradient_descent(standardize(y), standardize(X_df), w_initial, max_iters, gamma)            
            #print('stand y: ',standardize(y), 'stand X_df: ',standardize(X_df), 'w_init: ',w_initial, 'Coef GD:', gradient_w )
            X_train, X_test, y_train, y_test = train_test_split(standardize(X_df.reshape(self.nbr_months_per_year,1)), standardize(y), random_state=1)
            
            res_ridge = ridge_regression(X_train,y_train, X_test, y_test)
            y_pred_GD = np.dot(X_test,gradient_w)
             
            # update coef
            coef = res_ridge[0]
            
            print('Coef GD:', gradient_w ,'Error GD:', metrics.mean_squared_error(y_test, y_pred_GD))
            print('Coef RR:', res_ridge[0] ,'Error RR', res_ridge[1])
            t12 = time()
            d2 = t12-t02
            print ("Duration of process 2 in Seconds %6.3f" % d2)             
        return coef , lag , ma_period, reset_period    
       
        


if __name__ == '__main__':
    
    #Step 1: optimizing lag, ma_period, reset_period and get the coefficients     
    #df_oil, start_date_str,end_date_str, nbr_months_per_year, nbr_iterations, max_lag, max_ma_period, max_reset_period ,init_coef1
    optimization = class_alternate(df_oil, '2016-01-31 00:00:00','2017-01-31 00:00:00', 12, 5, 12 , 12, 7, 0)
    t0 = time()
    coef , lag , ma_period, reset_period   = optimization.alternate()    
    t1 = time()
    d = t1 - t0
    print ("Total duration in Seconds %6.3f" % d)               
    print('final coef: ', coef)
    
    lag=1
    ma_period = 2
    reset_period = 2
    
    v = optimization.MA_func_vect(lag, ma_period,reset_period, np.empty(0))
    
    
    