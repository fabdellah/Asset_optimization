# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 14:55:35 2018

@author: fabdellah
"""


import numpy as np
from numpy import random

def payoff1(S,dv,injection_cost,withdrawal_cost):
     if dv > 0:
      h= -injection_cost*dv
     elif dv == 0:
      h = 0
     else:
      h = -withdrawal_cost*dv
        
      return h



def payoff(S,dv,injection_cost,withdrawal_cost):
     
      return -withdrawal_cost*dv


def penalty(S,v):
    
    #specify the penalty function otherwise return 0
    return 0
    




class gas_storage(object):
    """ 
    S0 : initial spot price
    T : time to maturity 
    steps : number of discrete times (delta_t = T/steps)
    r : riskless discount rate (constant)
    sigma :  volatility of returns
    
    """

    def __init__(self,  S0, V0, V_end, T, steps,M,  r, sigma, nbr_simulations,vMax,vMin , max_rate, min_rate , injection_cost, withdrawal_cost ):
     
            self.S0 = S0                                        # Parameters for the spot price
            self.T = T
            self.r = r
            self.sigma = sigma
            self.nbr_simulations = nbr_simulations
            self.V0 = V0
            self.V_end = V_end
        
            self.vMax = vMax                                    # Parameters for facility
            self.vMin = vMin                               
            self.max_rate = max_rate             
            self.min_rate = min_rate             
            
            self.injection_cost =injection_cost                 # Parameters for the payoff function
            self.withdrawal_cost = withdrawal_cost

            
            if S0 < 0 or T <= 0 or r < 0  or sigma < 0 or nbr_simulations < 0 :
              raise ValueError('Error: Negative inputs not allowed')
 
            self.steps = steps  
            self.M = M
            self.alpha = self.vMax/(self.M-1)
            self.delta_t = self.T / float(self.steps)
            self.discount = np.exp(-self.r * self.delta_t)
                       
        
    def simulated_price_matrix(self, seed = 1):
        """ Returns Monte Carlo simulated prices (matrix)
            rows: time
            columns: price-path simulation """

        
        np.random.seed(seed)
        simulated_price_matrix = np.zeros((self.steps + 2, self.nbr_simulations), dtype=np.float64)
        simulated_price_matrix[0,:] = self.S0
        for t in range(1, self.steps + 2):
            brownian = np.random.standard_normal( int(self.nbr_simulations / 2))
            brownian = np.concatenate((brownian, -brownian))        
            simulated_price_matrix[t, :] = (simulated_price_matrix[t - 1, :]      
                                  * np.exp((self.r - self.sigma ** 2 / 2.) * self.delta_t
                                  + self.sigma * brownian * np.sqrt(self.delta_t)))
            #needs to be specified according to the corresponding 2-factor model
        return simulated_price_matrix           

     
    
    def contract_value(self):
    
        
        value_matrix = np.zeros((self.simulated_price_matrix().shape[0],self.simulated_price_matrix().shape[1],self.M))  # time, path , volume level
        acc_cashflows = np.zeros_like(value_matrix)
        decision_rule = np.zeros_like(value_matrix)
        volume_level = np.zeros_like(self.simulated_price_matrix())
        
        decision_rule_avg = np.zeros((self.simulated_price_matrix().shape[0],self.M))
        volume_level_avg = np.zeros(self.simulated_price_matrix().shape[0])
        acc_cashflows_avg = np.zeros((self.simulated_price_matrix().shape[0],self.M))
        volume_level_avg[0] = self.V0
        #volume_level_avg[-1] = self.V_end
        
        volume_level[0, : ] = self.V0*np.ones(self.nbr_simulations)
        #volume_level[-1,: ] = self.V_end*np.ones(self.nbr_simulations)
        value_matrix[-1,: ,:] = penalty(self.simulated_price_matrix()[-1, :],volume_level[-1,:]) 
        acc_cashflows[-1,:,:] = penalty(self.simulated_price_matrix()[-1, :],volume_level[-1,:])
        
        
        
        from scipy.optimize import minimize
        
        for m in range(1,self.M):
            
            print ('-----------')
            print ('Volume level: %5.3f,' % ((m-1)*self.alpha ))
            volume_level[-1,:] = (m-1)*self.alpha
            volume_level_avg[-1] = (m-1)*self.alpha
            
            
            for t in range(self.steps , 0 , -1):
                
                
                regression = np.polyfit(self.simulated_price_matrix()[t, :], acc_cashflows[t+1, :, m] * self.discount, 2)
                continuation_value = np.polyval(regression, self.simulated_price_matrix()[t, :])
                
                for b in range(self.nbr_simulations):
                     f = lambda x: -1*( payoff(self.simulated_price_matrix()[t, b],
                                               x ,self.injection_cost,self.withdrawal_cost ) + continuation_value[b]  )
    
                     cons = ({'type': 'ineq', 'fun': lambda x:  (volume_level[t+1,b] - x - self.vMin)        },   
                             {'type': 'ineq', 'fun': lambda x:  (self.vMax - volume_level[t+1,b] + x)        },
                             {'type': 'ineq', 'fun': lambda x:  (self.max_rate - x)                          },
                             {'type': 'ineq', 'fun': lambda x:  (x - self.min_rate)                          },
                             {'type': 'ineq', 'fun': lambda x:  (self.max_rate*t - volume_level[t+1,b] + x ) })   
        
                     res = minimize(f, random.rand(1), constraints=cons)     
                     decision_rule[t,b,m] = res.x
    
                     if (decision_rule[t,b,m] != 0):         
                       volume_level[t,b] = volume_level[t+1,b]-  (np.sign(decision_rule[t,1,m]))*((abs(decision_rule[t,1,m])//self.alpha)+1)*self.alpha 
                     else:
                       volume_level[t,b] = volume_level[t+1,b]
                       
                     #print ('Time: %5.2f, Spot price: %5.1f , Decision rule (inject/withdraw): %5.1f , Volume level: %5.3f, Acc_cashflows: %5.2f '
                     #  % (t, self.simulated_price_matrix()[t, b] , decision_rule[t,b,m], volume_level[t,b], acc_cashflows[t,b,m] ))
  
                   
                acc_cashflows[t,:,m] = payoff(self.simulated_price_matrix()[t, :],
                             decision_rule[t,:,m] , self.injection_cost, self.withdrawal_cost) + acc_cashflows[t+1,:,m]*self.discount
                
                decision_rule_avg[t,m] = np.sum(decision_rule[t,:,m])/self.nbr_simulations
                volume_level_avg[t] = np.sum(volume_level[t,:])/self.nbr_simulations
                acc_cashflows_avg[t,m] = np.sum(acc_cashflows[t,:,m])/self.nbr_simulations
                
                print ('Time: %5.2f, Spot price: %5.1f ,  Decision rule (inject/withdraw): %5.1f , Acc_cashflows: %5.2f , Volume level: %5.3f'
                      % ( t, self.simulated_price_matrix()[t, 1] ,decision_rule[t,1,m], acc_cashflows_avg[t,m], volume_level[t,1]))
                #print('sign:  %5.3f , deux: %5.3f , v-add: %5.3f ' % ( np.sign(decision_rule[t,1,m]) , (abs(decision_rule[t,1,m])//self.alpha)+1 , (np.sign(decision_rule[t,1,m]))*((abs(decision_rule[t,1,m])//self.alpha)+1)*self.alpha ))
                

                
            import matplotlib.pyplot as plt
            time = np.arange(0,self.steps+2)
            plt.plot(time,volume_level[:,1])   
            plt.title('Volume level')
            plt.xlabel('Time')
            plt.ylabel('Volume [Mwh]')
            axes = plt.gca()
            axes.set_ylim(self.vMin,self.vMax)
            plt.show()
            #print ('Time: %5.2f, Spot price: %5.1f , Decision rule (inject/withdraw): %5.1f , Volume level: %5.3f, Acc_cashflows: %5.2f '
            #       % (t, self.simulated_price_matrix()[t, 1] , decision_rule[t,1,m], volume_level[t,1], acc_cashflows[t,1,m] ))

        
        
    
        return acc_cashflows[1,:,:] * self.discount             # at time 0

    
    def price(self):
        return round((np.sum(self.contract_value(),axis=0) /  float(self.nbr_simulations))[1] , 2)
            
            
    
# gas_storage(S0, V0,V_end, T, steps, M, r, sigma, nbr_simulations, vMax, vMin , max_rate , min_rate  , injection_cost, withdrawal_cost )  
         
facility1 =  gas_storage(25, 0,0, 1, 12, 11, 0.06, 0.945, 10, 500000, 0 , 34000, -53000  ,0.1, 0.1 )  
    
 
    
from time import time
t0 = time()
print ('Price: ' ,facility1.price() )   
t1 = time();
d1 = t1 - t0
print ("Duration in Seconds %6.3f" % d1)       
    
    
