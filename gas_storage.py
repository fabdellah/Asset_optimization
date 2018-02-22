# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 17:27:37 2018

@author: fabdellah
"""

"Class for gas storage  : compute the storage value as the average of the accumulated future cash flow (over all price paths)"
"Documnets used:"
"Gas Storage Valuation using a Monte Carlo Method. - Boogert & De Jong"
"Book: Valuation and Risk Management in Energy Markets.p.322 - Glen Swindle"


import numpy as np


def injection_cost(S,a1,b1):
    return (1+a1)*S+b1



def withdrawal_profit(S,a2,b2):
    return (1-a2)*S-b2



def payoff(S,dv,a1,b1,a2,b2):
    # if dv > 0:
    h= -injection_cost(S,a1,b1)*dv
    #elif dv == 0:
    #    h = 0
    #else:
    #    h = -withdrawal_profit(S,a2,b2)*dv
        
    return h


def penalty(S,v):
    
    #specify the penalty function otherwise return 0
    return 0
    


class gas_storage(object):
    """
    
    S0 : initial stock price
    T : time to maturity 
    K : number of discrete times (delta_t = T/K)
    r : riskless discount rate (constant)
    sigma :  volatility of returns
    
    """

    def __init__(self,  S0, T, K,  r, sigma, nbr_simulations,vMax,vMin , max_inj_rate, max_wit_rate , min_inj_rate , min_wit_rate ,a1,b1,a2,b2 ):
     
            self.S0 = S0                                        # Parameters for the spot price
            self.T = T
            self.r = r
            self.sigma = sigma
            self.nbr_simulations = nbr_simulations
        
            self.vMax = vMax                                    # Parameters for facility
            self.vMin = vMin                               
            self.max_inj_rate = max_inj_rate             
            self.max_wit_rate = max_wit_rate
            self.min_inj_rate = min_inj_rate             
            self.min_wit_rate = min_wit_rate
            
            self.a1 = a1                                        # Parameters for the payoff function
            self.b1 = b1
            self.a2 = a2
            self.b2 = b2
            
            if S0 < 0 or T <= 0 or r < 0  or sigma < 0 or nbr_simulations < 0 :
              raise ValueError('Error: Negative inputs not allowed')
 
            self.K = K  
            self.delta_t = self.T / float(self.K)
            self.discount = np.exp(-self.r * self.delta_t)
            
            
            
    def simulated_price_matrix(self, seed = 1):
        """ Returns Monte Carlo simulated prices (matrix)
            rows: time
            columns: price-path simulation """
        np.random.seed(seed)
        simulated_price_matrix = np.zeros((self.K + 1, self.nbr_simulations), dtype=np.float64)
        simulated_price_matrix[0,:] = self.S0
        for t in range(1, self.K + 1):
            brownian = np.random.standard_normal( int(self.nbr_simulations / 2))
            brownian = np.concatenate((brownian, -brownian))
            
            simulated_price_matrix[t, :] = (simulated_price_matrix[t - 1, :]      #needs to be specified according to the corresponding 2-factor model
                                  * np.exp((self.r - self.sigma ** 2 / 2.) * self.delta_t
                                  + self.sigma * brownian * np.sqrt(self.delta_t)))
        return simulated_price_matrix
            
       
    
    def contract_value(self):
        
        value_matrix = np.zeros_like(self.simulated_price_matrix())
        acc_cashflows = np.zeros_like(self.simulated_price_matrix())
        decision_rule = np.zeros_like(self.simulated_price_matrix())
        volume_level = np.zeros_like(self.simulated_price_matrix())
        
        decision_rule_avg = np.zeros(self.simulated_price_matrix().shape[0])
        volume_level_avg = np.zeros(self.simulated_price_matrix().shape[0])
        acc_cashflows_avg = np.zeros(self.simulated_price_matrix().shape[0])
        
        value_matrix[-1, :] = penalty(self.simulated_price_matrix()[-1, :],0)   #suppose the final volume = 0 at time T+1
        acc_cashflows[-1, :] = penalty(self.simulated_price_matrix()[-1, :],0)
        
        from scipy.optimize import minimize
        for t in range(self.T, 0 , -1):
            regression = np.polyfit(self.simulated_price_matrix()[t, :], acc_cashflows[t + 1, :] * self.discount, 5)
            continuation_value = np.polyval(regression, self.simulated_price_matrix()[t, :])
            
            for b in range(self.nbr_simulations):
                 f = lambda x: -1* payoff(self.simulated_price_matrix()[t, b], x ,self.a1, self.b1, self.a2, self.b2 ) + continuation_value[b]   

                 cons = ({'type': 'ineq', 'fun': lambda x:  volume_level[t,b] - x - self.vMin },
                         {'type': 'ineq', 'fun': lambda x:  self.vMax - volume_level[t,b] + x },
                         {'type': 'ineq', 'fun': lambda x:  self.max_inj_rate - x             },
                         {'type': 'ineq', 'fun': lambda x:  self.max_wit_rate - x             },
                         {'type': 'ineq', 'fun': lambda x:  x - self.min_inj_rate             },
                         {'type': 'ineq', 'fun': lambda x:  x - self.min_wit_rate             })   
    
                 res = minimize(f, 0, constraints=cons) 
                 decision_rule[t,b] = res.x

                 
            volume_level[t+1,:] = volume_level[t,:] + decision_rule[t,:]
            acc_cashflows[t,:] = payoff(self.simulated_price_matrix()[t, :], decision_rule[t,:] ,self.a1, self.b1, self.a2, self.b2) + acc_cashflows[t+1,:]*self.discount
            
            decision_rule_avg[t] = np.sum(decision_rule[t,:])/self.nbr_simulations
            volume_level_avg[t] = np.sum(volume_level[t,:])/self.nbr_simulations
            acc_cashflows_avg[t] = np.sum(acc_cashflows[t,:])/self.nbr_simulations
            
            print (self.simulated_price_matrix()[t, 1], decision_rule_avg[t], acc_cashflows_avg[t] ,  volume_level_avg[t], acc_cashflows_avg[t] , t)

        return acc_cashflows[1,:] * self.discount             # at time 0


    
    def price(self):
        return np.sum(self.contract_value()) /  float(self.nbr_simulations)        
            
            
    
# gas_storage(S0, T, K, r, sigma, nbr_simulations, vMax, vMin , max_inj_rate, max_wit_rate , min_inj_rate , min_wit_rate  ,a1, b1, a2, b2 )  
         
facility1 =  gas_storage(20, 5, 10 , 0.06, 0.2, 1000, 600000, 0 , 20000, 20000, 0 ,0  ,1, 1, 1, 1 )  
    
    
print ('Price: ', facility1.price())   
    

    
#f = lambda x: -1* payoff(self.simulated_price_matrix()[t, :], x ,self.a1, self.b1, self.a2, self.b2 ) + continuation_value   
    
# print ("decision rule (inject/withdraw): %5.1f , acc_cashflows: %5.2f,  Volume level %5.3f,  time: %5.2f," % (decision_rule_avg[t], acc_cashflows_avg[t] ,  volume_level_avg[t], acc_cashflows_avg[t] , t))
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            
            
            
            