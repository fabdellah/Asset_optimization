# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:11:29 2018

@author: fabdellah
"""



##Valuing American Options by Simulation: A Simple Least-Squares Approach

import numpy as np

class American_Options(object):
    """
    option_value : call option or put option
    S0 : initial stock price
    strike : strike price
    T : time to maturity 
    K : number of discrete times (delta_t = T/K)
    r : riskless discount rate (constant)
    sigma :  volatility of returns
    
    """

    def __init__(self, option_type, S0, strike, T, K, r, sigma, nbr_simulations):
            self.option_type = option_type
            self.S0 = S0
            self.strike = strike
            self.T = T
            self.K = K
            self.r = r
            self.sigma = sigma
            self.nbr_simulations = nbr_simulations

            if option_type != 'call' and option_type != 'put':
              raise ValueError("Error: option type not valid. Enter 'call' or 'put'")
            if S0 < 0 or strike < 0 or T <= 0 or r < 0  or sigma < 0 or nbr_simulations < 0 :
              raise ValueError('Error: Negative inputs not allowed')

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
            simulated_price_matrix[t, :] = (simulated_price_matrix[t - 1, :]
                                  * np.exp((self.r - self.sigma ** 2 / 2.) * self.delta_t
                                  + self.sigma * brownian * np.sqrt(self.delta_t)))
        return simulated_price_matrix

    
    def payoff_matrix(self):
        """ Returns the payoff of American Option (matrix) """
        if self.option_type == 'call':
            payoff = np.maximum(self.simulated_price_matrix() - self.strike,
                           np.zeros((self.K + 1, self.nbr_simulations),dtype=np.float64))
        else:
            payoff = np.maximum(self.strike - self.simulated_price_matrix(),
                            np.zeros((self.K + 1, self.nbr_simulations),
                            dtype=np.float64))
        return payoff

   
    def value_vector(self):
        value_matrix = np.zeros_like(self.payoff_matrix())
        value_matrix[-1, :] = self.payoff_matrix()[-1, :]
        for t in range(self.K - 1, 0 , -1):
            regression = np.polyfit(self.simulated_price_matrix()[t, :], value_matrix[t + 1, :] * self.discount, 5)
            continuation_value = np.polyval(regression, self.simulated_price_matrix()[t, :])
            value_matrix[t, :] = np.where(self.payoff_matrix()[t, :] > continuation_value,
                                          self.payoff_matrix()[t, :],
                                          value_matrix[t + 1, :] * self.discount)

        return value_matrix[1,:] * self.discount


    
    def price(self):
        return np.sum(self.value_vector()) /  float(self.nbr_simulations)



def prices():
    for S0 in (36., 38., 40., 42., 44.):  
        for volatility in (0.2, 0.4):  
            for T in (1.0, 2.0):  
                AmericanPUT = American_Options('put', S0, 40., T, 50, 0.06, volatility, 1500)
                print ("Initial price: %5.1f , Sigma: %5.2f, Expire: %5.1f , Option Value %5.3f" % (S0, volatility, T, AmericanPUT.price()) )
 
    
AmericanPUT = American_Options('put', 36., 40., 1., 50, 0.06, 0.2, 10000)
print ('Price: ', AmericanPUT.price())   

values = prices() 
    
    
    
    




























    
    
    
    