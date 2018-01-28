# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 23:09:32 2018
This file contains a set of functions used to do monte carlo simulations for a 
calibrated garch(1,1) model. 
@author: islipd
"""

import datetime as dt 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, ylabel, \
    xlabel

plt.style.use('seaborn')
from functools import partial
import pandas as pd
import numpy as np

def compute_squared_sigmas(X, initial_sigma, theta,t_):
    #compute the implied volatilities from the calibrated garch model
    
    #a0, a1, a2, mu are parameters contained in the input vector theta
    a0 = theta[0]
    a1 = theta[1]
    b1 = theta[2]
    mu = theta[3]
    
    #define the horizon
    T = t_
    
    #initialize the output vector 
    sigma2 = np.zeros_like(X)
    
    #use an initial guess for sigma 
    sigma2[0] = initial_sigma
    
    for t in range(1, T):
        # Here's where we apply the garch defining equation
        sigma2[:,t] = a0 + a1*(X[:,t-1]-mu)**2 + b1*sigma2[:,t-1]
    return sigma2

def MonteCarlo_Garch(startd, endd, N, date, theta, StocksSPX,dx, resid, sigma2_t):
    #this function computes a set of N paths from a calibrated garch model
    #startd
    #endd
    #N
    #date
    #theta
    #StocksSPX
    #dx
    #resid
    #sigma2_t
    
    #get the projected date series 
    date_proj = pd.date_range(start=startd, end=endd)
    date_proj = date_proj[date_proj.dayofweek <5]
    tau = len(date_proj)
    sigma2_proj = np.zeros([tau,N])
    dx_proj = np.zeros([tau,N])
    x_proj = np.zeros([tau,N])
    a0 = theta[0]
    a1 = theta[1]
    b1 = theta[2]
    mu = theta[3]
    start_ind = sum(StocksSPX.index < startd)
    sigma2_proj[0,:] = sigma2_t[:,start_ind]
    dx_proj[0,:] = dx[:,start_ind]
    x_proj[0,:]  = np.log(StocksSPX)[start_ind]
    for i in range(1,tau):
        sigma2_proj[i,:] = a0 + a1*(dx_proj[i-1,:] - mu)**2+b1*sigma2_proj[i-1,:]
        iid_innov = np.random.choice(resid[0,:],size =[1,N])
        dx_proj[i,:] = mu + iid_innov*(sigma2_proj[i,:]**0.5)
        x_proj[i,:] = x_proj[i-1,:] + dx_proj[i,:]     
    y_proj = np.e**x_proj
    return [y_proj, x_proj, sigma2_proj, dx_proj, date_proj]

def price_option_garch(y_proj,date_proj, gamma, rho,end,typ, strike):
    end_index = np.argmin(abs(date_proj - pd.to_datetime(end)))
    disc = np.e**(-1*rho*end_index)*y_proj[0,0]**(-1*gamma)
    if typ == 'call':
        payout = y_proj[end_index,:]**(gamma)*np.maximum(y_proj[end_index,:] - strike,0)
        return disc*payout.mean()
    if typ == 'put':
        payout = y_proj[end_index,:]**(gamma)*np.maximum(strike - y_proj[end_index,:],0)
        return disc*payout.mean()
    
def loss(y_proj, date_proj, rho, option_data,gamma):
    price = partial(price_option_garch, y_proj,date_proj,gamma, rho) 
    option_data["ActuaryPrice"] = option_data.apply(lambda row: price(row["Expiration"],row["Type"],row["Strike"]),axis = 1)
    lss = (option_data["ActuaryPrice"] - (option_data["Bid"]+option_data["Ask"])/2)**2
    lss = lss.sum()
    return lss

def lss_1d(y_proj, date_proj, rho, observed_price, strike, expiration, typ, gamma):
    price = partial(price_option_garch, y_proj,date_proj, gamma, rho) 
    actuary_price = price(expiration,typ,strike)
    lss = actuary_price - observed_price
    return lss**2


