# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 13:48:40 2018

@author: islipd
"""
from numpy import arange, ones, var, array
from numpy import sum as npsum
import datetime as dt 
from scipy.io import loadmat
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, ylabel, \
    xlabel
from Garch_Projection import *
import scipy
plt.style.use('seaborn')
import pandas as pd
import numpy as np
from Garch_Projection import MonteCarlo_Garch, compute_squared_sigmas, price_option_garch
from FitGARCHFP import FitGARCHFP
from  InvarianceTestEllipsoid import InvarianceTestEllipsoid
from autocorrelation import autocorrelation

y_proj = np.array(pd.read_pickle("C:\\Users\\islipd\\Documents\\Thesis Notebooks\\Pricing\\garch_SPX_projections.pkl"))
date_proj = pd.read_pickle("C:\\Users\\islipd\\Documents\\Thesis Notebooks\\Pricing\\proj_dates.pkl")
date_proj = pd.to_datetime(date_proj[0])
option_data = pd.read_csv("C:\\Users\\islipd\\Documents\\Thesis Notebooks\\Pricing\\SPX_OptionsData.csv",parse_dates=True)

option_data = option_data[option_data["Volume"] > 0]
option_data = option_data[option_data['Expiration'] > option_data[' DataDate']]
option_data['observed'] = 0.5*(option_data['Bid']+option_data['Ask'])

gamma, rho = [5,2/252]   
price = partial(price_option_garch, y_proj,date_proj,gamma, rho) 
gamma = np.arange(-10,5)
l = np.zeros_like(gamma)
for i in range(len(gamma)):
    l[i] = loss(y_proj, date_proj, rho, option_data,gamma[i])
    
objective = partial(loss,y_proj, date_proj, rho, option_data )
bnds = ((-10,10))
result = scipy.optimize.minimize_scalar(objective, bounds = bnds)

#%%
option_data = option_data[option_data['ActuaryPrice']>0]
gamma = np.zeros(option_data.shape[0])
thetas =  np.zeros(option_data.shape[0])
for i in range(len(gamma)):
    lss_i = partial(lss_1d, y_proj, date_proj, rho, option_data['observed'].iloc[i], option_data['Strike'].iloc[i], option_data['Expiration'].iloc[i], option_data['Type'].iloc[i])
    bnds = ((-10,10))
    result = scipy.optimize.minimize_scalar(lss_i, bounds = bnds)
    gamma[i] = result.x

#%%    
option_data['implied_gammas'] = gamma
option_data['normalized_strike'] = option_data['Strike']/option_data['UnderlyingPrice']
option_data['tau'] = pd.to_datetime(option_data.Expiration) - pd.to_datetime(option_data[' DataDate'])
option_data['tau'] = option_data['tau'].dt.days

calls = option_data[option_data.Type == 'call']
puts = option_data[option_data.Type == 'put']
#puts = option_data
calls = calls[calls.implied_gammas > -20]
puts = puts[puts.implied_gammas > -20]

Z = np.array([calls['normalized_strike'],calls['tau'],calls['implied_gammas']])
Y = np.array([puts['normalized_strike'],puts['tau'],puts['implied_gammas']])
import scipy.io as sio
sio.savemat('calls.mat', {'x':Z[0,:],'y':Z[1,:],'z':Z[2,:]})
sio.savemat('puts.mat', {'x':Y[0,:],'y':Y[1,:],'z':Y[2,:]})