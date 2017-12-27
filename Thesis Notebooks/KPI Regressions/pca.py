# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:33:01 2017

@author: islipd
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import datetime as dt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
weather = pd.read_excel("weather.xlsx")
#weather = weather[weather["Year"] <=2016]
#%%
def iid(invariants):
    """takes in a series and does iid analysis on the time series"""
    invariants = invariants.fillna(0)
    length = len(invariants)
    first_half = invariants[1:length//2]
    second_half = invariants[length//2+1:length]
    l = pd.concat([first_half, second_half], axis = 1)
    l.columns = ['1st', '2nd']
    plt.subplot(2,2,1)
    first_half.hist()
    plt.subplot(2,2,2)
    second_half.hist()
    plt.subplot(2,2,3)
    pd.tools.plotting.lag_plot(invariants)
    
#%% 
#deal with rain

rain = pd.concat([weather["Date/Time"],weather["Year"], weather["Month"],weather["Total Rain (mm)"]],axis = 1)
rain = rain[rain["Month"] >= 8]