# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 20:46:38 2018

@author: islipd
"""

import numpy as np
import pandas as pd
#default parameters: 


def rho(z):
    epsilon = 0.1
    if z>= epsilon:
        return z
    elif np.abs(z) < epsilon:
        return z**2/(4*epsilon) + 0.5*z + 0.25*epsilon
    else:
        return 0
    return -1

def Drho(z):
    epsilon = 0.1
    if z>= epsilon:
        return 1
    elif np.abs(z) < epsilon:
        return z/(2*epsilon) + 0.5
    else:
        return 0
    return -1
vrho = np.vectorize(rho)
vDrho = np.vectorize(Drho)

def CVaR(x, Beta, Lss, deltaV,c):
    
    return x[0] + (1/(1-Beta))*vrho(Lss - np.inner(deltaV,x[1:])-x[0]).mean() + c*abs(x[1:]).sum()

def grad_CVaR(x, Beta, Lss, deltaV,c):
    
    der = np.zeros_like(x)
    der[0] = 1 - (1/(1-Beta))*vDrho(Lss - np.inner(deltaV,x[1:])-x[0]).mean()
    left = np.expand_dims((1/(1-Beta))*vDrho(Lss - np.inner(deltaV,x[1:])-x[0]),axis=0)
    M = left.size
    der[1:] = -1*(1/M)*np.inner(left,np.transpose(deltaV)) + c
    return der

def var(x, Lss, deltaV,c):
    f = Lss - np.inner(deltaV,x)+c*abs(x).sum()
    return f.var()

def grad_var(x, Lss, deltaV,c):
    
    d_mu = -1*deltaV.mean(axis=0)
    f = np.expand_dims(Lss - np.inner(deltaV,x),axis=1)
    der = 2*np.multiply(f - f.mean(), (-1)*deltaV - d_mu).mean(axis=0)+c
    
    return der

def semi_var(x, Lss, deltaV,c,b):
    f = Lss - np.inner(deltaV,x)
    mu = b
    return (vrho(f-mu)**2).mean()

def grad_semi_var(x, Lss, deltaV,c,b):
    f = np.expand_dims(Lss - np.inner(deltaV,x),axis=1)
    d_mu = -1*deltaV.mean(axis=0)
    mu = b
    der =  2*np.multiply(vDrho(f - mu), (-1)*deltaV - d_mu).mean(axis=0)
    return der

def plot_schedule(dV, weights):
    from datetime import datetime
    schedule = pd.DataFrame(dV.columns[weights > 0.05],columns = ['A'])
    schedule.A
    schedule = pd.DataFrame([*schedule.A],columns = ['date', 'duration', 'strike (std)', 'index', 'option type'])
    schedule['ind'] = schedule['index'].str.cat(schedule['option type'], sep='--')
    schedule['ind'] = schedule['ind'].str.cat(schedule['duration'].astype(str), sep='--')
    schedule['ind'] = schedule['ind'].str.cat(schedule['strike (std)'].astype(str), sep='--')
    schedule['ind'] = schedule['ind'].str.cat(schedule['date'].astype(str), sep='--')
    schedule['Start_Date'] = pd.to_datetime(schedule['date']).astype(datetime)
    schedule ['End_Date'] = pd.to_datetime(schedule['date']+pd.to_timedelta(schedule['duration'], unit = 'd')).astype(datetime)
    
    import matplotlib.pyplot as plt
    import matplotlib.dates as matdt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = ax.xaxis_date()
    ax = plt.hlines(schedule['ind'], matdt.date2num(schedule['Start_Date']), matdt.date2num(schedule ['End_Date']))

