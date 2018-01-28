# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 15:01:23 2018

@author: islipd
"""
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

def priceOption(t_start, tau, K, T, Tin, kappa,r):
    
    CDDs = np.maximum(Tin[t_start:t_start+dt.timedelta(days=tau)] - T,0)
    HDDs = np.maximum(-1*Tin[t_start:t_start+dt.timedelta(days=tau)] + T,0)
    CDDscall = np.e**(-r*tau)*np.maximum(CDDs.sum() - K,0).mean()+kappa*np.maximum(CDDs.sum() - K,0).std()
    CDDsput = np.e**(-r*tau)*np.maximum(K-CDDs.sum(),0).mean()+kappa*np.maximum(K-CDDs.sum(),0).std()
    HDDscall = np.e**(-r*tau)*np.maximum(HDDs.sum() - K,0).mean()+kappa*np.maximum(HDDs.sum() - K,0).std()
    HDDsput = np.e**(-r*tau)*np.maximum(K-HDDs.sum(),0).mean()+kappa*np.maximum(K-HDDs.sum(),0).std()
    HDDsfutures = np.e**(-r*tau)*HDDs.sum().mean()
    CDDsfutures = np.e**(-r*tau)*CDDs.sum().mean()
    ind = ["call", 'put', "futures"]
    cols = ["HDD", 'CDD']
    prices = [(HDDscall,CDDscall), (HDDsput, CDDsput), (HDDsfutures, CDDsfutures)]
    
    return pd.DataFrame(prices, columns = cols, index = ind)



# In[19]:

def dividend_process(mu, m, phi, alpha, gamma, t_start, tau, r, delta_0, eta):
    sigma = 0.2/(252*(1+(phi**2)/(1-phi**2)))
    rho = r
    # ### Time to simulate some dividends
    nu_t = sigma*(np.random.normal(0, 1, eta.shape) + (phi/(1-phi**2)**0.5)*eta)
    nu_t = np.array(nu_t[t_start:t_start+dt.timedelta(days=tau)])
    
    #solve for initial dividend by selting delta such that the interest rate is 6%
    tol=1
    iters = []
    for i in range(100):
        iters.append(delta_0)
        Deltas = np.zeros(nu_t.shape)
        Deltas[0,:] = delta_0
        for i in range(tau):
            Deltas[i+1,:] = np.e**(alpha + mu*np.log(Deltas[i,:])+nu_t[i,:])
        E = np.log(Deltas[tau,:]/delta_0).mean()
        V = np.log(Deltas[tau,:]/delta_0).var()
        tol = r - rho + (gamma*E+0.5*(gamma**2)*V)/tau
        delta_0 = delta_0 - tol
    plt.figure(1)
    plt.subplot(211)
    plt.plot(iters)
    
    print(tol)
        
    Deltas = np.zeros(nu_t.shape)
    Deltas[0,:] = delta_0
    for i in range(tau):
        Deltas[i+1,:] = np.e**(alpha + mu*np.log(Deltas[i,:])+nu_t[i,:])
    
    plt.subplot(212)
    plt.plot(Deltas[:,0:3]);
    return [Deltas,delta_0]

def priceCaoWeiV0(t_start, tau, NumStds, T, Tin, Pin, indexType, optionType, Dividend, gamma, rho):
    #this function returns the premium for a weather derivative 
    #INPUTS:
    #t_start: starting date for derivatives
    #tau: duration of option
    #Numstds: set the strike of the option
    #T: Threshold to define index 
    #Tin, Pin: Temperature and Precipitation Monte Carlo 
    #index type and option Type: 'CDD', 'HDD', 'Precip'
    Deltas = Dividend[0]
    delta_0 = Dividend[1]
    if indexType == "CDD":
        CDDs = np.maximum(Tin[t_start:t_start+dt.timedelta(days=tau)] - T,0)
        meanCDD = np.maximum(Tin[t_start:t_start+dt.timedelta(days=tau)] - T,0).sum().mean()
        stdCDD = np.maximum(Tin[t_start:t_start+dt.timedelta(days=tau)] - T,0).sum().std()
        K =  meanCDD + NumStds*stdCDD
    if indexType == "HDD":
        HDDs = np.maximum(-1*Tin[t_start:t_start+dt.timedelta(days=tau)] + T,0)
        meanHDD = np.maximum(-1*Tin[t_start:t_start+dt.timedelta(days=tau)] + T,0).sum().mean()
        stdHDD = np.maximum(-1*Tin[t_start:t_start+dt.timedelta(days=tau)] + T,0).sum().std()
        K =  meanHDD + NumStds*stdHDD
    if indexType == "Precip":
        Precip = np.maximum(Pin[t_start:t_start+dt.timedelta(days=tau)],0)
        meanPrecip = Precip.sum().mean()
        stdPrecip = Precip.sum().std()
        K =  meanPrecip + NumStds*stdPrecip
    disc = np.e**(-1*rho*tau)*(delta_0**(-1*gamma))
    if optionType == "call" and indexType== "CDD":
        CDDscall = disc*(((Deltas[tau,:]**gamma)*np.maximum(CDDs.sum() - K,0))).mean()
        return CDDscall
    if optionType == "put" and indexType== "CDD":
        CDDsput = disc*((Deltas[tau,:]**gamma)*np.maximum(K-CDDs.sum(),0)).mean()
        return CDDsput
    if optionType == "call" and indexType== "HDD":
        HDDscall = disc*((Deltas[tau,:]**gamma)*np.maximum(HDDs.sum() - K,0)).mean()
        return HDDscall
    if optionType == "put" and indexType== "HDD":
        HDDsput = disc*((Deltas[tau,:]**gamma)*np.maximum(K-HDDs.sum(),0)).mean()
        return HDDsput  
    if optionType == "call" and indexType== "Precip":
        Precipcall = disc*((Deltas[tau,:]**gamma)*np.maximum(Precip.sum()-K,0)).mean()
        return Precipcall
    if optionType == "put" and indexType== "Precip":
        Precipput = disc*((Deltas[tau,:]**gamma)*np.maximum(K-Precip.sum(),0)).mean()
        return Precipput     
    return -1

def priceCaoWeidV(t_start, tau, NumStds, T, Tin, Pin, indexType, optionType, Dividend, gamma, rho):
    #this function returns the option deltas over the period of interest  
    #INPUTS:
    #t_start: starting date for derivatives
    #tau: duration of option
    #Numstds: set the strike of the option
    #T: Threshold to define index 
    #Tin, Pin: Temperature and Precipitation Monte Carlo 
    #index type and option Type: 'CDD', 'HDD', 'Precip'
    Deltas = Dividend[0]
    delta_0 = Dividend[1]
    if indexType == "CDD":
        CDDs = np.maximum(Tin[t_start:t_start+dt.timedelta(days=tau)] - T,0)
        meanCDD = np.maximum(Tin[t_start:t_start+dt.timedelta(days=tau)] - T,0).sum().mean()
        stdCDD = np.maximum(Tin[t_start:t_start+dt.timedelta(days=tau)] - T,0).sum().std()
        K =  meanCDD + NumStds*stdCDD
    if indexType == "HDD":
        HDDs = np.maximum(-1*Tin[t_start:t_start+dt.timedelta(days=tau)] + T,0)
        meanHDD = np.maximum(-1*Tin[t_start:t_start+dt.timedelta(days=tau)] + T,0).sum().mean()
        stdHDD = np.maximum(-1*Tin[t_start:t_start+dt.timedelta(days=tau)] + T,0).sum().std()
        K =  meanHDD + NumStds*stdHDD
    if indexType == "Precip":
        Precip = np.maximum(Pin[t_start:t_start+dt.timedelta(days=tau)],0)
        meanPrecip = Precip.sum().mean()
        stdPrecip = Precip.sum().std()
        K =  meanPrecip + NumStds*stdPrecip
    disc = np.e**(-rho*tau)*(delta_0**(-1*gamma))
    if optionType == "call" and indexType== "CDD":
        CDDscall = np.e**(-rho*tau)*np.maximum(CDDs.sum() - K,0) - disc*(((Deltas[tau,:]**gamma)*np.maximum(CDDs.sum() - K,0))).mean()
        return CDDscall
    if optionType == "put" and indexType== "CDD":
        CDDsput = np.e**(-rho*tau)*np.maximum(K-CDDs.sum(),0) - disc*((Deltas[tau,:]**gamma)*np.maximum(K-CDDs.sum(),0)).mean()
        return CDDsput
    if optionType == "call" and indexType== "HDD":
        HDDscall = np.e**(-rho*tau)*np.maximum(HDDs.sum() - K,0) - disc*((Deltas[tau,:]**gamma)*np.maximum(HDDs.sum() - K,0)).mean()
        return HDDscall
    if optionType == "put" and indexType== "HDD":
        HDDsput = np.e**(-rho*tau)*np.maximum(K-HDDs.sum(),0) - disc*((Deltas[tau,:]**gamma)*np.maximum(K-HDDs.sum(),0)).mean()
        return HDDsput  
    if optionType == "call" and indexType== "Precip":
        Precipcall = np.e**(-rho*tau)*np.maximum(Precip.sum() - K,0) - disc*((Deltas[tau,:]**gamma)*np.maximum(Precip.sum()-K,0)).mean()
        return Precipcall
    if optionType == "put" and indexType== "Precip":
        Precipput = np.e**(-rho*tau)*np.maximum(K-Precip.sum(),0) - disc*((Deltas[tau,:]**gamma)*np.maximum(K-Precip.sum(),0)).mean()
        return Precipput     
    return -1




# In[27]:

def MakeUniverse(t_start, tau, N_intervals, NumStd, N_strikes, N_taus, indextypes, optiontypes, Tin, Pin, Dividend, gamma, rho):
    
    t_starts = [t_start + dt.timedelta(days=int(x)) for x in list(np.linspace(0,tau,N_intervals).astype(int))]
    t_starts = t_starts[:-1]
    taus = list(np.diff(np.linspace(0,tau,N_intervals).astype(int)))
    taus = np.median(taus)
    taus = [taus*(x+1) for x in range(N_taus)]
    strikes = np.linspace(-NumStd,NumStd,N_strikes)
    ind = pd.MultiIndex.from_product([t_starts,taus,strikes,indextypes,optiontypes],names = ['tstart', 'tau', 'strike','index','option'])
    dV = pd.DataFrame(np.zeros([len(Tin.columns),len(ind)]), index = Tin.columns, columns = ind)
    V0 = pd.Series(np.zeros(len(ind)), index = ind)
    
    T = 18; #threshold
    
    for i in range(dV.shape[1]):
        dV.iloc[:,i] = priceCaoWeidV(dV.columns[i][0], int(dV.columns[i][1]), dV.columns[i][2],
                                     T, Tin, Pin, dV.columns[i][3], dV.columns[i][4], Dividend, gamma, rho)
        V0[i] = priceCaoWeiV0(V0.index[i][0], int(V0.index[i][1]), V0.index[i][2],
                                     T, Tin, Pin, V0.index[i][3], V0.index[i][4], Dividend, gamma, rho)    
    return [dV, V0]