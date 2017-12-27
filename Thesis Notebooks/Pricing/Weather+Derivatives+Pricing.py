
# coding: utf-8

# In[2]:

get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import datetime as dt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from  InvarianceTestEllipsoid import InvarianceTestEllipsoid
from autocorrelation import autocorrelation
from scipy import stats
import scipy.signal as sig
import statsmodels.api as sm

Tin = pd.read_pickle("C:\\Users\\islipd\\Documents\\Thesis Notebooks\\Tout.pkl")

eta = pd.read_pickle("C:\\Users\\islipd\\Documents\\Thesis Notebooks\\Tinnov.pkl")

Pin = pd.read_pickle("C:\\Users\\islipd\\Documents\\Thesis Notebooks\\Pout.pkl")

full_data = pd.read_pickle("C:\\Users\\islipd\\Documents\\Thesis Notebooks\\kpi.pkl")


# ### 0. Pricing Functions

# #### 1. Actuarial Pricing of Temperature Derivatives: This is the most crude way to price weather derivatives

# In[3]:

t_start = full_data.index[-1]
tau = 50
K = 40
alpha = 1
T = 18
r = 0.06/252
kappa = 0
CDDs = np.maximum(Tin[t_start:t_start+dt.timedelta(days=tau)] - T,0)
HDDs = np.maximum(-1*Tin[t_start:t_start+dt.timedelta(days=tau)] + T,0)

def priceOption(t_start, tau, K, T, Tin, kappa):
    
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
    
priceOption(t_start, tau, K, T, Tin,kappa)   


# ### Consumption Pricing: Cao & Wei
# #### Assumption 1: I am going to use the temperature process from Benth et al
# #### Assumption 2: Investor Utility given by: $U(c_t,t) = e^{-\rho t}\frac{c_t^{\gamma+1}}{\gamma + 1}$
# #### Assumption 3: The dividend follows: $ln(\delta_t) = \alpha + \mu\ ln(\delta_{t-1}) + \nu_t$
# ##### with $\nu_t = \sigma \epsilon_t + \sigma [\frac{\phi}{\sqrt{1-\phi^2}}\zeta_t + \eta_1\zeta_{t-1}+...+\eta_m\zeta_{t-m}]$

# #### Contemporaneous Correlations only

# In[6]:

mu = 0.9
m = 0
phi = 0.25
sigma = 0.2/(252*(1+(phi**2)/(1-phi**2)))
rho = r

alpha = 0.1
nu_t = np.array(sigma*(np.random.normal(0, 1, eta.shape) + (phi/(1-phi**2)**0.5)*eta))
gamma = -40
delta_0 = 1


# In[7]:

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
plt.plot(iters)
tol


# ### Time to simulate some dividends

# In[8]:

delta_0


# In[127]:

np.log(Deltas[tau,:]/delta_0).mean()


# In[9]:

nu_t = sigma*(np.random.normal(0, 1, eta.shape) + (phi/(1-phi**2)**0.5)*eta)
nu_t = np.array(nu_t[t_start:t_start+dt.timedelta(days=tau)])


Deltas = np.zeros(nu_t.shape)
Deltas[0,:] = delta_0
for i in range(tau):
    Deltas[i+1,:] = np.e**(alpha + mu*np.log(Deltas[i,:])+nu_t[i,:])
plt.plot(Deltas[:,0:100]);


# ###  Now we take the expectations under $Q$ over each of the simulations to get the premiums
# $C(t,T_1,T_2,X) = e^{-\rho(T_2-t)} \delta_t^{-\gamma}\ \mathbb{E}\big(\delta_{T_2}^{\gamma}q(X,T_1,T2)\big)$

# In[10]:

def priceOptionCaoWei(t_start, tau, K, T, Tin):
    
    CDDs = np.maximum(Tin[t_start:t_start+dt.timedelta(days=tau)] - T,0)
    HDDs = np.maximum(-1*Tin[t_start:t_start+dt.timedelta(days=tau)] + T,0)
    disc = np.e**(-rho*tau)*(delta_0**(-1*gamma))
    CDDscall = disc*(((Deltas[tau,:]**gamma)*np.maximum(CDDs.sum() - K,0))).mean()
    CDDsput = disc*((Deltas[tau,:]**gamma)*np.maximum(K-CDDs.sum(),0)).mean()
    HDDscall = disc*((Deltas[tau,:]**gamma)*np.maximum(HDDs.sum() - K,0)).mean()
    HDDsput = disc*((Deltas[tau,:]**gamma)*np.maximum(K-HDDs.sum(),0)).mean()
    HDDsfutures =  disc*((Deltas[tau,:]**gamma)*HDDs.sum()).mean()
    CDDsfutures =  disc*((Deltas[tau,:]**gamma)*CDDs.sum()).mean()
    ind = ["call", 'put', "futures"]
    cols = ["HDD", 'CDD']
    prices = [(HDDscall,CDDscall), (HDDsput, CDDsput), (HDDsfutures, CDDsfutures)]
    
    return pd.DataFrame(prices, columns = cols, index = ind)
priceOptionCaoWei(t_start, tau, K, T, Tin)

