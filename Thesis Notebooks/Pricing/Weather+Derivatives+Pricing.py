
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pricing_functions as prc
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


#### 0. Pricing Functions
#### 1. Actuarial Pricing of Temperature Derivatives: This is the most crude way to price weather derivatives

# In[17]:
t_start = full_data.index[-1]
tau = 365
K = 40
alpha = 1
T = 18
r = 0.06/252
kappa = 0
prc.priceOption(t_start, tau, K, T, Tin,kappa,r)  
 
# ### Consumption Pricing: Cao & Wei
# #### Assumption 1: I am going to use the temperature process from Benth et al
# #### Assumption 2: Investor Utility given by: $U(c_t,t) = e^{-\rho t}\frac{c_t^{\gamma+1}}{\gamma + 1}$
# #### Assumption 3: The dividend follows: $ln(\delta_t) = \alpha + \mu\ ln(\delta_{t-1}) + \nu_t$
# ##### with $\nu_t = \sigma \epsilon_t + \sigma [\frac{\phi}{\sqrt{1-\phi^2}}\zeta_t + \eta_1\zeta_{t-1}+...+\eta_m\zeta_{t-m}]$

# #### Contemporaneous Correlations only

# In[18]:
mu = 0.9
m = 0 #number of lagged correlations
phi = 0.25
sigma = 0.2/(252*(1+(phi**2)/(1-phi**2)))
rho = r
alpha = 0.1
gamma = -40
delta_0 = 3


# In[19]:
Dividend = prc.dividend_process(mu, m, phi, alpha, gamma, t_start, tau, r, delta_0, eta)

NumStds = 0
indexType = "CDD"
optionType = "call"

prc.priceCaoWeiV0(t_start, tau, NumStds, T, Tin, Pin, indexType, optionType, Dividend, gamma, rho)

prc.priceCaoWeidV(t_start, tau, NumStds, T, Tin, Pin, indexType, optionType, Dividend, gamma, rho)

#%%
#indextypes = ["HDD","CDD","Precip"]
#optiontypes = ["call", "put"]
indextypes = ["HDD","CDD","Precip"]
optiontypes = ["call"]
N_intervals = 8
N_strikes = 5
NumStd = 2
N_taus = 2

[dV, V0] = prc.MakeUniverse(t_start, tau, N_intervals, NumStd, N_strikes, N_taus, 
             indextypes, optiontypes, Tin, Pin, Dividend, gamma, rho)
    

# In[28]:
dV.to_pickle("C:\\Users\\islipd\\Documents\\Thesis Notebooks\\dV.pkl")
V0.to_pickle("C:\\Users\\islipd\\Documents\\Thesis Notebooks\\V0.pkl")


