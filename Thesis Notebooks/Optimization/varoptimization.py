# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 00:37:57 2018

@author: islipd
"""


# coding: utf-8

# # Optimal Weather Derivative Hedge 

# In[231]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import Optimization_functions

# In[232]:

Tin = pd.read_pickle("C:\\Users\\islipd\\Documents\\Thesis Notebooks\\Tout.pkl")

eta = pd.read_pickle("C:\\Users\\islipd\\Documents\\Thesis Notebooks\\Tinnov.pkl")

Pin = pd.read_pickle("C:\\Users\\islipd\\Documents\\Thesis Notebooks\\Pout.pkl")

Loss = pd.read_pickle("C:\\Users\\islipd\\Documents\\Thesis Notebooks\\Loss.pkl")

dV = pd.read_pickle("C:\\Users\\islipd\\Documents\\Thesis Notebooks\\dV.pkl")

V0 = pd.read_pickle("C:\\Users\\islipd\\Documents\\Thesis Notebooks\\V0.pkl")

# In[233]:
#parameter initialization
x = np.random.rand(dV.shape[1])
V0 = np.array(V0)
deltaV = np.array(dV)
Lss = np.array(Loss)
Beta = 0.99
omega = 10**(-9)
b = 0
c = omega*Lss.var()
C = (-1)*Lss.mean()/4;

# In[236]:
#Cvar optimization 
cons = ({'type': 'ineq',
         'fun' : lambda x: -1*np.inner(V0,x) + C, #
         'jac' : lambda x: -1*V0},
       {'type': 'ineq',
          'fun' : lambda x:-1*Lss.mean() + 1*np.inner(deltaV, x).mean(),
          'jac' : lambda x:deltaV.mean(axis = 0)})
bnds = ()
for i in range(dV.shape[1]):
    bnds = bnds + ((0, C),)

# In[237]:

res = minimize(semi_var, x , jac=grad_semi_var, bounds = bnds,
               constraints=cons, method='SLSQP', options={'disp': True}, 
               args = (Lss, deltaV,c,b))


# In[239]:

Lss_opt = Lss - np.inner(deltaV,res.x)
varwithout = Lss.var()
varwith = Lss_opt.var()
print("Variance without:" + str(varwithout))
print("Var with:" + str(varwith))
print("% decrease:" + str(100*(varwithout-varwith)/varwithout))
print("Cost of portfolio:" + str(np.inner(V0,res.x)))
weights = res.x/(res.x.sum());
pd.DataFrame(dV.columns[weights > 0.05])


# In[240]:

plt.figure
#pd.DataFrame([Lss_opt,Loss],index = ["With Derivatives", "Without"]).transpose().plot.hist();
bins = np.linspace(min(np.minimum(Lss_opt,Loss)), max(np.maximum(Lss_opt,Loss)), 20)
plt.hist(-1*Loss, bins, alpha=0.5, label="Without")
plt.hist(-1*Lss_opt, bins, alpha=0.5, label='With Derivatives')

plt.legend(loc='upper left')
plt.show()

# In[242]:


plt.figure
plt.plot(V0*res.x)
plt.title('Portfolio Weights');

plot_schedule(dV, weights)

