
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
x = np.random.rand(dV.shape[1]+1)
V0 = np.array(V0)
deltaV = np.array(dV)
Lss = np.array(Loss)
Beta = 0.99
omega = 10**(-3.5)

c = np.maximum(omega*np.percentile(Lss, 100*Beta), (-1)*omega*np.percentile(Lss, 100*Beta))
C = (-1)*Lss.mean()/4;

# In[236]:
#Cvar optimization 
Cvarcons = ({'type': 'ineq',
         'fun' : lambda x: -1*np.inner(V0,x[1:]) + C, #
         'jac' : lambda x: -1*np.insert(V0,0,0)},
       {'type': 'ineq',
          'fun' : lambda x:-1*Lss.mean() + 1*np.inner(deltaV, x[1:]).mean(),
          'jac' : lambda x: np.insert(1*deltaV.mean(axis = 0),0,0)})
Cvarbnds = ((None,None),)
for i in range(dV.shape[1]):
    Cvarbnds = Cvarbnds + ((0, C),)

# In[237]:

Cvar_res = minimize(CVaR, x , jac=grad_CVaR, bounds = Cvarbnds,
               constraints=Cvarcons, method='SLSQP', options={'disp': True}, 
               args = (Beta, Lss, deltaV,c))


# In[239]:

Lss_opt = Lss - np.inner(deltaV,Cvar_res.x[1:])
varwithout = np.percentile(Lss, 100*Beta)
varwith = np.percentile(Lss_opt, 100*Beta)
print("Var without:" + str(varwithout))
print("Var with:" + str(varwith))
print("% decrease:" + str(100*(varwith-varwithout)/varwithout))
print("Cost of portfolio:" + str(np.inner(V0,Cvar_res.x[1:])))
weights = Cvar_res.x[1:]/(Cvar_res.x[1:].sum());
pd.DataFrame(dV.columns[weights > 0.05])


# In[240]:

plt.figure
#pd.DataFrame([Lss_opt,Loss],index = ["With Derivatives", "Without"]).transpose().plot.hist();
bins = np.linspace(min(np.minimum(Lss_opt,Loss)), max(np.maximum(Lss_opt,Loss)), 100)
plt.hist(-1*Loss, bins, alpha=0.5, label="Without")
plt.hist(-1*Lss_opt, bins, alpha=0.5, label='With Derivatives')

plt.legend(loc='upper left')
plt.show()

# In[242]:


plt.figure
plt.plot(V0*Cvar_res.x[1:])
plt.title('Portfolio Weights');

plot_schedule(dV, weights)

