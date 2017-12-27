# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:05:01 2017

@author: islipd
"""
import model_rev0.py
##%%
#plots
merged.plot(kind = 'scatter', x = "Max Temp (°C)", y = 'kpi')

#%%
iid(merged['kpi'].diff())
#%%
plt.plot_date(merged["Date/Time"], merged['kpi'],'b-')
plt.plot_date(merged["Date/Time"], merged['Max Temp (°C)'],'r-')
plt.xlabel("Date") 
plt.ylabel("Max Temp and the Returns of homes passed")
#%%  
fig = plt.figure()
ax = fig.gca(projection='3d')
  
a = np.arange(0,1000,10)
k = np.arange(-10,10,1)
A, K = np.meshgrid(a,k)
zs = np.array([portVar((x,y),(Bell,Unit)) for x,y in zip(np.ravel(A), np.ravel(K))])
Z = zs.reshape(A.shape)
surf = ax.plot_surface(A,K,Z,cmap=cm.coolwarm, linewidth=0, antialiased = False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


#%%
merged.plot(kind = 'scatter', x = "Max Temp (°C)", y = 'kpi')

#%%
iid(merged['kpi'].diff())
#%%
plt.plot_date(merged["Date/Time"], merged['kpi'],'b-')
plt.plot_date(merged["Date/Time"], merged['Max Temp (°C)'],'r-')
plt.xlabel("Date") 
plt.ylabel("Max Temp and the Returns of homes passed")
#%%
#covariance analysis 
#del merged["Date/Time"]
merged["Returns"] = merged["kpi"].diff()

correlation_matrix(merged)
