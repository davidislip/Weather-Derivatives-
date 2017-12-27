# -*- coding: utf-8 -*-
"""
Created on Sun May  7 20:53:07 2017

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


#importing the data
kpi = pd.read_excel("productivity.xlsx")
weather = pd.read_excel("weather.xlsx")
weather = weather[weather["Total Precip Flag"] != "T"]
weather = weather.set_index(weather["Date/Time"])
kpi = kpi.set_index(kpi["Date"])
kpi["Date"] = pd.to_datetime(kpi["Date"], format="%m/%d/%Y")
grouped = kpi.groupby('Date')
unit_kpi = pd.DataFrame(grouped.sum()["Locations Passed"]/grouped.count()['Foreman'])
unit_kpi["Date/Time"] = unit_kpi.index

unit_kpi.columns = ['kpi', 'Date/Time']
#%%
#testing for iidness
def iid(invariants):
    """takes in a series and does iid analysis on the time series"""
    invariants = invariants.fillna(0)
    length = len(invariants)
    first_half = invariants[0:length//2]
    second_half = invariants[length//2+1:length]
    l = pd.concat([first_half, second_half], axis = 1)
    l.columns = ['1st', '2nd']
    plt.subplot(2,2,1)
    first_half.hist(bins = 10)
    plt.subplot(2,2,2)
    second_half.hist(bins = 10)
    plt.subplot(2,2,3)
    pd.tools.plotting.lag_plot(invariants)
    print(np.corrcoef(first_half,second_half))
    plt.subplot(2,2,4)
    invariants.plot()
def portVar(z, params):
  
  ar,at,kr,kt = z
  bell, unit = params
  
  temp = at*(kt-unit[0])
  rain = ar*(unit[1]-kr)
  temp[temp < 0] = 0
  rain[rain<0] = 0
  
  rain = rain.reset_index(drop =True)
  bell = bell.reset_index(drop =True)
  temp = temp.reset_index(drop =True)
  line = pd.Series(np.linspace(0, bell.iloc[-1], num = len(bell)))
  line_diff = (bell + temp.cumsum() + rain.cumsum() - line)
  #return (bell + temp.cumsum() + rain.cumsum()).diff().std() - 1*(bell + temp.cumsum() + rain.cumsum()).diff().mean()
  return -line_diff[line_diff < -20000].sum()
  #return (bell + temp.cumsum() + rain.cumsum()).diff().std() - 0*(bell + temp.cumsum() + rain.cumsum()).diff().mean()
def pl(z, params):
  
  ar,at,kr,kt = z
  bell, unit = params
  temp = at*(kt-unit[0])
  rain = ar*(unit[1]-kr)
  temp[temp < 0] = 0
  rain[rain<0] = 0
  return (bell + temp.cumsum() + rain.cumsum()).diff() - (temp.sum()+rain.sum())/len(temp)

  
def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(abs(df.corr()), interpolation="nearest", cmap=cmap)
    plt.title('Weather Correlation Plot')
    tick = np.arange(0,len(df.columns),1)
    ax1.set_xticks(tick)
    ax1.set_yticks(tick)
    ax1.set_xticklabels(list(df.columns),fontsize=8)
    ax1.set_yticklabels(list(df.columns),fontsize=8)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax)

    plt.show()
    
#%%
#merging the weather data with the kpi's
merged = pd.merge(weather, unit_kpi,how = 'outer',on='Date/Time')
#merged = merged[merged["kpi"] >0]
merged = merged.fillna(0)
merged = merged[merged["Date/Time"].dt.dayofweek < 6]
merged = merged[merged["Date/Time"]> kpi["Date"][0]]

del merged["Year"]
del merged["Month"]
del merged["Day"]
del merged["Data Quality"]
del merged['Cool Deg Days (°C)']
del merged['Min Temp Flag']
del merged['Max Temp Flag']
del merged['Mean Temp Flag']
del merged['Heat Deg Days Flag']
del merged['Cool Deg Days Flag']
del merged['Total Rain Flag']
del merged['Total Snow Flag']
del merged['Total Precip Flag']
del merged[ 'Snow on Grnd Flag']#covariance analysis 
del merged['Dir of Max Gust (10s deg)']
del merged['Dir of Max Gust Flag']
del merged['Spd of Max Gust (km/h)']
del merged[ 'Spd of Max Gust Flag']
merged = merged.set_index(merged["Date/Time"])
merged = merged.sort_values("Date/Time")
#%%
#assume 4 homes per person average to break even 
Bell = 1250*(merged['kpi'] - 4).cumsum()


#begin the optimization
Underlying = [merged["Mean Temp (°C)"], merged["Total Precip (mm)"]]
guess = (100,100,5,0)  
bounds = ((0,500), (0,500), (0,20), (-10,5))
result = minimize(portVar, x0 = guess, args = ((Bell,Underlying),), bounds = bounds)
#%%
#plot the optimized portfolio p/l and compare with the non-optimal
z = result.x
print(portVar(z, (Bell,Underlying)))
print(Bell.diff().std())
p= pl(z, (Bell,Underlying))
plt.plot_date(merged["Date/Time"], Bell.diff().cumsum(),'b-')
plt.plot_date(merged["Date/Time"], p.cumsum(),'r-')
#plt.plot_date(merged["Date/Time"], test,'r-')
plt.xlabel("Date") 
plt.ylabel("Unit Returns $ CAD")
plt.title("Unit P/L with and without a Fair Weather Hedge")
#%%
#plotstap water 
merged.plot(kind = 'scatter', x = "Max Temp (°C)", y = 'kpi')

#%%
iid(merged['kpi'].diff())
#%%
plt.plot_date(merged["Date/Time"], merged['kpi'],'b-')
plt.plot_date(merged["Date/Time"], merged['Max Temp (°C)'],'r-')
plt.xlabel("Date") 
plt.ylabel("Max Temp and the Returns of homes passed")

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
del merged["Date/Time"]
#merged["Returns"] = merged["kpi"].diff()

correlation_matrix(merged)

#%%  
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#  
#a = np.arange(0,1000,10)
#k = np.arange(-10,10,1)
#A, K = np.meshgrid(a,k)
#zs = np.array([portVar((x,y),(Bell,Underlying[0])) for x,y in zip(np.ravel(A), np.ravel(K))])
#Z = zs.reshape(A.shape)
#surf = ax.plot_surface(A,K,Z,cmap=cm.coolwarm, linewidth=0, antialiased = False)
#fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.show()















