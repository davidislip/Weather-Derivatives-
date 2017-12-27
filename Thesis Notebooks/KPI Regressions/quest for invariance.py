# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 10:45:41 2017

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
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn import linear_model
from scipy import stats


#importing the data
kpi = pd.read_excel("productivity.xlsx")
weather = pd.read_excel("weather.xlsx")
weather = weather[weather["Total Precip Flag"] != "T"]
weather = weather.set_index(weather["Date/Time"])
kpi = kpi.set_index(kpi["Date"])
kpi["Date"] = pd.to_datetime(kpi["Date"], format="%m/%d/%Y")
#group the data by date and get the aggregate daily kpis
grouped = kpi.groupby('Date')
kpi = kpi[kpi.index.dayofweek <5]
#unit_kpi = pd.DataFrame(grouped.sum()["Locations Passed"]/grouped.count()['Foreman'])
#add up all the metres drilled work 
unit_kpi = pd.DataFrame((grouped.sum()["Metres Prepped"]+grouped.sum()["Meters Drilled"]+grouped.sum()["Meters Cleaned Up"])/grouped.count()['Foreman'])
unit_kpi["Date/Time"] = unit_kpi.index

unit_kpi.columns = ['kpi', 'Date/Time']

unit_kpi = unit_kpi.dropna()
full_data = pd.merge(unit_kpi,weather,on = "Date/Time")
#Averaging days that have large differences 
#l = list(full_data.kpi)
#for i in range(len(l)-1):
#  if l[i]-l[i+1] > 250:
#    l[i] = l[i] - 0.4*(l[i]-l[i+1]) 
#    l[i+1] = l[i] + 0.4*(l[i]-l[i+1])
#  elif l[i]-l[i+1] < -250:
#    l[i] = l[i] - 0.4*(l[i]-l[i+1])
#    l[i+1] = l[i] + 0.4*(l[i]-l[i+1])
#  if l[i] > 350:
#    l[i] = 0.5*l[i-1] + 0.5*l[i+1]
#  elif l[i] < 100:
#    l[i] = 0.5*l[i-1] + 0.5*l[i+1]
#full_data["kpi"] = l
full_data = full_data[full_data.kpi <500]
full_data = full_data[full_data.kpi >50]
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
    first_half.hist(bins = 8)
    plt.subplot(2,2,2)
    second_half.hist(bins = 8)
    plt.subplot(2,2,3)
    pd.tools.plotting.lag_plot(invariants)
    print(np.corrcoef(first_half,second_half))
    plt.subplot(2,2,4)
    invariants.plot()
    
#%%

first_half = full_data.kpi[1:len(full_data)  - 1]
second_half = full_data.kpi[2:len(full_data)]
X = np.array([first_half, second_half]) 
X = np.transpose(X)                          
# fit a Minimum Covariance Determinant (MCD) robust estimator to data
robust_cov = MinCovDet().fit(X)

# compare estimators learnt from the full data set with true parameters
emp_cov = EmpiricalCovariance().fit(X)


fig = plt.figure()

# Show data set
subfig1 = plt.subplot(1, 1, 1)
inlier_plot = subfig1.scatter(first_half, second_half,
                              color='black', label='daily diff in homes passed')

subfig1.set_title("Mahalanobis distances of a contaminated data set:")

# Show contours of the distance functions
xx, yy = np.meshgrid(np.linspace(plt.xlim()[0], plt.xlim()[1], 800),
                     np.linspace(plt.ylim()[0], plt.ylim()[1], 100))

zz = np.c_[xx.ravel(), yy.ravel()]

mahal_emp_cov = emp_cov.mahalanobis(zz)
mahal_emp_cov = mahal_emp_cov.reshape(xx.shape)
emp_cov_contour = subfig1.contour(xx, yy, np.sqrt(mahal_emp_cov),
                                  cmap=plt.cm.PuBu_r,
                                  linestyles='dashed')

mahal_robust_cov = robust_cov.mahalanobis(zz)
mahal_robust_cov = mahal_robust_cov.reshape(xx.shape)
robust_contour = subfig1.contour(xx, yy, np.sqrt(mahal_robust_cov),
                                 cmap=plt.cm.YlOrBr_r, linestyles='dotted')

subfig1.legend([emp_cov_contour.collections[1], robust_contour.collections[1],
                inlier_plot],
               ['MLE dist', 'robust dist', 'kpis'],
               loc="upper right", borderaxespad=0)
print(np.corrcoef(first_half,second_half))
#%%

full_data = full_data.drop(['Year', 'Month', 'Day', 'Data Quality','Max Temp (°C)', 
                            'Max Temp Flag', 'Min Temp (°C)', 'Min Temp Flag',
                            'Mean Temp Flag',
                            'Heat Deg Days Flag', 
                            'Cool Deg Days Flag','Total Precip Flag',
                            'Snow on Grnd (cm)', 'Snow on Grnd Flag', 'Dir of Max Gust (10s deg)',
                             'Dir of Max Gust Flag', 'Spd of Max Gust (km/h)',
                              'Spd of Max Gust Flag','Total Rain Flag','Total Snow (cm)','Total Snow Flag'],axis =1)
#%% Regression Model 

factors = full_data.dropna()
factors["lnT"] = np.log(1.8*full_data['Mean Temp (°C)']+32)

l = list(weather['Total Precip (mm)'])
S = []  
s = 0
for i in range(len(l)):
  if l[i] >0.1:
    s += l[i]
  else:
    s = 0
  S.append(s)
  
weather["cumPrecip"] = S

# Create linear regression object
factors = pd.merge(factors, weather[["Date/Time","cumPrecip"]], on = "Date/Time")
factors['dt'] = (factors["Date/Time"] - factors["Date/Time"][0]).apply(lambda x: float(x.total_seconds()/86400))
X = factors[["Mean Temp (°C)",'Total Precip (mm)',"cumPrecip"]][1:-30]
X["lag kpi"] = list(factors["kpi"])[0:-31]
Y = factors.kpi[1:-30]

X_test = factors[["Mean Temp (°C)",'Total Precip (mm)',"cumPrecip"]][-30:]
X_test["lag kpi"] = list(factors["kpi"])[-31:-1]
Y_test = factors.kpi[-30:]
# Create linear regression object
regr = linear_model.LinearRegression()
#Train the model 
regr.fit(X, Y)
print('Coefficients: \n', regr.coef_)

e = Y - regr.predict(X)
#%%
# Create linear regression object
factors['dt'] = (factors["Date/Time"] - factors["Date/Time"][0]).apply(lambda x: float(x.total_seconds()/86400))
factors['sin(t)'] = np.sin(factors.dt*np.pi/182.5)
factors['cos(t)'] = np.cos(factors.dt*np.pi/182.5)
X = factors[['sin(t)','cos(t)','Total Precip (mm)',"cumPrecip"]][1:-30]
X["lag kpi"] = list(factors["kpi"])[0:-31]
Y = factors.kpi[1:-30]

X_test = factors[['sin(t)','cos(t)','Total Precip (mm)',"cumPrecip"]][-30:]
X_test["lag kpi"] = list(factors["kpi"])[-31:-1]
Y_test = factors.kpi[-30:]
# Create linear regression object
regr = linear_model.LinearRegression()
#Train the model 
regr.fit(X, Y)
print('Coefficients: \n', regr.coef_)

e = Y - regr.predict(X)
#%%
#Residuals AR(2) model

Ui = pd.DataFrame({'0':list(e[0:-2]),'1':list(e[1:-1])})
U2 = pd.Series(list(e[2:]))

ar2 = linear_model.LinearRegression()

ar2.fit(Ui, U2)

E = U2 - ar2.predict(Ui)
E = E.reshape(len(E),)

#%%
#Residuals AR(1) model

#Ui = pd.DataFrame({'0':list(e[0:-2]),'1':list(e[1:-1])})
U1 = pd.Series(list(e[0:-1])).reshape(-1,1)
U2 = pd.Series(list(e[1:]))

ar1 = linear_model.LinearRegression()

ar1.fit(U1, U2)

E = U2 - ar1.predict(U1)
E = E.reshape(len(E),)
#%%
plt.plot(factors['dt'][-30:], Y_test,  color='black')
plt.plot(factors['dt'][-30:], regr.predict(X_test), color='blue',
         linewidth=3)
plt.plot(factors['dt'][1:-30], Y,  color='black')
plt.plot(factors['dt'][1:-30], regr.predict(X), color='red',
         linewidth=3)
#%%
from pandas.tools.plotting import autocorrelation_plot

autocorrelation_plot(e)
#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs=np.array(X['Mean Temp (°C)']),ys=np.array(X['lag kpi']),zs=np.array(Y))