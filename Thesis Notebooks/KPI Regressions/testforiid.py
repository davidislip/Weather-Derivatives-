# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 17:50:21 2017

@author: islipd
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#testing for iidness
def iid(invariants):
    """takes in a series and does iid analysis on the time series"""
    plt.figure(1)
    invariants = invariants.fillna(0)
    length = len(invariants)
    first_half = invariants[0:length//2]
    second_half = invariants[length//2+1:length]
    l = pd.concat([first_half, second_half], axis = 1)
    l.columns = ['1st', '2nd']
    plt.subplot(2,2,1)
    first_half.hist(bins = 8)
    plt.title('First Half of Series Histogram')
    plt.ylabel("Count")
    plt.subplot(2,2,2)
    second_half.hist(bins = 8)
    plt.title('Second Half of Series Histogram')
    plt.ylabel("Count")
    plt.subplot(2,2,3)
    pd.tools.plotting.lag_plot(invariants)
    plt.title('Lag Plot')
    print(np.corrcoef(first_half,second_half))
    plt.subplot(2,2,4)
    plt.plot(invariants,"*")
    plt.title('Scatter Plot')
    plt.ylabel("Residuals")
    plt.figure(2)
    from pandas.tools.plotting import autocorrelation_plot
    autocorrelation_plot(invariants)
    plt.title('Autocorrelation Plot')

def mahal_plot(e):
  first_half = e[1:len(e)  - 1]
  second_half = e[2:len(e)]
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

  subfig1.set_title("Mahalanobis distances of the iid invariants:")

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
                                 cmap=plt.cm.YlOrBr_r, color = 'red',linewidth = "3")

  subfig1.legend([emp_cov_contour.collections[1], robust_contour.collections[1],
                inlier_plot],
               ['MLE dist', 'robust dist', 'kpis'],
               loc="upper right", borderaxespad=0)
  print(np.corrcoef(first_half,second_half))
  return (robust_cov, emp_cov)