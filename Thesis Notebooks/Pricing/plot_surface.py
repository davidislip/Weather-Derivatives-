# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 20:20:13 2018

@author: islipd
"""

import numpy as np
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# some 3-dim points

data = np.transpose(np.load('implied_gamma_surface.npy'))
#%%
# regular grid covering the domain of the data
X,Y = np.meshgrid(np.arange(min(data[:,0]), 2*max(data[:,0]), 0.5), np.arange(min(data[:,1]), 2*max(data[:,1]), 0.5))
XX = X.flatten()
YY = Y.flatten()

order = 1   # 1: linear, 2: quadratic
if order == 1:
    # best-fit linear plane
    A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])    # coefficients
    
    # evaluate it on grid
    Z = C[0]*X + C[1]*Y + C[2]
    
    # or expressed using matrix/vector product
    #Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

elif order == 2:
    # best-fit quadratic curve
    A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
    
    # evaluate it on a grid
    Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)
elif order == 4:
    # best-fit quadratic curve
    x = data[:,0]
    y = data[:,1]
    A = np.c_[np.ones(x.shape[0]),x ,y ,x*y ,x**2 ,y**2 ,x*(y**2) ,(x**2)*y ,
              y**3, x**3, x*(y**3), (x**2)*(y**2), (x**3)*y, y**4, x**4]
    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
    
    # evaluate it on a grid
    Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2, 
                     XX*(YY**2) ,(XX**2)*YY ,YY**3, XX**3, XX*(YY**3), 
                     (XX**2)*(YY**2), (XX**3)*YY, YY**4, XX**4], C).reshape(X.shape)
    Z[Z>0] = 0
# plot points and fitted surface
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
ax.scatter(data[:,0], data[:,1], data[:,2], c='r', s=50)
plt.xlabel('Normalized Strike')
plt.ylabel('tau')
ax.set_zlabel('gamma')
ax.axis('equal')
ax.axis('tight')
plt.show()
#%%
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D

tck, u= interpolate.splprep(data, k=5)
#here we generate the new interpolated dataset, 
#increase the resolution by increasing the spacing, 500 in this example
new = interpolate.splev(np.linspace(0,1,500), tck, der=0)

#now lets plot it!
fig = plt.figure()
ax = Axes3D(fig)
ax.plot(data[0], data[1], data[2], label='originalpoints', lw =2, c='Dodgerblue')
ax.plot(new[0], new[1], new[2], label='fit', lw =2, c='red')
ax.legend()
plt.savefig('junk.png')
plt.show()
