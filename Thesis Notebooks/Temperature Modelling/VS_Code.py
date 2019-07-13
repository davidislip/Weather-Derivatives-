#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir("C:\\Users\\David\\Workspace\\Weather-Derivatives-\\Thesis Notebooks\\Temperature Modelling")
	print(os.getcwd())
except:
	pass

#%%
import sqlite3
import pandas as pd
import numpy as np
from Model_One import *
from Model_Two import *
get_ipython().run_line_magic('matplotlib', 'inline')
M = 100
tau = 2*365
#conn = sqlite3.connect("C:\\Users\\islipd\\Documents\\Thesis Notebooks\\Temperature Modelling\\Simulations.db")
T = {}
epsi = {}

[T[0], epsi[0]] =  model1(M, tau)

H = 0.6
[T[1], epsi[1]] = model2(M, tau, H)


#%%
H = 0.3
[T[2], epsi[2]] = model2(M, tau, H)


#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot(T[0].iloc[:,0],'g--',alpha = 1);
plt.figure(figsize=(12,6))
plt.plot(T[1].iloc[:,0],'b--',alpha = 1);


#%%
plt.figure(figsize=(12,6))
plt.plot(T[0],'g--',alpha = 0.01);
plt.figure(figsize=(12,6))
plt.plot(T[1],'b--',alpha = 0.01);
plt.figure(figsize=(12,6))
plt.plot(T[2],'r--',alpha = 0.01);


