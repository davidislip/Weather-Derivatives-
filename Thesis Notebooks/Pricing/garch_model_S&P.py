''' In this script we fit a GARCH(1,1) model to the realized time series of
 Apple stock returns imposing the constraint: a+b<=gamma, for different
 values of gamma in [0.4, 1].
 Then we plot the maximum likelihood as a function of gamma.
 for gamma very  to 1 the maximum likelihood profile appears almost
 flat, therefore we suggest to impose the stationarity condition a+b<1 when
 fitting the model. 
'''

# For details, see here: https://www.arpm.co/lab/redirect.php?permalink=ExerRElMLEsda_copy(2)
# Prepare the environment


from numpy import arange, ones, var, array
from numpy import sum as npsum
import datetime as dt 
from scipy.io import loadmat
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, ylabel, \
    xlabel

plt.style.use('seaborn')

import pandas as pd
import numpy as np
from Garch_Projection import MonteCarlo_Garch, compute_squared_sigmas, price_option_garch
from FitGARCHFP import FitGARCHFP
from  InvarianceTestEllipsoid import InvarianceTestEllipsoid
from autocorrelation import autocorrelation
#%%
# Upload daily stock prices from db_Stocks
sp = pd.read_csv("C:\\Users\\islipd\\Documents\\Thesis Notebooks\\Pricing\\SPX_index.csv",index_col="Date",parse_dates=True)

StocksSPX = sp['Adj Close']['1990-05-03':'2017-12-30']

# Pick data for Apple, compute the compounded returns from dividend-adjusted stock prices

dx = np.expand_dims(np.array(np.log(StocksSPX).diff().dropna()),axis=0) # Apple returns
date = StocksSPX.index[1:]

t_ = dx.shape[1]

# GARCH(1,1) fit

# initialize sigma**2 with a forward exponential smoothing
lam = 0.7
sig2_0 = lam*var(dx,ddof=1) + (1 - lam)*npsum((lam ** arange(1,t_+1)) * (dx ** 2))

# starting guess for the vector of parameters [c,a,b]
p0 = [0.7, .1, .2]

# constraint: a+b <= gamma
# gamma_grid=0.8:0.0range(1)
gamma_grid = arange(0.4,1.03,0.03)

# constant flexible probabilities
FP = ones((1, t_)) / t_

# fit
[par, _, _, lik] = FitGARCHFP(dx, sig2_0, p0, gamma_grid)

# Figure

figure()
plot(gamma_grid, lik, lw=1.5)
ylabel('log-likelihood')
xlabel('$\gamma$(constraint: a + b $\leq$ $\gamma$)')
plt.xlim([min(gamma_grid),max(gamma_grid)]);
#save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

#%%
#volatility series estimation
 #                 par0[0]: guess for c
 #                 par0[1]: guess for a
 #                 par0[2]: guess for b
 #                 par0[3]: guess for mu
theta = [par[0,-1], par[1,-1], par[2,-1],par[3,-1]]
#theta = [par[0,-1], par[1,-1], par[2,-1],0]
sigma2_t = compute_squared_sigmas(dx, sig2_0, theta,t_)
resid = np.divide(dx,sigma2_t**0.5)

#%%
#monte carlo simulation of the paths 
tau = 1000 
N = 1000 # number of paths to simulate

option_data = pd.read_csv("C:\\Users\\islipd\\Documents\\Thesis Notebooks\\Pricing\\SPX_OptionsData.csv",parse_dates=True)
startd = min(option_data[' DataDate'])
endd = max(option_data['Expiration'])
[y_proj, x_proj, sigma2_proj, dx_proj,date_proj] = MonteCarlo_Garch(startd, endd, N, date, theta, StocksSPX,dx, resid, sigma2_t)
#%%
# plot of figure of the projection cone using the generated paths 
fig = plt.figure(figsize=(12,6))
plt.xlabel('Time')
plt.ylabel('S&P index value')
plt.title('S&P Projection')
plt.plot(date[date<=startd].to_pydatetime(),StocksSPX[date[0]:startd],'b-');
plt.plot(date_proj.to_pydatetime(),y_proj,'g--',alpha=0.05);
#%%
lag_ = 10  # number of lags (for auto correlation test)
acf = autocorrelation(resid, lag_)
lag = 10 # lag to be printed
ell_scale = 2  # ellipsoid radius coefficient
fit = 1  # normal fitting
fig = plt.figure(figsize=(12,6))
InvarianceTestEllipsoid(resid, acf[0,1:], lag, fit, ell_scale);

#%%
#export the variables for use in other scripts
pd.DataFrame(par).to_pickle("garchpar.pkl")
pd.DataFrame(y_proj).to_pickle("garch_SPX_projections.pkl")
pd.DataFrame(x_proj).to_pickle("garch_SPX_logret.pkl")
pd.DataFrame(sigma2_t).to_pickle("garch_sigma2t.pkl")
pd.DataFrame(date_proj).to_pickle("proj_dates.pkl")
#%%
