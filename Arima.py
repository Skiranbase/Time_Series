#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 18:46:26 2019
@author: kiran
"""
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import statsmodels as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15,6

#importing library and preparing dataset
mylynx_df = pd.read_csv('LYNXdata.csv', header = 0, names = ['year','trappings'], index_col=0)
mylynxts = pd.Series(mylynx_df['trappings'].values, index = pd.DatetimeIndex(data=(tuple(pd.date_range(31/12/1821, periods = 114, freq = 'A-DEC'))), freq= 'A-DEC'))

#Dickey-fuller test
def stationarity_test(mylynxts):
    from statsmodels.tsa.stattools import adfuller
    print('Results of Dickey-Fuller Test:')
    df_test = adfuller(mylynxts, autolag='AIC')
    df_output = pd.Series(df_test[0:4], index=['Test Statistic','p-value','#lags_used','Number of Observation Used'])
    print(df_output)
stationarity_test(mylynxts)

#Arima Model
model = ARIMA(mylynxts, order=(3,0,0))
results_AR = model.fit()
plt.plot(mylynxts)
plt.plot(results_AR.fittedvalues, color='red')

'''
information criteria and resdiuals need to be checked.
'''
#information summary
results_AR.summary()


#residual plot
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = plot_acf(results_AR.resid, lags=20, ax = ax1)

#importing function for nomral distribution
from scipy.stats import norm
plt.figure(figsize=(10,6))
plt.hist(results_AR.resid, bins='auto', density=True, rwidth=0.85, label='residuals') #density true - norm.dist line curve
mu,std = norm.fit(results_AR.resid)
xmin,xmax = plt.xlim()
x = np.linspace(xmin,xmax,100)
p = norm.pdf(x,mu,std)
plt.plot(x,p,'m',linewidth=2)
plt.grid(axis='y',alpha=0.2)
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.title('Residuals 2,0,0 vs Normal Distribution - Mean ='+ str(round(mu,2))+', std ='+str(round(std,2)))
plt.show()
