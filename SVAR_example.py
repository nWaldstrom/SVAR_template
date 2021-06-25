import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from fredapi import Fred


import matplotlib.pyplot as plt
import matplotlib.pylab as pylab


#%%
import pandas_datareader as pdr
import pandas as pd
import datetime

start = datetime.datetime (1961, 1, 1)
end = datetime.datetime (2018, 1, 1)

df = pdr.DataReader(['GDPC1', 'HOHWMN02USQ065S', 'BOGZ1FL072052006Q', 'CPALTT01USQ661S', 'PCEC', 'FPI'],
                    'fred', start, end)

N = 6
names = ['GDP', 'Labor', 'Interest rate', 'Price index', 'Consumption', 'Investment']
shortnames = ['GDP', 'hours', 'r', 'P', 'C', 'I']


df = df.rename(columns={'GDPC1': 'GDP', 'HOHWMN02USQ065S' : 'hours', 'BOGZ1FL072052006Q' : 'r', 'CPALTT01USQ661S' : 'P', 'PCEC' : 'C', 'FPI': 'I'})
          
df_trans = df.copy()
df_trans['GDP'] = 100 * np.log(df['GDP'])
df_trans['hours'] = 100 * np.log(df['hours'])
df_trans['P'] = 100 * np.log(df['P'])
df_trans['C'] = 100 * np.log(df['C'])
df_trans['I'] = 100 * np.log(df['I'])



#%%

from statsmodels.tsa.api import VAR


model = VAR(df_trans)

results = model.fit(2, trend = 'ctt')
irf = results.irf(40)
# irf.plot(orth=True)

#%%

def compare_irfs(irf1, irf2):
    lstyles = ['-', '--']
    names_list = ['Model 1', 'Model 2']
    irfs_list = [irf1, irf2]
    fig, axs = plt.subplots(2, 3)
    for k in range(2):
        # 1st column is time, 2nd is which variable is shocked, and 3rd column is response variable
        axs[0, 0].plot(irfs_list[k][:,2,0], label=names_list[k], linestyle=lstyles[k])
        axs[0, 1].plot(irfs_list[k][:,2,1], linestyle=lstyles[k])
        axs[0, 2].plot(irfs_list[k][:,2,2], linestyle=lstyles[k])
        axs[1, 0].plot(irfs_list[k][:,2,3], linestyle=lstyles[k])
        axs[1, 1].plot(irfs_list[k][:,2,4], linestyle=lstyles[k])
        axs[1, 2].plot(irfs_list[k][:,2,5], linestyle=lstyles[k])
    
    
    for j in range(round(N/2)):
        axs[0, j].set_title(names[j])
    for j in range(round(N/2)):
        axs[1, j].set_title(names[3+j])
        
    plt.tight_layout()
    plt.show()


#%% alt ordering
names_alt = names[::-1]
shortnames_alt = shortnames[::-1]
df_trans_alt = df_trans[shortnames_alt]

model_alt = VAR(df_trans_alt)
results_alt = model_alt.fit(2, trend = 'ctt')
irf_alt = results_alt.irf(40)

compare_irfs(irf.orth_irfs, np.flip(irf_alt.orth_irfs, axis =2))
plt.tight_layout()
plt.show()



#%%




def IRF_conf(irf1, err_bands):
    lstyles = ['-', '--', '--']
    names_list = ['Model 1', None, None]
    lb = irf1 + err_bands[0]
    ub = irf1 - err_bands[0]
    irfs_list = [irf1, lb, ub]
    fig, axs = plt.subplots(2, 3)
    for k in range(3):
        # 1st column is time, 2nd is which variable is shocked, and 3rd column is response variable
        axs[0, 0].plot(irfs_list[k][:,2,0], label=names_list[k])
        axs[0, 1].plot(irfs_list[k][:,2,1], linestyle=lstyles[k])
        axs[0, 2].plot(irfs_list[k][:,2,2], linestyle=lstyles[k])
        axs[1, 0].plot(irfs_list[k][:,2,3], linestyle=lstyles[k])
        axs[1, 1].plot(irfs_list[k][:,2,4], linestyle=lstyles[k])
        axs[1, 2].plot(irfs_list[k][:,2,5], linestyle=lstyles[k])
    
    
    for j in range(round(N/2)):
        axs[0, j].set_title(names[j])
    for j in range(round(N/2)):
        axs[1, j].set_title(names[3+j])
        
    plt.tight_layout()
    plt.show()

err_bands = results.irf_errband_mc(orth=True, repl=1000, steps=40, signif=0.05, seed=123, burn=150, cum=False)
IRF_conf(irf.orth_irfs, err_bands)





