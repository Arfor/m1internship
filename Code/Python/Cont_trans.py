# -*- coding: utf-8 -*-
"""
Created on Thu May  2 10:34:04 2019

Example of Continuous Deformation

@author: Arfor
"""
import numpy as np
from numpy import linalg as LA
import holoviews as hv
from holoviews import opts
hv.extension('bokeh', 'matplotlib')
from matplotlib import pyplot as plt

#%%
n=7
r=(np.random.rand(n,n)-0.5)*2
H0= (r+r.T)/2
r=(np.random.rand(n,n)-0.5)*2
H1= (r+r.T)/2
a = np.linspace(0,1,50)

#%% Calculate the Hamiltonian as the system continuously changes
data = np.array(())
for x in a:
    H_cont = x*H1 + (1-x)*H0
    eVal, eVec = LA.eigh(H_cont)
    data = np.append(data, eVal, axis=0)
data = np.reshape(data, (len(a),n))
#%%


for i in range(n):
    plt.plot(data[:,i], '-')
plt.axhline(y=0, color="grey", linestyle='--')

