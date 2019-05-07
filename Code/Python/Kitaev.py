# -*- coding: utf-8 -*-
"""
Created on Thu May  2 16:48:24 2019

@author: Arfor
"""

import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
import holoviews as hv
hv.extension('bokeh')

sz = np.array([[1,0],[0,-1]])
sx = np.array([[0,1],[1,0]])
sy = np.array([[0,-1j],[1j,0]])

def hamil(L,m,t,d):
    diag = np.diag(np.full((L),-m),k=0)
    off = np.diag(np.full((L-1),-t),k=1)
    sc = np.diag(np.full((L-1),-d),k=1) + np.diag(np.full((L-1),d),k=-1)
    return np.kron(sz,diag + off + off.T) + np.kron(1j*sy, sc)

#%%
L,m,t,d = 6,1,1,1

a = np.linspace(0.01,4,10)

data = np.array(())
for m in a:
    H = hamil(L,m,t,t)
    eVal, eVec = LA.eigh(H)
    data = np.append(data, eVal, axis=0)
    #%%
data = np.reshape(data, (len(a),L*2))
#%% Plot the spectrum of the Hamiltonian
for i in range(L*2):
    plt.plot(data[:,i], '-')
    plt.xticks(np.arange(0,len(a),1), np.round(np.linspace(0,max(a),len(a)),1))
    plt.xlim((0,len(a)-1))
plt.axhline(y=0, color="grey", linestyle='--')
