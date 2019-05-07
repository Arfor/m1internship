# -*- coding: utf-8 -*-
"""
Created on Fri May  3 18:37:48 2019

@author: Arfor
"""
from matplotlib import pyplot as plt
import kwant
import numpy as np
from numpy import linalg as LA

sys = kwant.Builder()
#%%
t,d,m = 1.0,1.0,0.2
L = 30
a=1
sz = np.array([[1,0],[0,-1]])
sx = np.array([[0,1],[1,0]])
sy = np.array([[0,-1j],[1j,0]])
lat = kwant.lattice.chain(a)

#%%
def kitaev(L=6,m=1,t=1.0, d = None):
        # On-site Hamiltonian
    if d is None:
        d = t      
    a = 1
    lat = kwant.lattice.chain(a)
    sys = kwant.Builder()
    sys[(lat(i) for i in range(L))] = -m*sz
            # Hopping
    sys[lat.neighbors()] = -t*sz - d*1j*sy

    return sys
#%%
energies = []
evectors= []
mus = np.linspace(0.01,4,10)
for mu in mus:
    sys = kitaev(L=8, m=mu).finalized()
    # Obtain the Hamiltonian as a dense matrix
    ham = sys.hamiltonian_submatrix()
    #diagonalize
    eVal,eVec = LA.eigh(ham)
    energies.append(eVal)
    evectors.append(eVec)
#%%
spec=np.asarray(energies)
for i in range(spec.shape[1]):
    plt.plot(spec[:,i], '-', color="red")
    plt.xticks(np.arange(0,len(mus),1), np.round(np.linspace(0,max(mus),len(mus)),1))
    plt.xlim((0,len(mus)-1))
plt.axhline(y=0, color="grey", linestyle='--')

#%%
L=8
plt.plot(np.arange(L),abs(eVec[:L,L]))
plt.plot(np.arange(L),abs(eVec[L:,L]), '--')

#%%
evectors[1][:L,L]
