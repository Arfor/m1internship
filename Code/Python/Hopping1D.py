# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:27:22 2019

@author: Arfor

Hopping in a 1D Chain
"""
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
#%%
def e_periodic_homo(k, t, p):
    #p = periodicity
    e1=-2*t*np.cos(k*p)
    return e1

def e_periodic_dimer(k, t1, t2, p):
    #p = periodicity
    e1=np.sqrt(t1**2 + t2**2 + t1*t2*np.cos(k*p))
    e2=-np.sqrt(t1**2 + t2**2 + t1*t2*np.cos(k*p))
    return e1, e2

def hop1_matrix(N,t, periodic = 0):
    m = np.zeros((N,N));
    for i in range(N-1):
        m[i,i+1]=t
        m[i+1,i]=t
    
    if periodic == True:
        m[0,N-1]=t
        m[N-1,0]=t
    return m
#%%
a=2
Brill1 = np.linspace(-np.pi/(a), np.pi/(a), num=51)
hoi = e_periodic_dimer(Brill1,1, 1, 2)

plt.plot(Brill1,hoi[0])
plt.plot(Brill1,hoi[1])
plt.title("1st Brillouin Zone")
plt.show()
#%% Diagonalise shizzle
N = 10
m = hop1_matrix(N,1, periodic = 1)
e,v = LA.eig(m)

#%% Plotting the eigenvector in the spatial basis
for i in range(10):
    plt.plot(np.linspace(1,N,N),v[:,i])
    plt.title("$\epsilon=%.3f$"%(e[i]))
    plt.show()
#plt.plot(e[i])

#%%Plot of the spectrum
plt.plot(e,'o')









    