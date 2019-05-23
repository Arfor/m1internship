# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:01:58 2019

@author: Arfor
"""

import kwant
import tinyarray as tiny
import numpy as np
from numpy import linalg as LA
import scipy.sparse.linalg as sla
from types import SimpleNamespace
from matplotlib import pyplot as plt
import time

s_0 = np.identity(2)
s_z = np.array([[1, 0], [0, -1]])
s_x = np.array([[0, 1], [1, 0]])
s_y = np.array([[0, -1j], [1j, 0]])
# pauli = [sx,sy,sz] #pauli = {'x':sx, 'y':sy, 'z':sz} #as dictionary

#define the Pauli matrices 
pauli = SimpleNamespace(s0=np.array([[1., 0.], [0., 1.]]),
                        sx=np.array([[0., 1.], [1., 0.]]),
                        sy=np.array([[0., -1j], [1j, 0.]]),
                        sz=np.array([[1., 0.], [0., -1.]]))

pauli.s0s0 = tiny.array(np.kron(pauli.s0, pauli.s0)) # I(4)
pauli.s0sx = tiny.array(np.kron(pauli.s0, pauli.sx)) # 4x4 \sigma_x
pauli.s0sy = tiny.array(np.kron(pauli.s0, pauli.sy)) # 4x4 \sigma_y
pauli.s0sz = tiny.array(np.kron(pauli.s0, pauli.sz)) # 4x4 \sigma_z
pauli.sxs0 = tiny.array(np.kron(pauli.sx, pauli.s0)) # \tau_x
pauli.sys0 = tiny.array(np.kron(pauli.sy, pauli.s0)) # \tau_y
pauli.szs0 = tiny.array(np.kron(pauli.sz, pauli.s0)) # \tau_z
#%%
#define a Boolean function to shape your system
radius = 12
def disk(position): 
    x,y = position
    return x**2 + y**2 < radius**2

def magn_texture(position,azi_winding, radi_winding):
    x,y = position
    theta = np.arctan2(x,y)
    q = azi_winding
    p = radi_winding
    R = radius
    r = np.sqrt(x**2 + y**2)
    B = [np.sin(np.pi*p*(r/R))*np.cos(q*theta), np.sin(np.pi*p*(r/R))*np.sin(q*theta), np.cos(np.pi*p*(r/R))]
    return B

def onsite(site, t, mu, j, azi_winding, radi_winding, delta): #define a function to determine the onsite energy term of the Hamiltonian
    position = site.pos #site is a class! Apart from real space position contains the type of atom (to which family it belongs, how many orbitals etc)
    B = magn_texture(position,azi_winding,radi_winding) #calculate direction of magnetic field at position (x,y)
#    skyrmion_interaction = j*(B[0]*pauli.s0sx + B[1]*pauli.s0sy + B[2]*pauli.s0sz)
    return 4*t*pauli.szs0 - mu*pauli.szs0 + pauli.szs0*mu + delta*pauli.sxs0
    
def hopping(position1,position2,t): #define the hopping terms in your system
    return -t*pauli.szs0
#%%
sys = kwant.Builder() #initialize your system
sqlat = kwant.lattice.square()

sys[sqlat.shape(disk,(0,0))]= onsite
sys[sqlat.neighbors()]= hopping

sys= sys.finalized()
kwant.plot(sys)
#%%
pars = dict(t=1, mu=0, j=0.2, delta=0, azi_winding=1, radi_winding=1)

ham = sys.hamiltonian_submatrix(params=pars) #is np a variable when calling this function instead of pacakage numpy?

#%%
t1 = time.time()
eVal,eVec =LA.eigh(ham)
t2 = time.time()
print(t2-t1)
#%%
x = [site.pos[0] for site in sys.sites]
y = [site.pos[1] for site in sys.sites]

#%%
prob_dens =np.abs(eVec[600])**2
u_up,u_down,v_up,v_down = prob_dens[0::4], prob_dens[1::4], prob_dens[2::4], prob_dens[3::4]
tot_up = u_up + v_up
tot_down = u_down + v_down

plt.hist2d(x,y, bins=2*radius-1, weights=tot_down, cmap='inferno')
plt.colorbar()


#%%
plt.scatter(x,y,c=tot_down, cmap='inferno_r')
plt.colorbar()

#%%
plt.plot(eVal)


