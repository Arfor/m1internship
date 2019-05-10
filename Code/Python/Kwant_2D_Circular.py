# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:03:44 2019

@author: Arfor
"""
import kwant
import numpy as np
import tinyarray as tiny
from types import SimpleNamespace
import scipy.sparse.linalg as la

sx = tiny.array([[0,1],[1,0]])
sy = tiny.array([[0,-1j],[1j,0]])
sz = tiny.array([[1,0],[0,-1]])
pauli = [sx,sy,sz] #pauli = {'x':sx, 'y':sy, 'z':sz} #as dictionary

#%%

sys = kwant.Builder()
sqlat = kwant.lattice.square()

def disk(position):
    x,y = position
    return x**2 + y**2 < 5**2

sys[sqlat.shape(disk,(0,0))] = 4

#%%
sys[sqlat.neighbors(1)] = 1
#sys[kwant.HoppingKind((1,1),sqlat,sqlat)]=1
#sys[kwant.HoppingKind((0,1),sqlat,sqlat)]=0
kwant.plot(sys)

ok = sys.finalized()
ham =  ok.hamiltonian_submatrix(sparse=True)
ham_np = ok.hamiltonian_submatrix()
#%%
eVal_sp = la.eigsh(ham,k=10)
eVal_np,eVec_np = np.linalg.eigh(ham_np)
#%%








