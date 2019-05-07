# Tutorial 2.2.2. Transport through a quantum wire
# ================================================
#
# Physics background
# ------------------
#  Conductance of a quantum wire; subbands
#
# Kwant features highlighted
# --------------------------
#  - Builder for setting up transport systems easily
#  - Making scattering region and leads
#  - Using the simple sparse solver for computing Landauer conductance

from matplotlib import pyplot
import kwant
import numpy as np
from numpy import linalg as LA
import scipy.sparse.linalg as sla

# First, define the tight-binding system

sys = kwant.Builder()
#%%
# Here, we are only working with square lattices
a = 1
lat = kwant.lattice.chain(a)
#%%
t,d,m = 1.0,1.0,0.2
L = 30

sz = np.array([[1,0],[0,-1]])
sx = np.array([[0,1],[1,0]])
sy = np.array([[0,-1j],[1j,0]])

def make_system(L=10,m=1,t=1.0):
        # On-site Hamiltonian
    sys = kwant.Builder()
    sys[(lat(i) for i in range(L))] = m
            # Hopping
    sys[lat.neighbors()] = -t
    return sys


#%%
energies = []
param = np.linspace(0.1,1,10)
for p in param:
    sys = make_system(L,m,p).finalized()
    # Obtain the Hamiltonian as a dense matrix
    ham = sys.hamiltonian_submatrix()
    #diagonalize
    eVal,eVec = LA.eigh(ham)
    energies.append(eVal)

#%%

eVal, eVec = LA.eigh(ham)
pyplot.figure()
pyplot.plot(eVal)
#pyplot.xlabel("magnetic field [arbitrary units]")
#pyplot.ylabel("energy [t]")
pyplot.show()