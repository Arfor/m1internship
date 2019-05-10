# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:03:44 2019

@author: Arfor
"""

import kwant
import numpy as np
import tinyarray as tiny
from types import SimpleNamespace
#%%
sys = kwant.Builder()

#%% Making graphene
graphene = kwant.lattice.general(
        [(1,0),(0.5,0.5*np.sqrt(3))],
        [(0,0),(0.2,0.5)]
        )

a,b =graphene.sublattices
#site = b(5,1)
#site.pos
sys[[a(x,y) for x in range(6) for y in range(5)]]=1
sys[[b(x,y) for x in range(6) for y in range(5)]]=1


#%%
#sys[a.neighbors(1)]=0.5 #hoppinh in same lattice
#sys[b.neighbors(1)]=0.5
sys[kwant.builder.HoppingKind((1,0),a,b)] = 1
kwant.plot(sys)

#%%
lat = kwant.lattice.square()
def ring(pos):
    x, y = pos
    return 7**2 <= x**2 + y**2 <= 13**2
sys[lat.shape(ring, (10, 0))] = 0
sys[lat.neighbors()] = 1
#%%Leads
sym = kwant.TranslationalSymmetry((-2, 1))
lead = kwant.Builder(sym)
lead[(lat(0, x) for x in range(-3, 3))] = 0
lead[lat.neighbors()] = 1
sys.attach_lead(lead)
sys.attach_lead(lead, lat(0, 0))
#%%

kwant.plot(sys)
