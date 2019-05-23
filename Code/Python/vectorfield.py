# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:19:39 2019

@author: Arfor
"""

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from types import SimpleNamespace
import matplotlib.pyplot as plt
import numpy as np

def profile(x,y,params):
    p = params.azi_winding
    q = params.radi_winding
    R = params.radius
    r = np.sqrt(x**2 + y**2)
    t = np.arctan(y/x)

#    if x != 0:
#        t = np.arctan(y/x)
#    else:
#        t = np.arctan(10E6*y)
    B = [np.sin(2*np.pi*q*(r/R))*np.cos(2*np.pi*q*t), np.sin(2*np.pi*q*(r/R))*np.sin(q*t), np.cos(2*np.pi*q*(r/R))]
    return B

#%%
params = SimpleNamespace(radi_winding=1,azi_winding=2,radius=10)
profile(1,1,params)
#%%
# Make the grid
x, y,z= np.meshgrid(np.arange(-10,10,1.1),np.arange(-10,10,1.1),0)

# Make the direction data for the arrows
u,v,w = profile(x,y,params)
#%%
fig = plt.figure(figsize=(12,12))
ax = fig.gca(projection='3d')

ax.quiver(x, y, 0, u, v, w, normalize=True, pivot='middle', bounds=((-10,10),(-10,10),(-1,1)))

plt.show()
#%%
fig.savefig(r'C:\Users\Arfor\Documents\Arfor\Paris\Université Paris-Saclay\Semester 2\Internship 1 - Skyrmion Quantum Transport\hoi.png',dpi=300)