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

def hop1_matrix(N,t, periodic = 0):
    m = np.zeros((N,N));
    for i in range(N-1):
        m[i,i+1]=-t
        m[i+1,i]=-t
    
    if periodic == True:
        m[0,N-1]=-t
        m[N-1,0]=-t
    return m
#%% Dimerized Chain
a=1
t=1
N = 30

Brill1 = np.linspace(-np.pi/(a), np.pi/(a), num=N+1)

hoi = e_periodic_homo(Brill1,t,a)

plt.plot(Brill1,hoi,'o')

plt.title("1st Brillouin Zone")
plt.show()
#%% Diagonalise shizzle

m = hop1_matrix(N,1, periodic = 0)
e,v = LA.eigh(m)

#%% Plotting the eigenvector in the spatial basis
for i in range(10):
    plt.plot(np.linspace(1,N,N),v[:,i])
    plt.title("$\epsilon=%.3f$"%(e[i]))
    plt.show()

#%%Plot of the continuous spectrum overlayed with the quantized spectrum
fig, ax = plt.subplots(figsize=(10,6))

fig.suptitle("Spectrum Homogeneous Chain N={}".format(N), fontsize=17)
ax.plot(Brill1,hoi,'o', color="red", label="Periodic BC")

plt.axhline(y=e[0], color="gray", lw=1.1, label = "Open BC")
for i in e[1:]:
    plt.axhline(y=i, color="gray", lw=1.1)

ax.set_xlabel('$k$', fontsize=15)
ax.set_ylabel('$\epsilon(k)$', fontsize=15)

ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi/(2*a)))
ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi/(4*a)))

def format_func(value, tick_number):
    # find number of multiples of pi/2
    N = int(np.round(2 * value*a / (np.pi)))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\frac{\pi}{%.1da}$"%(2*a)
    elif N == 2:
        return r"$\frac{\pi}{%.1da}$"%(a)
    elif N == -1:
        return r"-$\frac{\pi}{%.1da}$"%(2*a)
    elif N == -2:
        return r"-$\frac{\pi}{%.1da}$"%(a)
#    elif N % 2 > 0:
#        return r"${0}\pi/2$".format(N)
#    else:
#        return r"${0}\pi$".format(N // 2)
ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
ax.tick_params(labelsize=15)
leg = ax.legend(frameon=False, fontsize=13, bbox_to_anchor=(1.23, 0.58))
ax.set_xlim(-np.pi/a, np.pi/a);

#%%

path_save = r"C:\Users\Arfor\Documents\Arfor\Paris\Université Paris-Saclay\Semester 2\Internship 1 - Space Dependent Spin Orbit Coupling\figures\\"
fig.savefig(path_save + "spectrum_homo_N{}.png".format(N),bbox_inches='tight', dpi=300)


# 



    