# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 11:03:16 2019

@author: Arfor
Periodic and Open Dimerized chain
"""
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
#%%
def e_periodic_dimer(k, t1, t2, p):
    """
    Analytic solution to the eigen-energies in a chain with PBC
    
    Parameters
    ----------
    t1,t2 : Hopping parameters
    
    p : periodicity of dimers
    """
    e1=np.sqrt(t1**2 + t2**2 + 2*t1*t2*np.cos(k*p))
    e2=-np.sqrt(t1**2 + t2**2 + 2*t1*t2*np.cos(k*p))
    return e1, e2

def hop2_matrix(N,t1,t2, periodic = 0):
    D = int(N/2) #number of complete dimers we can make from 8 atoms
    m = np.zeros((N,N)); #create complete matrix

    for i in range(D):
    # fill in the t1's first
        m[(2*i),(2*i)+1]=-t1
        m[(2*i)+1,(2*i)]=-t1
    # now fill in the t2's
    for j  in range(1,D): #because t2 only comes into play after the first dimer
        m[(2*(j)), (2*(j))-1]=-t2 
        m[(2*j)-1, (2*j)]=-t2
    
    if periodic == True:
        m[0,N-1]=-t2
        m[N-1,0]=-t2
    return m
#%% Dimerized Chain
a=2
t1=1
t2=4
N=20

Brill1 = np.linspace(-np.pi/(a), np.pi/(a), num=N/2+1)

peri = e_periodic_dimer(Brill1,t1, t2,a) #periodic exact solution

m = hop2_matrix(N,t1,t2,0) #Open chain solution
eigenValues, eigenVectors = LA.eigh(m) #returns eigenvectors as column vectors

idx = eigenValues.argsort()[::-1]   
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:,idx]
fig = plt.figure()

plt.plot(Brill1,peri[0], 'o', Brill1,peri[1],'o', color="red")
#ax1.title("1st Brillouin Zone")
for i in eigenValues:
    plt.axhline(y=i, color="gray", lw=1.1)
#ax2.plot(e, 'o', color='blue')
plt.show()

#%% Plotting the probability density in the spatial basis
fig, ax = plt.subplots(int(N/4),4, sharex='col', sharey='row', figsize=(15,10))
fig.suptitle("Probability Density $|\psi(x)|$ of Eigenvectors Open Chain (N=20, t'/t=5)", fontsize=23)
fig.subplots_adjust(hspace=0.2, wspace = 0.1)
v_max = np.abs(np.max(eigenVectors))**2
for j in range(4):
    for i in range(int(N/4)):
        ax[i,j].plot(np.linspace(1,N,N),np.abs(eigenVectors[:,(4*j)+i])**2)
        ax[i,j].set_xlim(1,N)
        ax[i,j].set_ylim(0,0.5)
        
        textstr = "$\epsilon = ${:.1f}".format(eigenValues[(4*j)+i])
        ax[i,j].text(0.5,0.8, textstr,verticalalignment='center', horizontalalignment='center', fontsize=19,  transform=ax[i,j].transAxes)
        ax[i,j].tick_params(labelsize=15)
        
#%%Plot the probability amplitude of the localized edge states
localStates = [eigenVectors[:,i] for i in range(len(eigenValues)) if np.round(eigenValues[i], decimals=1)==0.0]
fig, ax = plt.subplots(1,2, sharex='col', sharey='row', figsize=(12,5))
fig.suptitle("Probability Amplitude of Localized states(N={:d}, t'/t={:.0f})".format(N,t2/t1), fontsize=23)
fig.subplots_adjust(hspace=0.2, wspace = 0.1)
v_max = np.max(localStates)
for j in range(2):
        ax[j].plot(np.linspace(1,N,N),localStates[j])
        ax[j].set_xlim(1,N)
        ax[j].set_ylim(-v_max,v_max)
        ax[j].tick_params(labelsize=15)

     
#%%Plot the Spectrum
fig, ax = plt.subplots(figsize=(10,6))

fig.suptitle("Spectrum of a Periodic Dimerized Chain", fontsize=17)
ax.plot(Brill1,peri[0],'o--', color="red", lw=0.5, label="Periodic BC")
ax.plot(Brill1,peri[1],'o--', color="red", lw=0.5)
#    plt.axhline(y=eigenValues[0], color="gray", lw=1.1, label = "Open BC")
#    for i in eigenValues[1:]:
#        plt.axhline(y=i, color="gray", lw=1.1)
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
ax.set_xlim(-np.pi/a, np.pi/a);
textstr = '\n'.join((
        r"N={}".format(N),
    r'$t={:g}$'.format(t1),
    r"$t'={:g}$".format(t2)))
ax.text(1.01,1, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top')
#leg = ax.legend(frameon=False, fontsize=13, bbox_to_anchor=(1.23, 0.58))


#%%

path_save = r"C:\Users\Arfor\Documents\Arfor\Paris\Université Paris-Saclay\Semester 2\Internship 1 - Space Dependent Spin Orbit Coupling\figures\\"
fig.savefig(path_save + "vectors_dimer_N{}_5.png".format(N,t1/t2),bbox_inches='tight', dpi=300)

#fig.savefig(path_save + "spectrum_dimer_N{}_t_t'={:g}.png".format(N,t1/t2),bbox_inches='tight', dpi=300)
