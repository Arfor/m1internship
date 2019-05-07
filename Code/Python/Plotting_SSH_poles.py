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

def format_func(value, tick_number):
    # find number of multiples of pi/2
    N = int(np.round(2 * value*a / (np.pi)))
    if N == 0:
        return "0"
#    elif N == 1:
#        return r"$\frac{\pi}{%.1da}$"%(2*a)
    elif N == 2:
        return r"$\frac{\pi}{%.1da}$"%(a)
#    elif N == -1:
#        return r"-$\frac{\pi}{%.1da}$"%(2*a)
    elif N == -2:
        return r"-$\frac{\pi}{%.1da}$"%(a)
#    elif N % 2 > 0:
#        return r"${0}\pi/2$".format(N)
#    else:
#        return r"${0}\pi$".format(N // 2)
#%% Dimerized Chain
a=2
t = [(1,0),(1,0.5),(1,1),(0.5,1),(0,1)]
t1,t2=t[0]
N=20

Brill1 = np.linspace(-np.pi/(a), np.pi/(a), num=100)

peri = e_periodic_dimer(Brill1,t1, t2,a) #periodic exact solution

m = hop2_matrix(N,t1,t2,0) #Open chain solution
eigenValues, eigenVectors = LA.eigh(m)

idx = eigenValues.argsort()[::-1]   
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:,idx]
fig = plt.figure()

plt.plot(Brill1,peri[0], Brill1,peri[1], color="red")
#ax1.title("1st Brillouin Zone")
for i in eigenValues:
    plt.axhline(y=i, color="gray", lw=1.1)
#ax2.plot(e, 'o', color='blue')
plt.show()

#%%
fig, ax = plt.subplots(1,5, sharey=True, figsize=(16,3))
#fig.suptitle("", fontsize=17)
fig.subplots_adjust(hspace=0.2, wspace = 0.2)

for j in range(5):
        t1,t2 = t[j]
        textstr = '\n'.join((
        r'$t={:g}$'.format(t1),
        r"$t'={:g}$".format(t2)))
        ax[j].text(-0.3,0.38, textstr, fontsize=14,
        verticalalignment='top')
        peri = e_periodic_dimer(Brill1,t1, t2,a) #periodic exact solution
        #plot the spectrum on the first row
        ax[j].plot(Brill1,peri[0], Brill1,peri[1], color="red")

        
        #solve the open chain    
        m = hop2_matrix(N,t1,t2,0)
        eigenValues, eigenVectors = LA.eigh(m)
        idx = eigenValues.argsort()[::-1]   
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        v_max = np.abs(np.max(eigenVectors))**2
        #add the discrete eigenvalues
        for e in eigenValues:
            ax[j].axhline(y=e, color="gray", lw=0.9) 
        ax[j].xaxis.set_major_locator(plt.MultipleLocator(np.pi/(2*a)))
        ax[j].xaxis.set_minor_locator(plt.MultipleLocator(np.pi/(4*a)))
        
        
        ax[j].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        ax[j].tick_params(labelsize=15)
        ax[j].set_xlim(-np.pi/a, np.pi/a);
            
ax[0].set_ylabel('$\epsilon(k)$', fontsize=15)
ax[2].set_xlabel('$k$', fontsize=15)

#leg = ax.legend(frameon=False, fontsize=13, bbox_to_anchor=(1.23, 0.58))
#%% Plot how the factors in front of the spins matrices change when swept over k
def vectors(k,t1,t2,a):
    x = np.round(-t1-t2*np.cos(2*k*a),5)      
    y = np.round(-t2*np.sin(2*k*a),5)  
    return x,y

fig, ax = plt.subplots(1,5, sharey=True,figsize=(16,3))
#fig.suptitle("", fontsize=17)
fig.subplots_adjust(hspace=0.2, wspace = 0.2)

for j in range(5):
        t1,t2 = t[j]
        textstr = '\n \n'.join((
        r'$t={:g}$'.format(t1),
        r"$t'={:g}$".format(t2)))
        ax[j].text(0,0, textstr, fontsize=14,
        horizontalalignment='center',verticalalignment='center')
        
        sx,sy = vectors(Brill1, t1,t2,a)
        ax[j].plot(sx,sy,color="red")
        ax[j].set_xlim(-1.1,1.1)
        ax[j].set_ylim(-1.1,1.1)
        ax[j].axhline(0,color= 'gray',lw=0.8)
        ax[j].axvline(0,color= 'gray',lw=0.8)
        ax[j].plot(0,0,'.',color='black')
        
ax[0].plot(-1,0, 'o',color="red")
ax[0].set_ylabel('$\sigma_y$', fontsize=15,rotation=0)
ax[2].set_xlabel('$\sigma_x$', fontsize=15)

#leg = ax.legend(frameon=False, fontsize=13, bbox_to_anchor=(1.23, 0.58))

#%%

path_save = r"C:\Users\Arfor\Documents\Arfor\Paris\Université Paris-Saclay\Semester 2\Internship 1 - Space Dependent Spin Orbit Coupling\figures\\"
fig.savefig(path_save + "sy_vs_sx'.png",bbox_inches='tight', dpi=300)
