import time
import numpy as np 
from numpy import linalg as LA
from matplotlib import pyplot as plt

def magn_texture(position,azi_winding, radi_winding):
    x,y = position
    theta = np.arctan2(x,y)
    q = azi_winding
    p = radi_winding
    R = radius
    r = np.sqrt(x**2 + y**2)
    B = [np.sin(np.pi*p*(r/R))*np.cos(q*theta), np.sin(np.pi*p*(r/R))*np.sin(q*theta), np.cos(np.pi*p*(r/R))]
    return B

def get_spectrum(system, params, timing=False, plot=False):
    #system has to be a finalized Kwant system
    pars = dict(t=1, mu=-0.1, j=0.2, delta=0.2, azi_winding=1, radi_winding=1)
    ham = system.hamiltonian_submatrix(params=pars)
    
    if timing:        
        t1 = time.time()
        eVal =LA.eigvalsh(ham)
        t2 = time.time()
        print('Hamiltonian size = {0:d}x{0:d} \nSolving eigenvalues took {1:.3f}s'.format(len(eVal),t2-t1))
    else:
        eVal =LA.eigvalsh(ham)
    
    if plot:
        plt.plot(eVal)

    return eVal

def track_spectrum(system, params, variable = None, values=None, gap_n=6, bulk_n=0, timing=False, plot=False):
    #system has to be a finalized Kwant system
    if not isinstance(variable, str):
        raise Exception('Specify which parameter in params to vary using it\'s dictionary key (string).')
    try:
        iter(params)
    except TypeError:
        raise Exception('Values have to be iterable: List or Array preferential')          
    
    size = len(system.hamiltonian_submatrix(params=params))
    select_evals = np.arange(size//2-gap_n,size//2+gap_n)
    if bulk_n != 0:
        select_evals = np.append(np.arange(0, size//2-1-gap_n, step=(size//2-gap_n)//bulk_n), \
                                 np.append(select_evals, np.sort(np.arange(size//2,size)[::-(size//2)//bulk_n])))           
    spec = []
    for v in values:
        params[variable]=v
        ham = system.hamiltonian_submatrix(params=params)
        eVal = LA.eigvalsh(ham)
        for i in select_evals:
            spec.append(eVal[i])
                                               
    spec=np.reshape(np.asarray(spec), (len(values),len(select_evals)))

    if plot:
        fig, ax = plt.subplots(figsize=(10,6))
        fig.suptitle("Spectrum({})".format(variable), fontsize=17)
        for i in range(len(select_evals)):
            ax.plot(values,spec[:,i])
            ax.set_xlabel('{}'.format(variable), fontsize=15)
            ax.set_ylabel('$\epsilon$', fontsize=15)
    
    return spec