import time
import numpy as np 
import scipy.sparse.linalg as sla
import numpy.linalg as LA
from matplotlib import pyplot as plt
from types import SimpleNamespace
import kwant
import holoviews as hv

hv.extension('bokeh','matplotlib', logo=False)

dims = dict(kx = 'k_x',ky = 'k_y', mu = 'µ', delta = 'Δ', t = 't', E='ε',mu_t = 'µ/t',E_d='ε/Δ', \
			azi_winding='q', radi_winding= 'p', radius = 'r', R0 = 'R_0', j='j')

pauli = SimpleNamespace(s0=np.array([[1., 0.], [0., 1.]]),
                        sx=np.array([[0., 1.], [1., 0.]]),
                        sy=np.array([[0., -1j], [1j, 0.]]),
                        sz=np.array([[1., 0.], [0., -1.]]))

#extend Pauli matrices to particle-hole space (see e.g. BdG 'trick' paper)
pauli.s0s0 = np.kron(pauli.s0, pauli.s0) # I(4)
pauli.s0sx = np.kron(pauli.s0, pauli.sx) # \sigma_x
pauli.s0sy = np.kron(pauli.s0, pauli.sy) # \sigma_y
pauli.s0sz = np.kron(pauli.s0, pauli.sz) # \sigma_z
pauli.sxs0 = np.kron(pauli.sx, pauli.s0) # \tau_x
pauli.sxsx = np.kron(pauli.sx, pauli.sx)
pauli.sxsy = np.kron(pauli.sx, pauli.sy)
pauli.sxsz = np.kron(pauli.sx, pauli.sz)
pauli.sys0 = np.kron(pauli.sy, pauli.s0) # \tau_y
pauli.sysx = np.kron(pauli.sy, pauli.sx)
pauli.sysy = np.kron(pauli.sy, pauli.sy)  
pauli.sysz = np.kron(pauli.sy, pauli.sz)
pauli.szs0 = np.kron(pauli.sz, pauli.s0) # \tau_z
pauli.szsx = np.kron(pauli.sz, pauli.sx) 
pauli.szsy = np.kron(pauli.sz, pauli.sy)
pauli.szsz = np.kron(pauli.sz, pauli.sz)
sigma = np.moveaxis(np.array([pauli.s0sx,pauli.s0sy, pauli.s0sz]),0,2)

def onsite(site, radius, t, mu, j, azi_winding, radi_winding, delta): #define a function to determine the onsite energy term of the Hamiltonian
    position = site.pos #site is a class! Apart from real space position contains the type of atom (to which family it belongs, how many orbitals etc)
    B = magn_texture(position, radius, azi_winding, radi_winding) #calculate direction of magnetic field at position (x,y)
    return 4*t*pauli.szs0 - mu*pauli.szs0 + delta*pauli.sxs0 + j*np.dot(sigma,B)
    
def hopping(position1,position2,t): #define the hopping terms in your system
    return -t*pauli.szs0

def build_disk(radius=10, plot=False):
    sys = kwant.Builder() #initialize your system
    sqlat = kwant.lattice.square(norbs=2) #needed in order to separate the electron and hole contribution to the scattering matrix (each site has one electron and one hole orbital)

    #define a Boolean function to shape your system
    def disk(position): 
        x,y = position
        return x**2 + y**2 < radius**2

    sys[sqlat.shape(disk,(0,0))]= onsite
    sys[sqlat.neighbors()]= hopping

    if plot:
        system_plot = kwant.plot(sys)
        
    return sys.finalized()

def magn_texture(position, radius=10, azi_winding=1, radi_winding=1):
    x,y = position
    theta = np.arctan2(x,y)
    q = azi_winding
    p = radi_winding
    R = radius
    r = np.sqrt(x**2 + y**2)
    B = [np.sin(np.pi*p*(r/R))*np.cos(q*theta), np.sin(np.pi*p*(r/R))*np.sin(q*theta), np.cos(np.pi*p*(r/R))]
    return B

def get_spectrum(system, params, timing=False, plot=False):
    #system has to be a finalized Kwant system. Uses numpy.linalg
    ham = system.hamiltonian_submatrix(params=params)
    
    if timing:        
        t1 = time.time()
        eVal =np.linalg.eigvalsh(ham)
        t2 = time.time()
        print('Hamiltonian size = {0:d}x{0:d} \nSolving eigenvalues took {1:.3f}s'.format(len(eVal),t2-t1))
    else:
        eVal =LA.eigvalsh(ham)
    
    if plot:
        fig, ax = plt.subplots(figsize=(10,6))
        fig.suptitle("Spectrum", fontsize=17)
        plt.plot(eVal)
        ax.set_xlabel('x', fontsize=15)
        ax.set_ylabel('$\epsilon$', fontsize=15)
        params_box(ax, params)    
        
    return eVal

def track_spectrum(system, params, variable = None, values=None, gap_n=6, bulk_n=0, timing=False, plot=False):
    #system has to be a finalized Kwant system. Uses numpy.linalg
    if not isinstance(variable, str):
        raise Exception('Specify which parameter in params to vary using its dictionary key (string).')
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
    
    t1 = time.time()
    for v in values:
        params[variable]=v
        ham = system.hamiltonian_submatrix(params=params)
        eVal = np.linalg.eigvalsh(ham)
        for i in select_evals:
            spec.append(eVal[i])
    
    if timing:
        t2 = time.time()
        print('Hamiltonian size = {0:d}x{0:d} \nSolving {1:d} times took {2:.3f}s'.format(len(eVal),len(values),t2-t1))
    
    spec=np.reshape(np.asarray(spec), (len(values),len(select_evals)))

    if plot:
        N_eVal = len(select_evals)
        fig, ax = plt.subplots(figsize=(10,6))
        fig.suptitle("Spectrum({})".format(variable), fontsize=17)
        for i in range(N_eVal//2):
            ax.plot(values,spec[:,i], color='blue', alpha=0.5+i/N_eVal)
        for i in range(N_eVal//2, len(select_evals)):
            ax.plot(values,spec[:,i], color='red', alpha=1.5-i/N_eVal)           
        ax.set_xlabel('{}'.format(variable), fontsize=15)
        ax.set_ylabel('$\epsilon$', fontsize=15)
        params_box(ax, params, variable)
    return spec

def params_box(ax, params, variable=None):
    pars = params.copy()
    param_text= ''
    
    if variable is not None:    
        if not isinstance(variable, str):
            raise Exception('Specify which variable is being varied using its dictionary key (string).')
        pars.pop(variable)
        
    for key in pars:
        param_text = param_text + '{} = {}\n'.format(key, pars[key])    
    # place text boxes
    ax.text(1.03, 0.95, param_text, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=dict(facecolor='blue', alpha=0.1))
    

def dos_bulk_2d(radius=30,t=1,mu=0,delta=0,j=0, smooth=True):
    #radius = number of points from kx=0 to kx=pi
    x,y = np.meshgrid(np.linspace(-np.pi,np.pi,2*radius),np.linspace(-np.pi,np.pi,2*radius))
    e_up = j + np.sqrt( (2*t*(np.cos(x)+np.cos(y)-2)-mu)**2 + delta**2 )
    e_down = -j + np.sqrt( (2*t*(np.cos(x)+np.cos(y)-2)-mu)**2 + delta**2 )
    spec = np.asarray([e_up,-e_up, e_down, -e_down]).flatten()
    
    plot_opts=dict(width=350, height=350,invert_axes=True, default_tools=['ypan','wheel_zoom','reset', 'save'], framewise=True)
    style_opts=dict(color='deepskyblue',alpha=0.5)
    fermi_level = hv.VLine(mu).opts(**plot_opts, color='black', line_dash='dashed', line_width=1.5)
    
    if smooth:
        bandwidth = 0.1 + round(15/radius)/40
        plot_opts['bandwidth']=bandwidth
        distr = hv.Distribution(spec, kdims=dims['E']).opts(plot=plot_opts, style=style_opts)
        plot = (distr*fermi_level)
    else:
        n_bins = int(9 + radius/2)
        hist = hv.Histogram(np.histogram(spec, bins=n_bins, density=True), kdims=dims['E'], vdims='Density').opts(plot=plot_opts, style=style_opts)
        plot = (hist*fermi_level)
        
    return plot.opts(title='DoS 2D ({}x{})'.format(radius,radius))

def dos_finite_2d(radius=5,t=1, mu=0, j=0, delta=0, azi_winding=1, radi_winding=1, smooth=True, toolbar=False):
    #L is the radius of the disk
    sys = build_disk(radius=radius)
    params = dict(radius=radius, t=t, mu=mu, j=j, delta=delta, azi_winding=azi_winding, radi_winding=radi_winding)
    
    spec = get_spectrum(sys, params)
    
    plot_opts=dict(width=350, height=350, invert_axes=True, framewise=True, \
                   default_tools=['ypan','wheel_zoom','reset', 'save'])
    style_opts=dict(color='tomato',alpha=0.5)
    
    if not toolbar:
        plot_opts['toolbar']='disable'
        
    fermi_level = hv.VLine(mu).opts(**plot_opts, color='black', line_dash='dashed', line_width=1.5)

    if smooth == True:
        bandwidth =  0.1 + round(15/radius)/40
        plot_opts['bandwidth'] = bandwidth
        distr = hv.Distribution(spec, kdims=dims['E']).opts(bandwidth=bandwidth).opts(plot=plot_opts, style=style_opts)
        plot = (distr*fermi_level)
    else:
        n_bins = int(9 + radius/2)
        hist = hv.Histogram(np.histogram(spec, bins=n_bins, density=True), kdims=dims['E'], vdims='Density').opts(plot=plot_opts, style=style_opts)
        plot = (hist*fermi_level)
        
    return plot.opts(title='Dos Finite System R={:d}'.format(radius))

def plot_density(sys, params, eigen, n):
    #sys = Kwant system, eigen=(eVal, eVec), n= number of wavefunction to plot
    eVal, eVec = eigen
    prob_dens = np.abs(eVec)**2
    
    vmax=np.max(prob_dens[:,n])

    fig, axes = plt.subplots(2,2, sharey=True, sharex=True, figsize=(6,6))

    plt.suptitle('Wavefunction to $\epsilon$={:.3E}'.format(eVal[n]), y=1,fontsize=16)
    
    axes[0,0].set_title('$c^{\\dagger}_{\\uparrow}$')
    kwant.plotter.density(sys, prob_dens[0::4, n], vmin=0, vmax=vmax, cmap='magma', ax = axes[0,0], background='#000000')
    axes[0,0].text(0.025, 1.05,'|ψ|={:.2f}'.format(np.sum(prob_dens[0::4, n])), \
         transform = axes[0,0].transAxes, bbox=dict(facecolor='white', alpha=0.5))

    axes[0,1].set_title('$c^{\\dagger}_{\\downarrow}$')
    kwant.plotter.density(sys, prob_dens[1::4, n], vmin=0, vmax=vmax, ax=axes[0,1], cmap='inferno',background='#000000')
    axes[0,1].text(0.025, 1.05,'|ψ|={:.2f}'.format(np.sum(prob_dens[1::4, n])), \
         transform = axes[0,1].transAxes, bbox=dict(facecolor='white', alpha=0.5))

    axes[1,0].set_title('$c^{}_{\\uparrow}$')
    kwant.plotter.density(sys, prob_dens[3::4, n], vmin=0, vmax=vmax, ax=axes[1,0], cmap='inferno',background='#000000')
    axes[1,0].text(0.025, 1.05,'|ψ|={:.2f}'.format(np.sum(prob_dens[3::4, n])), \
         transform = axes[1,0].transAxes, bbox=dict(facecolor='white', alpha=0.5))

    axes[1,1].set_title('$c^{}_{\\downarrow}$')
    kwant.plotter.density(sys, prob_dens[2::4, n], vmin=0, vmax=vmax, ax=axes[1,1], cmap='inferno',background='#000000')
    axes[1,1].text(0.025, 1.05,'|ψ|={:.2f}'.format(np.sum(prob_dens[2::4, n])), \
         transform = axes[1,1].transAxes, bbox=dict(facecolor='white', alpha=0.5))

    pars = params.copy()
    param_text= '\n'

    for key in pars:
        param_text = param_text + ' {} = {} \n'.format(dims[key], pars[key])    
    # place text boxes
    fig.text(0.95, 0.5, param_text, fontsize=14, horizontalalignment='left',\
         verticalalignment='center', bbox=dict(facecolor='grey', alpha=0.1))
    
    return fig

def solve_sparse(sys, params, k=32, sigma=0, timing=False, show_plot=True, save_plot_to = None):
	'''
    params: parameters used to build the kwant system
    
    save_to : needs to be a path 
    returns : First k sorted eigenpairs with eigenvalues around sigma. 
    '''
	ham = sys.hamiltonian_submatrix(params=params, sparse=True)   
    
	if timing:
		import time
		t1 = time.time()
		eigen = sla.eigsh(ham, k=k, sigma=sigma)
		t2 = time.time()
		print('Hamiltonian size = {0:d}x{0:d} \nSolving for {1:d} solutions around 0 took {2:.3f}s'.format(ham.shape[0],k,t2-t1))

	else:
		eigen = sla.eigsh(ham, k=k, sigma=sigma)
        
	p = eigen[0].argsort() #find the permutation that sorts the eigenvalues
	eVec= eigen[1][:,p] #sort the eigenvectors with that same permutation
	eVal = eigen[0][p]
    
	if show_plot or save_plot_to != None:
		from matplotlib import pyplot as plt
		plt.plot(eVal/params['delta'],'.')
		plt.xlim(0,len(eVal))
		plt.suptitle('Spectrum (k={})'.format(k))
		plt.ylabel('$\epsilon/\Delta$', fontsize=14)

		pars = params.copy()
		param_text= '\n'

		for key in pars:
			param_text = param_text + ' {} = {} \n'.format(dims[key], pars[key])    
		# place text boxes
		plt.text(1.05, 0.5, param_text, fontsize=14, horizontalalignment='left',transform = plt.gca().transAxes,\
			 verticalalignment='center', bbox=dict(facecolor='grey', alpha=0.1))

		if not save_plot_to==None:
			if not isinstance(save_plot_to, str):
				raise Exception('Path has to be a string')
			plt.savefig(save_plot_to + '\spectrum.png', dpi=250, bbox_inches='tight')        

		plt.show()
		plt.close()
        
	return (eVal, eVec)