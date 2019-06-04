import numpy as np 
from matplotlib import pyplot as plt
import kwant

dims = dict(kx = 'k_x',ky = 'k_y', mu = 'µ', delta = 'Δ', t = 't', E='ε',mu_t = 'µ/t',E_d='ε/Δ', \
			azi_winding='q', radi_winding= 'p', radius = 'r', R0 = 'R_0', j='j')
#
def rebuild_sys(params, shape= 'disk', plot=False):
	'''
	For now only disk-shape possible
	'''
	
	sys = kwant.Builder() #initialize your system
	sqlat = kwant.lattice.square(norbs=1)

	#define a Boolean function to shape your system

	def disk(position): 
		x,y = position
		return x**2 + y**2 < params['radius']**2

	if shape == 'disk':
		sys[sqlat.shape(disk,(0,0))]= 1

	if plot:
		kwant.plot(sys)
		plt.close()

	return sys.finalized()
#
def plot_spec(eVal, params, save_plot_to = None, dpi=250, show=True):
	fig = plt.figure()
	plt.plot(eVal/params['delta'],'.')
	plt.xlim(0,len(eVal))
	plt.suptitle('Spectrum (k={})'.format(len(eVal)))
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
		plt.savefig(save_plot_to + '\spectrum.png', dpi=dpi, bbox_inches='tight')        
	
	if show:
		plt.show()
	
	plt.close()
	
	return fig
#
def plot_dos(eVal, params, save_plot_to=None, dpi=250, n_bins=20, show=True):
	fig = plt.figure()
	e_d = eVal/params['delta']
	plt.hist(e_d, bins=np.linspace(-max(e_d), max(e_d),n_bins), align='mid')
	plt.axvline(0,color='grey')
	plt.suptitle('DoS (k={})'.format(len(eVal)))
	plt.xlabel('$\epsilon/\Delta$', fontsize=14)

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
		plt.savefig(save_plot_to + '\DoS.png', dpi=dpi, bbox_inches='tight')        
	
	if show:
		plt.show()
		
	plt.close()	
	
	return fig
	

import holoviews as hv
hv.extension('bokeh', logo=False)
from bokeh.plotting import show
#
def plot_dos_hv(eVal, params, smooth=False, toolbar=False, save_plot_to=None, n_bins=20):
	plot_opts=dict(width=350, height=350, invert_axes=True, framewise=True, \
	default_tools=['ypan','wheel_zoom','reset', 'save'])
	style_opts=dict(color='tomato',alpha=0.5)

	if not toolbar:
		plot_opts['toolbar']='disable'
		
	fermi_level = hv.VLine(params['mu']).opts(**plot_opts, color='grey', line_dash='dashed', line_width=1.5)
	if smooth == True:
		bandwidth =  0.09
		plot_opts['bandwidth'] = bandwidth
		hist = hv.Distribution(eVal/params['delta'], kdims=dims['E_d']).opts(plot=plot_opts, style=style_opts)
	else:
		e_d = eVal/params['delta']
		hist = hv.Histogram(np.histogram(e_d, bins=np.linspace(-max(e_d), max(e_d),n_bins), density=False), \
		kdims=dims['E_d'], vdims='Density').opts(plot=plot_opts, style=style_opts)
	plot = (hist*fermi_level)
	
	if not save_plot_to == None:
		if not isinstance(save_plot_to, str):
			raise Exception('Path has to be a string')
		hv.save(plot, save_plot_to + '\DoS_hv.png') 
		
	return plot.opts(title='DoS (k={:d})'.format(len(eVal)))
#
def plot_density(sys, dens, params, eVal = None, save_plot_to = None, dpi=50, show=True):

	#sys = Kwant system, eVal is associated eigenvalue to probability_density
	prob_dens = dens

	vmax=np.max(prob_dens)

	fig, axes = plt.subplots(2,2, sharey=True, sharex=True, figsize=(6,6))

	if eVal is not None:
		plt.suptitle('Wavefunction to $\epsilon$={:.3E}'.format(eVal), y=1,fontsize=16)
	
	plot_opts= dict(vmin=0, vmax=vmax, cmap='inferno', background='#000000')

	axes[0,0].set_title('$|c^{}_{\\uparrow}|^2$')
	kwant.plotter.density(sys, prob_dens[0::4], ax=axes[0,0], **plot_opts)
	axes[0,0].text(0.025, 1.05,'$\\rho$'+'={:.2f}'.format(np.sum(prob_dens[0::4])), \
		transform = axes[0,0].transAxes,  bbox=dict(facecolor='white', alpha=0.5))

	axes[0,1].set_title('$|c^{}_{\\downarrow}|^2$')
	kwant.plotter.density(sys, prob_dens[1::4], ax=axes[0,1], **plot_opts )
	axes[0,1].text(0.025, 1.05,'$\\rho$'+'={:.2f}'.format(np.sum(prob_dens[1::4])), \
		 transform = axes[0,1].transAxes, bbox=dict(facecolor='white', alpha=0.5))

	axes[1,0].set_title('$|c^{\\dagger}_{\\uparrow}|^2$')
	kwant.plotter.density(sys, prob_dens[3::4], ax=axes[1,0], **plot_opts)
	axes[1,0].text(0.025, 1.05,'$\\rho$'+'={:.2f}'.format(np.sum(prob_dens[3::4])), \
		 transform = axes[1,0].transAxes, bbox=dict(facecolor='white', alpha=0.5))

	axes[1,1].set_title('$|c^{\\dagger}_{\\downarrow}|^2$')
	kwant.plotter.density(sys, prob_dens[2::4], ax=axes[1,1], **plot_opts)
	axes[1,1].text(0.025, 1.05,'$\\rho$'+'={:.2f}'.format(np.sum(prob_dens[2::4])), \
		 transform = axes[1,1].transAxes, bbox=dict(facecolor='white', alpha=0.5))

	pars = params.copy()
	param_text= '\n'

	for key in pars:
		param_text = param_text + ' {} = {} \n'.format(dims[key], pars[key])    
	# place text boxes
	fig.text(0.95, 0.5, param_text, fontsize=14, horizontalalignment='left',\
		 verticalalignment='center', bbox=dict(facecolor='grey', alpha=0.1))

	
	if not save_plot_to==None:
		if not isinstance(save_plot_to, str):
			raise Exception('Path has to be a string')
		if eVal is not None:
			fig.savefig(save_plot_to + '__e={:.2E}'.format(eVal), format='png', bbox_inches='tight', dpi=dpi, metadata=params)
		else:
			fig.savefig(save_plot_to, bbox_inches='tight', format='png', dpi=dpi, metadata=params)
	
	if show:
		plt.show()
	
	plt.close()
	return fig
	
#
def plot_skyrmion_kwant(sys, params, ham=None):
	#Needs finalized kwant system
	if ham==None:
		ham= sys.hamiltonian_submatrix(params=params, sparse=True)
	
	b_z = np.real(ham.diagonal()[::4])-(4*params['t']-params['mu']) #Since our function now only contains the magnetic term, we take out the Bz component to check
	b_x = np.real(ham.diagonal(k=1)[::4])
	b_y = np.imag(ham.diagonal(k=1)[::4])
	opts = dict(cmap='cividis', oversampling=1, num_lead_cells=2, vmin=-params['j'], vmax=params['j'])
	
	fig, ax = plt.subplots(1,3, sharey=True)
	ax[0].set_title('$n_x$')
	kwant.plotter.map(sys, b_x, **opts, ax=ax[0])
	ax[1].set_title('$n_y$')
	kwant.plotter.map(sys, b_y, **opts, ax=ax[1])
	ax[2].set_title('$n_z$')
	kwant.plotter.map(sys, b_z, **opts, ax=ax[2])
	plt.close()

	return fig

#
def sum_densities(eigen, n=10, plot=True):
    #please make sure the eigenpairs are already sorted (for example by using functions.solve_sparse when solving system)
    eVals, eVecs = eigen
    arg_min_e = np.argmin(np.abs(eVals)) #find index of eigenvalue closest to zero
    
    if isinstance(n,int):
        n_vecs = np.arange(arg_min_e-n//2,np.ceil(arg_min_e+n/2), dtype=int)
    elif isinstance(n,(list,np.ndarray)):
        n_vecs = n       
    else:
        raise Exception("The argument n has to be a list, array or integer. If integer, the lowest n states (absolute eigenvalue) will be taken")    
    
    
    summed = np.zeros_like(eVecs[:,0], dtype=float)

    for n in n_vecs:
        summed = np.add(summed, np.abs(eVecs[:,n])**2)
   
    return summed
    

#